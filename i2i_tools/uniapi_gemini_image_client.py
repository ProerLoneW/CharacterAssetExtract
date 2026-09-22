#!/usr/bin/env python3
"""UniAPI Gemini-native client for multi-reference image restoration.

This client uses Gemini's ``generateContent`` protocol: local reference images
are embedded as ordered ``inlineData`` parts, the restoration instructions are
sent as a text part, and ``responseModalities`` explicitly requests an image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .uniapi_image_client import (
        InputImage,
        SavedImage,
        UniAPIImageError,
        _atomic_write,
        _extract_api_error,
        _numbered_output_paths,
        _read_user_prompt,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
    )
except ImportError:  # Allow direct script imports from i2i_tools/.
    from uniapi_image_client import (
        InputImage,
        SavedImage,
        UniAPIImageError,
        _atomic_write,
        _extract_api_error,
        _numbered_output_paths,
        _read_user_prompt,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
    )


DEFAULT_GEMINI_BASE_URL = "https://api.uniapi.io/gemini/v1beta"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3-pro-image"
MAX_REFERENCE_IMAGES = 14
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
MAX_ERROR_BODY_BYTES = 16 * 1024
VALID_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
}
VALID_IMAGE_SIZES = {"1K", "2K", "4K"}


@dataclass(frozen=True)
class GeminiImageModelProfile:
    label: str
    use_case: str
    default_image_size: str | None
    supported_image_sizes: frozenset[str]
    reference_guidance: str


GEMINI_IMAGE_MODEL_PROFILES: dict[str, GeminiImageModelProfile] = {
    "gemini-3-pro-image": GeminiImageModelProfile(
        label="Nano Banana Pro",
        use_case="Highest fidelity for professional assets and complex instructions",
        default_image_size="2K",
        supported_image_sizes=frozenset({"1K", "2K", "4K"}),
        reference_guidance="Up to 14 total; favor <=5 character or <=6 object references",
    ),
    "gemini-3.1-flash-image": GeminiImageModelProfile(
        label="Nano Banana 2",
        use_case="Best quality, speed, and cost balance for production batches",
        default_image_size="2K",
        supported_image_sizes=frozenset({"1K", "2K", "4K"}),
        reference_guidance="Up to 14 total; up to 4 characters or 10 objects at high fidelity",
    ),
    "gemini-3.1-flash-lite-image": GeminiImageModelProfile(
        label="Nano Banana 2 Lite",
        use_case="Lowest latency and cost for previews and large-scale screening",
        default_image_size="1K",
        supported_image_sizes=frozenset({"1K"}),
        reference_guidance="Up to 14 object references; use only the clearest inputs",
    ),
    "gemini-2.5-flash-image": GeminiImageModelProfile(
        label="Nano Banana",
        use_case="Legacy fast image generation and editing at fixed 1K resolution",
        default_image_size=None,
        supported_image_sizes=frozenset(),
        reference_guidance="Works best with no more than 3 reference images",
    ),
}


@dataclass(frozen=True)
class GeminiRestorationResult:
    final_prompt: str
    input_images: list[InputImage]
    saved_images: list[SavedImage]
    response_text: list[str]
    response: dict[str, Any]


def _generate_content_url(base_url: str, model: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if not model.strip():
        raise ValueError("model must not be empty")
    encoded_model = urllib.parse.quote(model.strip(), safe="-._")
    if normalized.endswith(":generateContent"):
        return normalized
    if normalized.endswith("/models"):
        return f"{normalized}/{encoded_model}:generateContent"
    return f"{normalized}/models/{encoded_model}:generateContent"


def build_generate_content_payload(
    images: Sequence[InputImage],
    prompt: str,
    *,
    aspect_ratio: str,
    image_size: str | None,
    include_text_response: bool = True,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build a Gemini native image+text request in reference-image order."""
    if not images:
        raise ValueError("At least one input image is required")
    if len(images) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Gemini image generation accepts at most {MAX_REFERENCE_IMAGES} reference images"
        )
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
    if image_size is not None and image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"Unsupported image size: {image_size}")
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")

    parts: list[dict[str, Any]] = []
    for image in images:
        encoded = base64.b64encode(image.path.read_bytes()).decode("ascii")
        parts.append(
            {
                "inlineData": {
                    "mimeType": image.content_type,
                    "data": encoded,
                }
            }
        )
    parts.append({"text": prompt.strip()})

    image_config: dict[str, Any] = {"aspectRatio": aspect_ratio}
    if image_size is not None:
        image_config["imageSize"] = image_size
    generation_config: dict[str, Any] = {
        "responseModalities": ["TEXT", "IMAGE"] if include_text_response else ["IMAGE"],
        "imageConfig": image_config,
    }
    if temperature is not None:
        generation_config["temperature"] = temperature
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def resolve_model_image_size(model: str, requested_size: str | None) -> str | None:
    """Resolve ``auto`` and reject explicit sizes unsupported by known models."""
    normalized_size = requested_size or "auto"
    if normalized_size != "auto" and normalized_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"Unsupported image size: {normalized_size}")
    profile = GEMINI_IMAGE_MODEL_PROFILES.get(model)
    if profile is None:
        return "2K" if normalized_size == "auto" else normalized_size
    if normalized_size == "auto":
        return profile.default_image_size
    if normalized_size not in profile.supported_image_sizes:
        supported = ", ".join(sorted(profile.supported_image_sizes)) or "fixed 1K only"
        raise ValueError(f"Model {model} does not accept image size {normalized_size}; use {supported}")
    return normalized_size


class UniAPIGeminiImageClient:
    """Call Gemini native ``generateContent`` through UniAPI."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        timeout_seconds: float = 240.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("A non-empty UniAPI API key is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        *,
        image_paths: Sequence[str | Path],
        prompt: str,
        model: str = DEFAULT_GEMINI_IMAGE_MODEL,
        aspect_ratio: str = "1:1",
        image_size: str | None = "auto",
        include_text_response: bool = True,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], list[InputImage]]:
        images = inspect_images(image_paths)
        resolved_image_size = resolve_model_image_size(model, image_size)
        payload = build_generate_content_payload(
            images,
            prompt,
            aspect_ratio=aspect_ratio,
            image_size=resolved_image_size,
            include_text_response=include_text_response,
            temperature=temperature,
        )
        endpoint = _generate_content_url(self.base_url, model)
        response = self._request_with_retries(endpoint, payload)
        return response, images

    def _request_with_retries(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_error: UniAPIImageError | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "general-video-annotation-i2i/2.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    request_id = response.headers.get("x-request-id")
                try:
                    decoded = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise UniAPIImageError(
                        "UniAPI returned a non-JSON success response",
                        request_id=request_id,
                    ) from exc
                if not isinstance(decoded, dict):
                    raise UniAPIImageError("UniAPI response must be a JSON object")
                if isinstance(decoded.get("error"), dict):
                    detail = _extract_api_error(raw)
                    raise UniAPIImageError(
                        f"UniAPI returned an API error: {detail}",
                        request_id=request_id,
                    )
                return decoded
            except urllib.error.HTTPError as exc:
                raw = exc.read(MAX_ERROR_BODY_BYTES)
                request_id = exc.headers.get("x-request-id") if exc.headers else None
                retryable = exc.code in RETRYABLE_STATUS_CODES
                last_error = UniAPIImageError(
                    f"UniAPI HTTP {exc.code}: {_extract_api_error(raw)}",
                    status_code=exc.code,
                    request_id=request_id,
                    retryable=retryable,
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = exc.reason if hasattr(exc, "reason") else exc
                last_error = UniAPIImageError(
                    f"UniAPI network error: {reason}",
                    retryable=True,
                )
            if not last_error.retryable or attempt >= self.max_retries:
                raise last_error
            delay = min(20.0, (2**attempt) + random.uniform(0.0, 0.5))
            time.sleep(delay)
        raise last_error or UniAPIImageError("UniAPI request failed")


def extract_gemini_outputs(response: Mapping[str, Any]) -> tuple[list[tuple[bytes, str]], list[str]]:
    """Extract final (non-thinking) image and text parts from candidates."""
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise UniAPIImageError("Gemini response contains no candidates[]")

    images: list[tuple[bytes, str]] = []
    texts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline_data, dict):
                continue
            encoded = inline_data.get("data")
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type")
            if not isinstance(encoded, str) or not encoded:
                continue
            if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
                raise UniAPIImageError("Gemini returned inline data that is not an image")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise UniAPIImageError("Gemini returned invalid base64 image data") from exc
            if not decoded:
                raise UniAPIImageError("Gemini returned an empty image")
            images.append((decoded, mime_type))

    if not images:
        reasons = [
            item.get("finishReason")
            for item in candidates
            if isinstance(item, dict) and item.get("finishReason")
        ]
        detail = f"; finish reasons: {', '.join(map(str, reasons))}" if reasons else ""
        raise UniAPIImageError(f"Gemini response contains no final image parts{detail}")
    return images, texts


def save_gemini_response_images(
    response: Mapping[str, Any],
    output_path: str | Path,
    *,
    input_paths: Sequence[str | Path] = (),
    overwrite: bool = False,
) -> tuple[list[SavedImage], list[str]]:
    images, texts = extract_gemini_outputs(response)
    paths = _numbered_output_paths(Path(output_path).expanduser().resolve(), len(images))
    _validate_output_targets(input_paths, paths)
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    saved: list[SavedImage] = []
    for (content, mime_type), path in zip(images, paths):
        # Gemini image models commonly return image/jpeg even when asked for a
        # PNG; treat any image/* as acceptable and write the bytes verbatim.
        # Downstream code (e.g. extract_character_clothing) re-encodes through
        # Pillow and applies alpha-bake before declaring the asset final.
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise UniAPIImageError(
                f"Gemini returned non-image mime type {mime_type!r}"
            )
        _atomic_write(path, content, overwrite=overwrite)
        saved.append(
            SavedImage(
                path=path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                revised_prompt=None,
            )
        )
    return saved, texts


def _normalize_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if not path.suffix:
        return path.with_suffix(".png")
    if path.suffix.lower() != ".png":
        raise ValueError("Gemini native image output must use a .png output path")
    return path


def restore_gemini_reference(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    user_prompt: str,
    output_path: str | Path,
    task_instructions: str,
    asset_description: str | None = None,
    overwrite: bool = False,
    generation_options: Mapping[str, Any] | None = None,
) -> GeminiRestorationResult:
    """Restore one asset using ordered image parts plus a text prompt."""
    options = dict(generation_options or {})
    resolved_output = _normalize_output_path(output_path)
    _validate_output_targets(image_paths, [resolved_output])
    final_prompt = compose_prompt(
        task_instructions,
        user_prompt,
        asset_description=asset_description,
    )
    response, inputs = client.generate(
        image_paths=image_paths,
        prompt=final_prompt,
        **options,
    )
    saved, texts = save_gemini_response_images(
        response,
        resolved_output,
        input_paths=image_paths,
        overwrite=overwrite,
    )
    return GeminiRestorationResult(
        final_prompt=final_prompt,
        input_images=inputs,
        saved_images=saved,
        response_text=texts,
        response=response,
    )


def build_gemini_task_parser(
    description: str,
    *,
    default_aspect_ratio: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="IMAGE",
        help="Ordered references; put the strongest/main reference first (maximum 14)",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Additional restoration instructions")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 prompt text file")
    parser.add_argument("--asset-description", help="Known identity/layout/object details")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--model",
        default=os.getenv("UNIAPI_GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL),
        help=(
            "Gemini image model ID; known choices: "
            + ", ".join(GEMINI_IMAGE_MODEL_PROFILES)
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("UNIAPI_GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
    )
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument(
        "--aspect-ratio",
        default=default_aspect_ratio,
        choices=sorted(VALID_ASPECT_RATIOS),
    )
    parser.add_argument(
        "--image-size",
        default="auto",
        choices=["auto", *sorted(VALID_IMAGE_SIZES)],
        help="Default: auto (2K for Pro/Flash, 1K for Lite, omitted for the fixed-1K legacy model)",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Request IMAGE only instead of TEXT+IMAGE",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate without reading the key or calling API")
    return parser


def run_gemini_task_cli(
    *,
    task_name: str,
    task_instructions: str,
    default_aspect_ratio: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_gemini_task_parser(
        f"Restore a high-quality {task_name} reference with Gemini native multimodal generation",
        default_aspect_ratio=default_aspect_ratio,
    )
    args = parser.parse_args(argv)
    try:
        user_prompt = _read_user_prompt(args)
        if not user_prompt:
            raise ValueError("Prompt must not be empty")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if args.max_retries < 0:
            raise ValueError("--max-retries must be non-negative")
        if args.temperature is not None and not 0 <= args.temperature <= 2:
            raise ValueError("--temperature must be between 0 and 2")

        inputs = inspect_images(args.images)
        if len(inputs) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Gemini image generation accepts at most {MAX_REFERENCE_IMAGES} reference images"
            )
        output_path = _normalize_output_path(args.output).resolve()
        prompt = compose_prompt(
            task_instructions,
            user_prompt,
            asset_description=args.asset_description,
        )
        endpoint = _generate_content_url(args.base_url, args.model)
        resolved_image_size = resolve_model_image_size(args.model, args.image_size)
        model_profile = GEMINI_IMAGE_MODEL_PROFILES.get(args.model)
        request_summary = {
            "task": task_name,
            "protocol": "gemini-native-generateContent",
            "endpoint": endpoint,
            "model": args.model,
            "images": [
                {
                    "index": index,
                    "path": str(image.path),
                    "mime_type": image.content_type,
                    "size_bytes": image.size_bytes,
                    "sha256": image.sha256,
                }
                for index, image in enumerate(inputs, start=1)
            ],
            "prompt": prompt,
            "response_modalities": ["IMAGE"] if args.image_only else ["TEXT", "IMAGE"],
            "aspect_ratio": args.aspect_ratio,
            "image_size_requested": args.image_size,
            "image_size": resolved_image_size,
            "model_profile": (
                {
                    "label": model_profile.label,
                    "use_case": model_profile.use_case,
                    "reference_guidance": model_profile.reference_guidance,
                }
                if model_profile
                else None
            ),
            "temperature": args.temperature,
            "output": str(output_path),
        }
        metadata_path = None
        if not args.no_metadata:
            metadata_path = (
                args.metadata_output.expanduser().resolve()
                if args.metadata_output
                else output_path.with_suffix(output_path.suffix + ".json")
            )
        _validate_output_targets(args.images, [output_path], metadata_path=metadata_path)
        if args.dry_run:
            print(json.dumps(request_summary, ensure_ascii=False, indent=2))
            return 0

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        if metadata_path is not None and metadata_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {metadata_path}")
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {args.api_key_env} is not set; API keys are intentionally not accepted on the command line"
            )

        client = UniAPIGeminiImageClient(
            api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        response, _ = client.generate(
            image_paths=args.images,
            prompt=prompt,
            model=args.model,
            aspect_ratio=args.aspect_ratio,
            image_size=resolved_image_size,
            include_text_response=not args.image_only,
            temperature=args.temperature,
        )
        saved, response_text = save_gemini_response_images(
            response,
            output_path,
            input_paths=args.images,
            overwrite=args.overwrite,
        )
        metadata = {
            "schema_version": "uniapi-gemini-i2i-asset-v2",
            "request": request_summary,
            "response": {
                "model_version": response.get("modelVersion"),
                "usage": response.get("usageMetadata"),
                "text": response_text,
                "finish_reasons": [
                    candidate.get("finishReason")
                    for candidate in response.get("candidates", [])
                    if isinstance(candidate, dict) and candidate.get("finishReason")
                ],
            },
            "outputs": [
                {
                    "path": str(item.path),
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in saved
            ],
        }
        if metadata_path is not None:
            encoded = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            _atomic_write(metadata_path, encoded, overwrite=args.overwrite)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, UniAPIImageError) as exc:
        request_id = getattr(exc, "request_id", None)
        suffix = f" (request_id={request_id})" if request_id else ""
        print(f"error: {exc}{suffix}", file=sys.stderr)
        return 2
