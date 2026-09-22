#!/usr/bin/env python3
"""UniAPI client for Volcengine ARK Doubao Seedream image generation.

Endpoint surface used here is the OpenAI-compatible ``/v1/images/generations``
JSON route, accessed through the UniAPI proxy (``https://api.uniapi.io/v1``)
which fronts the upstream ``ark.cn-beijing.volces.com/api/v3`` endpoint.

Differences from :mod:`uniapi_image_client` (OpenAI /v1/images/edits):

* Request body is JSON, not multipart; reference images are passed as URLs or
  ``data:image/<mime>;base64,…`` strings under the ``image`` key (list, up to
  ~14 inputs).
* Output sizes are enumerated (``2K`` / ``3K`` / explicit ``WxH``) rather than
  the OpenAI ``1024x1024`` family.
* ``moderation`` / ``input_fidelity`` / ``background`` are NOT supported; we
  never send them and the proxy rejects unknown fields.
* The proxy may return ``image/jpeg`` (RGB) even when ``output_format=png`` is
  set; downstream alpha-bake is the safe path to true transparency.

Supported models:

* ``doubao-seedream-5-0-260128`` (Seedream 5.0 lite, default)
* ``doubao-seedream-5-0-lite-260128`` (older 5.0 lite variant)
* ``doubao-seedream-4-5-251128`` (Seedream 4.5)
* ``doubao-seedream-4-0-250828`` (Seedream 4.0)
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
        _download,
        _extract_api_error,
        _numbered_output_paths,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
    )
except ImportError:  # pragma: no cover - script import path
    from uniapi_image_client import (
        InputImage,
        SavedImage,
        UniAPIImageError,
        _atomic_write,
        _download,
        _extract_api_error,
        _numbered_output_paths,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
    )


DEFAULT_DOUBEO_BASE_URL = "https://api.uniapi.io/v1"
DEFAULT_DOUBEO_MODEL = "doubao-seedream-5-0-260128"
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
MAX_ERROR_BODY_BYTES = 16 * 1024
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_REFERENCE_IMAGES = 14


@dataclass(frozen=True)
class DoubaoImageModelProfile:
    label: str
    use_case: str
    default_size: str
    supported_sizes: frozenset[str]
    reference_guidance: str


DOUBAO_IMAGE_MODEL_PROFILES: dict[str, DoubaoImageModelProfile] = {
    "doubao-seedream-5-0-260128": DoubaoImageModelProfile(
        label="Doubao Seedream 5.0 lite (2026-01-28)",
        use_case="Latest Seedream 5.0 lite; best quality/cost on the ARK endpoint.",
        default_size="2K",
        supported_sizes=frozenset({"2K", "3K", "1024x1024", "1536x2048", "2048x2048"}),
        reference_guidance="Up to 14 reference images (URL or base64). Min total pixels = 3686400 (~1920x1920).",
    ),
    "doubao-seedream-5-0-lite-260128": DoubaoImageModelProfile(
        label="Doubao Seedream 5.0 lite (variant)",
        use_case="Alias for the older 5.0 lite model name.",
        default_size="2K",
        supported_sizes=frozenset({"2K", "3K", "1024x1024", "1536x2048", "2048x2048"}),
        reference_guidance="Up to 14 reference images (URL or base64). Min total pixels = 3686400.",
    ),
    "doubao-seedream-4-5-251128": DoubaoImageModelProfile(
        label="Doubao Seedream 4.5",
        use_case="Seedream 4.5; supports 2K and 4K output.",
        default_size="2K",
        supported_sizes=frozenset({"2K", "4K", "1024x1024", "1536x2048", "2048x2048", "4096x4096"}),
        reference_guidance="Up to 14 reference images (URL or base64). Min total pixels = 3686400.",
    ),
    "doubao-seedream-4-0-250828": DoubaoImageModelProfile(
        label="Doubao Seedream 4.0",
        use_case="Seedream 4.0; supports 1K, 2K and 4K output.",
        default_size="2K",
        supported_sizes=frozenset({"1K", "2K", "4K", "1024x1024", "1536x2048", "2048x2048", "4096x4096"}),
        reference_guidance="Up to 14 reference images (URL or base64). Min total pixels = 3686400.",
    ),
}


SUPPORTED_DOUBEAO_MODELS: tuple[str, ...] = tuple(sorted(DOUBAO_IMAGE_MODEL_PROFILES))


def resolve_doubao_model_size(model: str, requested_size: str | None) -> str:
    """Resolve ``auto`` and reject explicit sizes unsupported by known models."""

    profile = DOUBAO_IMAGE_MODEL_PROFILES.get(model)
    if profile is None:
        return requested_size or "2K"
    requested = requested_size or "auto"
    if requested == "auto":
        return profile.default_size
    if requested not in profile.supported_sizes:
        supported = ", ".join(sorted(profile.supported_sizes))
        raise ValueError(
            f"Doubao model {model} does not accept size {requested}; use one of: {supported}"
        )
    return requested


# Doubao requires every image to be at least 3686400 pixels (~1920x1920) per
# request. Aliases like "2K" map to a square size that meets the floor, but
# callers who explicitly pass a smaller WxH would otherwise hit the API-side
# "image size must be at least 3686400 pixels" 400. Validate the floor client
# side so we get a clear error up front instead of opaque 400s.
import re as _re

_SIZE_RE = _re.compile(r"^(\d+)x(\d+)$")


def _assert_meets_doubao_pixel_floor(size: str) -> None:
    match = _SIZE_RE.match(size)
    if not match:
        return  # Preset like "2K" / "3K" — server resolves the actual size.
    width, height = int(match.group(1)), int(match.group(2))
    if width * height < 3_686_400:
        raise ValueError(
            f"Doubao requires at least 3686400 pixels per request; "
            f"got {width}x{height} = {width * height}. "
            f"Use a preset like '2K' / '3K' or pass a larger WxH."
        )


def _encode_image_for_doubao(image: InputImage) -> str:
    """Encode a local image as a base64 data URI accepted by the ARK endpoint."""
    encoded = base64.b64encode(image.path.read_bytes()).decode("ascii")
    return f"data:{image.content_type};base64,{encoded}"


def _generations_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if normalized.endswith("/images/generations"):
        return normalized
    return f"{normalized}/images/generations"


@dataclass(frozen=True)
class DoubaoRestorationResult:
    """Programmatic counterpart to :class:`RestorationResult`."""

    final_prompt: str
    input_images: list[InputImage]
    saved_images: list[SavedImage]
    response_text: list[str]
    response: dict[str, Any]


class UniAPIDoubaoImageClient:
    """Call ARK Doubao ``/v1/images/generations`` via the UniAPI proxy.

    Sends a JSON request body with one or more ``image`` entries (base64 data
    URI per local image), the model id, prompt, size, watermark flag, and
    response_format. Decodes ``b64_json`` results into local files.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_DOUBEO_BASE_URL,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("A non-empty UniAPI API key is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_key = api_key.strip()
        self.endpoint = _generations_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        *,
        image_paths: Sequence[str | Path],
        prompt: str,
        model: str = DEFAULT_DOUBEO_MODEL,
        size: str = "2K",
        response_format: str = "b64_json",
        output_format: str = "png",
        watermark: bool = False,
        sequential_image_generation: str = "disabled",
        n: int = 1,
        seed: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[InputImage]]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if response_format not in {"b64_json", "url"}:
            raise ValueError("response_format must be 'b64_json' or 'url'")
        if output_format not in {"png", "jpeg"}:
            raise ValueError("output_format must be 'png' or 'jpeg'")
        if sequential_image_generation not in {"auto", "disabled"}:
            raise ValueError("sequential_image_generation must be 'auto' or 'disabled'")
        if n < 1:
            raise ValueError("n must be at least 1")

        images = inspect_images(image_paths)
        if len(images) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Doubao accepts at most {MAX_REFERENCE_IMAGES} reference images"
            )
        resolved_size = resolve_doubao_model_size(model, size)
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": resolved_size,
            "response_format": response_format,
            "output_format": output_format,
            "watermark": watermark,
            "sequential_image_generation": sequential_image_generation,
            "n": n,
        }
        if images:
            body["image"] = [_encode_image_for_doubao(img) for img in images]
        if seed is not None:
            body["seed"] = seed
        if extra_body:
            body.update({key: value for key, value in extra_body.items() if value is not None})

        payload_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = self._request_with_retries(payload_bytes)
        return response, images

    def _request_with_retries(self, body: bytes) -> dict[str, Any]:
        last_error: UniAPIImageError | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "general-video-annotation-i2i/3.0",
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
                        "Doubao returned a non-JSON success response",
                        request_id=request_id,
                    ) from exc
                if not isinstance(decoded, dict):
                    raise UniAPIImageError("Doubao response must be a JSON object")
                if isinstance(decoded.get("error"), dict):
                    detail = _extract_api_error(raw)
                    raise UniAPIImageError(
                        f"Doubao returned an API error: {detail}",
                        request_id=request_id,
                    )
                return decoded
            except urllib.error.HTTPError as exc:
                raw = exc.read(MAX_ERROR_BODY_BYTES)
                request_id = exc.headers.get("x-request-id") if exc.headers else None
                retryable = exc.code in RETRYABLE_STATUS_CODES
                last_error = UniAPIImageError(
                    f"Doubao HTTP {exc.code}: {_extract_api_error(raw)}",
                    status_code=exc.code,
                    request_id=request_id,
                    retryable=retryable,
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = exc.reason if hasattr(exc, "reason") else exc
                last_error = UniAPIImageError(
                    f"Doubao network error: {reason}",
                    retryable=True,
                )
            if not last_error.retryable or attempt >= self.max_retries:
                raise last_error
            delay = min(20.0, (2**attempt) + random.uniform(0.0, 0.5))
            time.sleep(delay)
        raise last_error or UniAPIImageError("Doubao request failed")


def save_doubao_response_images(
    response: Mapping[str, Any],
    output_path: str | Path,
    *,
    input_paths: Sequence[str | Path] = (),
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
) -> tuple[list[SavedImage], list[str]]:
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise UniAPIImageError("Doubao response contains no images in data[]")
    paths = _numbered_output_paths(Path(output_path).expanduser().resolve(), len(data))
    _validate_output_targets(input_paths, paths)
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    saved: list[SavedImage] = []
    for item, path in zip(data, paths):
        if not isinstance(item, dict):
            raise UniAPIImageError("Invalid item in Doubao data[]")
        encoded = item.get("b64_json")
        url = item.get("url")
        if isinstance(encoded, str) and encoded:
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise UniAPIImageError("Doubao returned invalid base64 image data") from exc
        elif isinstance(url, str) and url:
            content = _download(url, timeout_seconds)
        else:
            raise UniAPIImageError("Doubao image item has neither b64_json nor url")
        if not content:
            raise UniAPIImageError("Doubao returned an empty image")
        _atomic_write(path, content, overwrite=overwrite)
        saved.append(
            SavedImage(
                path=path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                revised_prompt=item.get("revised_prompt"),
            )
        )
    return saved, []


def restore_doubao_reference(
    client: UniAPIDoubaoImageClient,
    *,
    image_paths: Sequence[str | Path],
    user_prompt: str,
    output_path: str | Path,
    task_instructions: str,
    asset_description: str | None = None,
    overwrite: bool = False,
    generation_options: Mapping[str, Any] | None = None,
) -> DoubaoRestorationResult:
    """Programmatic counterpart to the task CLIs."""
    options = dict(generation_options or {})
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
    saved, texts = save_doubao_response_images(
        response,
        output_path,
        input_paths=image_paths,
        overwrite=overwrite,
        timeout_seconds=client.timeout_seconds,
    )
    return DoubaoRestorationResult(
        final_prompt=final_prompt,
        input_images=inputs,
        saved_images=saved,
        response_text=texts,
        response=response,
    )


def build_doubao_task_parser(
    description: str,
    *,
    default_size: str = "2K",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="IMAGE",
        help=(
            "Reference images (URL or local file); put the strongest/main "
            "reference first (maximum 14)."
        ),
    )
    parser.add_argument("--prompt", help="Additional restoration instructions")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 prompt text file")
    parser.add_argument("--asset-description", help="Known identity/layout/object details")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--model",
        default=os.getenv("UNIAPI_DOUBEO_IMAGE_MODEL", DEFAULT_DOUBEO_MODEL),
        help="Doubao model id; known choices: " + ", ".join(SUPPORTED_DOUBEAO_MODELS),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("UNIAPI_DOUBEO_BASE_URL", DEFAULT_DOUBEO_BASE_URL),
    )
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument(
        "--size",
        default=default_size,
        help="Output size: 2K / 3K / explicit WxH. Default: 2K.",
    )
    parser.add_argument(
        "--watermark",
        choices=["yes", "no"],
        default="no",
        help="Add the Volcengine AI watermark to the output image.",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "jpeg"],
        default="png",
        help="Output container format. Default: png (still JPEG-decoded by the proxy in some cases).",
    )
    parser.add_argument(
        "--response-format",
        choices=["b64_json", "url"],
        default="b64_json",
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_doubao_task_cli(
    *,
    task_name: str,
    task_instructions: str,
    argv: Sequence[str] | None = None,
    default_size: str = "2K",
) -> int:
    parser = build_doubao_task_parser(
        f"Restore / generate via Doubao Seedream on the ARK endpoint (default model {DEFAULT_DOUBEO_MODEL})",
        default_size=default_size,
    )
    args = parser.parse_args(argv)
    try:
        if args.prompt_file is not None:
            user_prompt = args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
        else:
            user_prompt = (args.prompt or "").strip()
        if not user_prompt:
            raise ValueError("Prompt must not be empty")
        if args.n < 1:
            raise ValueError("--n must be at least 1")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if args.max_retries < 0:
            raise ValueError("--max-retries must be non-negative")

        inputs = inspect_images(args.images)
        if len(inputs) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Doubao accepts at most {MAX_REFERENCE_IMAGES} reference images"
            )
        resolved_size = resolve_doubao_model_size(args.model, args.size)
        final_prompt = compose_prompt(
            task_instructions,
            user_prompt,
            asset_description=args.asset_description,
        )
        endpoint = _generations_url(args.base_url)
        output_path = args.output.expanduser().resolve()
        if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            output_path = output_path.with_suffix(".png")
        metadata_path = None
        if not args.no_metadata:
            metadata_path = (
                args.metadata_output.expanduser().resolve()
                if args.metadata_output
                else output_path.with_suffix(output_path.suffix + ".json")
            )
        _validate_output_targets(args.images, [output_path], metadata_path=metadata_path)

        request_summary = {
            "task": task_name,
            "protocol": "doubao-ark-generations",
            "endpoint": endpoint,
            "model": args.model,
            "size_requested": args.size,
            "size_resolved": resolved_size,
            "watermark": args.watermark == "yes",
            "output_format": args.output_format,
            "response_format": args.response_format,
            "n": args.n,
            "seed": args.seed,
            "inputs": [
                {
                    "index": index,
                    "path": str(image.path),
                    "size_bytes": image.size_bytes,
                    "sha256": image.sha256,
                }
                for index, image in enumerate(inputs, start=1)
            ],
            "prompt": final_prompt,
            "output": str(output_path),
        }
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

        client = UniAPIDoubaoImageClient(
            api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        response, _ = client.generate(
            image_paths=args.images,
            prompt=final_prompt,
            model=args.model,
            size=resolved_size,
            response_format=args.response_format,
            output_format=args.output_format,
            watermark=(args.watermark == "yes"),
            n=args.n,
            seed=args.seed,
        )
        saved, _ = save_doubao_response_images(
            response,
            output_path,
            input_paths=args.images,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout,
        )
        metadata = {
            "schema_version": "uniapi-doubao-i2i-asset-v1",
            "request": request_summary,
            "response": {
                "created": response.get("created"),
                "usage": response.get("usage"),
                "model": response.get("model"),
            },
            "outputs": [
                {
                    "path": str(item.path),
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "revised_prompt": item.revised_prompt,
                }
                for item in saved
            ],
        }
        if metadata_path is not None:
            encoded_metadata = (
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            _atomic_write(metadata_path, encoded_metadata, overwrite=args.overwrite)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, UniAPIImageError) as exc:
        request_id = getattr(exc, "request_id", None)
        suffix = f" (request_id={request_id})" if request_id else ""
        print(f"error: {exc}{suffix}", file=sys.stderr)
        return 2
