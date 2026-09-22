#!/usr/bin/env python3
"""Small dependency-free client for UniAPI's OpenAI-compatible image edit API.

The edit endpoint is used instead of the generation endpoint because reference
images are uploaded as multipart data.  Multiple references are represented by
repeated ``image[]`` fields, matching the upstream Images API convention.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import secrets
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_BASE_URL = "https://api.uniapi.io/v1"
DEFAULT_MODEL = "gpt-image-1"
SUPPORTED_INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
MAX_ERROR_BODY_BYTES = 16 * 1024
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class OpenAIImageModelProfile:
    label: str
    use_case: str
    default_size: str
    supported_sizes: frozenset[str]
    reference_guidance: str
    supports_input_fidelity: bool = False


OPENAI_IMAGE_MODEL_PROFILES: dict[str, OpenAIImageModelProfile] = {
    "gpt-image-2": OpenAIImageModelProfile(
        label="GPT Image 2",
        use_case="Highest fidelity GPT image generation/editing on UniAPI",
        default_size="1024x1024",
        supported_sizes=frozenset({"1024x1024", "1024x1536", "1536x1024", "auto"}),
        reference_guidance="Up to ~10 reference images; favor <=5 high-quality inputs",
        supports_input_fidelity=False,
    ),
    "gpt-image-2.5-sunburst": OpenAIImageModelProfile(
        label="GPT Image 2.5 Sunburst",
        use_case="Premium quality, better instruction following for editing",
        default_size="1024x1024",
        supported_sizes=frozenset({"1024x1024", "1024x1536", "1536x1024", "auto"}),
        reference_guidance="Up to ~10 reference images; very high fidelity",
        supports_input_fidelity=False,
    ),
    "gpt-image-1": OpenAIImageModelProfile(
        label="GPT Image 1",
        use_case="Legacy stable generation/editing",
        default_size="1024x1024",
        supported_sizes=frozenset({"1024x1024", "1024x1536", "1536x1024", "auto"}),
        reference_guidance="Up to ~10 reference images",
        supports_input_fidelity=True,
    ),
}


class UniAPIImageError(RuntimeError):
    """A safe, actionable UniAPI request/response error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable


@dataclass(frozen=True)
class InputImage:
    path: Path
    content_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SavedImage:
    path: Path
    sha256: str
    size_bytes: int
    revised_prompt: str | None


@dataclass(frozen=True)
class RestorationResult:
    final_prompt: str
    input_images: list[InputImage]
    saved_images: list[SavedImage]
    response: dict[str, Any]


def _detect_image_type(path: Path) -> str:
    """Check both extension and magic bytes; fail before spending an API call."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input format for {path}: expected PNG, JPEG, or WEBP"
        )
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    guessed = mimetypes.guess_type(path.name)[0]
    raise ValueError(f"{path} does not contain a valid {guessed or 'image'} header")


def inspect_images(image_paths: Sequence[str | Path]) -> list[InputImage]:
    if not image_paths:
        raise ValueError("At least one input image is required")
    inspected: list[InputImage] = []
    seen: set[Path] = set()
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate input image: {path}")
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        content_type = _detect_image_type(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inspected.append(
            InputImage(
                path=path,
                content_type=content_type,
                size_bytes=path.stat().st_size,
                sha256=digest.hexdigest(),
            )
        )
    return inspected


def encode_multipart(
    fields: Mapping[str, str | int],
    images: Sequence[InputImage],
    *,
    image_field: str = "image[]",
    boundary: str | None = None,
) -> tuple[bytes, str]:
    """Encode fields with one repeated multipart file field per image."""
    boundary = boundary or f"----i2i-{secrets.token_hex(18)}"
    boundary_bytes = boundary.encode("ascii")
    body = bytearray()

    def add_line(value: bytes) -> None:
        body.extend(value)
        body.extend(b"\r\n")

    for name, value in fields.items():
        add_line(b"--" + boundary_bytes)
        add_line(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        add_line(b"")
        add_line(str(value).encode("utf-8"))

    for image in images:
        safe_name = image.path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        add_line(b"--" + boundary_bytes)
        add_line(
            (
                f'Content-Disposition: form-data; name="{image_field}"; '
                f'filename="{safe_name}"'
            ).encode("utf-8")
        )
        add_line(f"Content-Type: {image.content_type}".encode("ascii"))
        add_line(b"")
        body.extend(image.path.read_bytes())
        body.extend(b"\r\n")
    add_line(b"--" + boundary_bytes + b"--")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _edits_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if normalized.endswith("/images/edits"):
        return normalized
    return f"{normalized}/images/edits"


def resolve_openai_model_size(model: str, requested_size: str | None) -> str:
    """Resolve ``auto`` and reject explicit sizes unsupported by known models."""

    profile = OPENAI_IMAGE_MODEL_PROFILES.get(model)
    if profile is None:
        # Unknown model: still honor the request size if it's a sane default.
        return requested_size or "auto"
    requested = requested_size or "auto"
    if requested == "auto":
        return "auto"
    if requested not in profile.supported_sizes:
        supported = ", ".join(sorted(profile.supported_sizes))
        raise ValueError(
            f"OpenAI model {model} does not accept size {requested}; use one of: {supported}"
        )
    return requested


def _extract_api_error(raw: bytes) -> str:
    clipped = raw[:MAX_ERROR_BODY_BYTES]
    try:
        parsed = json.loads(clipped.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        text = clipped.decode("utf-8", errors="replace").strip()
        return text or "empty response body"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message") or "API request failed"
        return f"{code}: {message}" if code else str(message)
    return json.dumps(parsed, ensure_ascii=False)[:2000]


class UniAPIImageClient:
    """Calls `/v1/images/edits` and returns the decoded JSON response."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("A non-empty UniAPI API key is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_key = api_key.strip()
        self.endpoint = _edits_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def edit(
        self,
        *,
        image_paths: Sequence[str | Path],
        prompt: str,
        model: str = DEFAULT_MODEL,
        size: str = "auto",
        quality: str = "high",
        n: int = 1,
        output_format: str = "png",
        input_fidelity: str | None = "high",
        response_format: str | None = None,
        background: str | None = None,
        moderation: str | None = None,
        output_compression: int | None = None,
        image_field: str = "image[]",
    ) -> tuple[dict[str, Any], list[InputImage]]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if n < 1:
            raise ValueError("n must be at least 1")
        if output_compression is not None and not 0 <= output_compression <= 100:
            raise ValueError("output_compression must be between 0 and 100")
        if image_field not in {"image[]", "image"}:
            raise ValueError("image_field must be 'image[]' or 'image'")

        images = inspect_images(image_paths)
        fields: dict[str, str | int] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
            "output_format": output_format,
        }
        # input_fidelity is only supported by gpt-image-1; gpt-image-2/2.5 reject it.
        profile = OPENAI_IMAGE_MODEL_PROFILES.get(model)
        supports_input_fidelity = bool(profile and profile.supports_input_fidelity)
        optional_fields: dict[str, str | int | None] = {
            "input_fidelity": input_fidelity if supports_input_fidelity else None,
            "response_format": response_format,
            "background": background,
            "moderation": moderation,
            "output_compression": output_compression,
        }
        fields.update({key: value for key, value in optional_fields.items() if value is not None})
        # Strip fields the model doesn't support to avoid opaque 4xx errors.
        resolved_size = resolve_openai_model_size(model, size)
        fields["size"] = resolved_size
        body, content_type = encode_multipart(fields, images, image_field=image_field)
        response = self._request_with_retries(body, content_type)
        return response, images

    def _request_with_retries(self, body: bytes, content_type: str) -> dict[str, Any]:
        last_error: UniAPIImageError | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": content_type,
                    "Accept": "application/json",
                    "User-Agent": "general-video-annotation-i2i/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    request_id = response.headers.get("x-request-id")
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise UniAPIImageError(
                        "UniAPI returned a non-JSON success response",
                        request_id=request_id,
                    ) from exc
                if not isinstance(payload, dict):
                    raise UniAPIImageError("UniAPI response must be a JSON object")
                return payload
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
                last_error = UniAPIImageError(
                    f"UniAPI network error: {exc.reason if hasattr(exc, 'reason') else exc}",
                    retryable=True,
                )
            if not last_error.retryable or attempt >= self.max_retries:
                raise last_error
            delay = min(20.0, (2**attempt) + random.uniform(0.0, 0.5))
            time.sleep(delay)
        raise last_error or UniAPIImageError("UniAPI request failed")


def _output_format_from_path(path: Path, explicit: str | None) -> tuple[Path, str]:
    suffix_map = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
    if not path.suffix:
        chosen = explicit or "png"
        suffix = ".jpg" if chosen == "jpeg" else f".{chosen}"
        return path.with_suffix(suffix), chosen
    inferred = suffix_map.get(path.suffix.lower())
    if inferred is None:
        raise ValueError("Output path must end in .png, .jpg, .jpeg, or .webp")
    if explicit and explicit != inferred:
        raise ValueError(
            f"--output-format {explicit} conflicts with output extension {path.suffix}"
        )
    return path, explicit or inferred


def _numbered_output_paths(base: Path, count: int) -> list[Path]:
    if count == 1:
        return [base]
    return [base.with_name(f"{base.stem}_{index:02d}{base.suffix}") for index in range(1, count + 1)]


def _validate_output_targets(
    image_paths: Sequence[str | Path],
    output_paths: Sequence[Path],
    *,
    metadata_path: Path | None = None,
) -> None:
    inputs = {Path(path).expanduser().resolve() for path in image_paths}
    outputs = [path.expanduser().resolve() for path in output_paths]
    if inputs.intersection(outputs):
        raise ValueError("An output path must not overwrite an input reference image")
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be unique")
    if metadata_path is not None:
        resolved_metadata = metadata_path.expanduser().resolve()
        if resolved_metadata in inputs or resolved_metadata in set(outputs):
            raise ValueError("Metadata output must differ from all input and image output paths")


def _download(url: str, timeout_seconds: float) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UniAPIImageError("Generated image URL must be an absolute HTTP(S) URL")
    request = urllib.request.Request(url, headers={"User-Agent": "general-video-annotation-i2i/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise UniAPIImageError("Generated image download exceeds 100 MiB")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise UniAPIImageError("Generated image download exceeds 100 MiB")
    return data


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_response_images(
    response: Mapping[str, Any],
    output_path: str | Path,
    *,
    timeout_seconds: float = 180.0,
    overwrite: bool = False,
) -> list[SavedImage]:
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise UniAPIImageError("UniAPI response contains no images in data[]")
    paths = _numbered_output_paths(Path(output_path).expanduser().resolve(), len(data))
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    saved: list[SavedImage] = []
    for item, path in zip(data, paths):
        if not isinstance(item, dict):
            raise UniAPIImageError("Invalid item in UniAPI data[]")
        encoded = item.get("b64_json")
        url = item.get("url")
        if isinstance(encoded, str) and encoded:
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise UniAPIImageError("UniAPI returned invalid base64 image data") from exc
        elif isinstance(url, str) and url:
            content = _download(url, timeout_seconds)
        else:
            raise UniAPIImageError("Image result has neither b64_json nor url")
        if not content:
            raise UniAPIImageError("UniAPI returned an empty image")
        _atomic_write(path, content, overwrite=overwrite)
        saved.append(
            SavedImage(
                path=path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                revised_prompt=item.get("revised_prompt"),
            )
        )
    return saved


def compose_prompt(
    task_instructions: str,
    user_prompt: str,
    *,
    asset_description: str | None = None,
) -> str:
    sections = [task_instructions.strip()]
    if asset_description and asset_description.strip():
        sections.append(f"Known asset description:\n{asset_description.strip()}")
    sections.append(f"Additional user direction:\n{user_prompt.strip()}")
    return "\n\n".join(sections)


def restore_reference(
    client: UniAPIImageClient,
    *,
    image_paths: Sequence[str | Path],
    user_prompt: str,
    output_path: str | Path,
    task_instructions: str,
    asset_description: str | None = None,
    overwrite: bool = False,
    edit_options: Mapping[str, Any] | None = None,
) -> RestorationResult:
    """Programmatic counterpart to the task CLIs.

    ``edit_options`` is forwarded to :meth:`UniAPIImageClient.edit`, allowing a
    pipeline to set model, size, quality, fidelity, and other supported fields.
    """
    options = dict(edit_options or {})
    resolved_output, inferred_format = _output_format_from_path(
        Path(output_path).expanduser(), options.get("output_format")
    )
    options["output_format"] = inferred_format
    expected_count = int(options.get("n", 1))
    _validate_output_targets(
        image_paths,
        _numbered_output_paths(resolved_output, expected_count),
    )
    final_prompt = compose_prompt(
        task_instructions,
        user_prompt,
        asset_description=asset_description,
    )
    response, inputs = client.edit(
        image_paths=image_paths,
        prompt=final_prompt,
        **options,
    )
    saved = save_response_images(
        response,
        resolved_output,
        timeout_seconds=client.timeout_seconds,
        overwrite=overwrite,
    )
    return RestorationResult(
        final_prompt=final_prompt,
        input_images=inputs,
        saved_images=saved,
        response=response,
    )


def build_task_parser(description: str, *, default_size: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="IMAGE",
        help="Ordered reference images; put the strongest/main reference first",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Additional restoration instructions")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 text file containing instructions")
    parser.add_argument("--asset-description", help="Known identity/layout/object details")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--model", default=os.getenv("UNIAPI_IMAGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("UNIAPI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument("--size", default=default_size)
    parser.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--output-format", choices=["png", "jpeg", "webp"])
    parser.add_argument("--output-compression", type=int)
    parser.add_argument(
        "--input-fidelity",
        default="high",
        choices=["high", "low", "omit"],
        help="Use 'omit' for models/proxies that do not accept this field",
    )
    parser.add_argument("--response-format", choices=["b64_json", "url"])
    parser.add_argument("--background", choices=["auto", "opaque", "transparent"])
    parser.add_argument("--moderation", choices=["auto", "low"])
    parser.add_argument(
        "--image-field",
        default="image[]",
        choices=["image[]", "image"],
        help="Use 'image' only if a proxy rejects the upstream repeated image[] field",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print request metadata")
    return parser


def _read_user_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        try:
            return args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Cannot read prompt file {args.prompt_file}: {exc}") from exc
    return (args.prompt or "").strip()


def run_task_cli(
    *,
    task_name: str,
    task_instructions: str,
    default_size: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_task_parser(f"Restore a high-quality {task_name} reference from multiple images", default_size=default_size)
    args = parser.parse_args(argv)
    try:
        user_prompt = _read_user_prompt(args)
        if not user_prompt:
            raise ValueError("Prompt must not be empty")
        if args.n < 1:
            raise ValueError("--n must be at least 1")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if args.max_retries < 0:
            raise ValueError("--max-retries must be non-negative")
        if args.output_compression is not None and not 0 <= args.output_compression <= 100:
            raise ValueError("--output-compression must be between 0 and 100")
        inputs = inspect_images(args.images)
        output_path, output_format = _output_format_from_path(args.output.expanduser(), args.output_format)
        prompt = compose_prompt(
            task_instructions,
            user_prompt,
            asset_description=args.asset_description,
        )
        request_summary = {
            "task": task_name,
            "endpoint": _edits_url(args.base_url),
            "model": args.model,
            "images": [
                {
                    "path": str(image.path),
                    "size_bytes": image.size_bytes,
                    "sha256": image.sha256,
                }
                for image in inputs
            ],
            "prompt": prompt,
            "size": args.size,
            "quality": args.quality,
            "n": args.n,
            "output_format": output_format,
            "input_fidelity": None if args.input_fidelity == "omit" else args.input_fidelity,
            "output": str(output_path.resolve()),
        }
        planned_outputs = _numbered_output_paths(output_path.resolve(), args.n)
        planned_metadata = None
        if not args.no_metadata:
            planned_metadata = (
                args.metadata_output.expanduser().resolve()
                if args.metadata_output
                else output_path.resolve().with_suffix(output_path.suffix + ".json")
            )
        _validate_output_targets(
            args.images,
            planned_outputs,
            metadata_path=planned_metadata,
        )
        if args.dry_run:
            print(json.dumps(request_summary, ensure_ascii=False, indent=2))
            return 0

        for planned_path in planned_outputs:
            if planned_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing output: {planned_path}")
        if planned_metadata is not None:
            if planned_metadata.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing output: {planned_metadata}")

        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {args.api_key_env} is not set; API keys are intentionally not accepted on the command line"
            )
        client = UniAPIImageClient(
            api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        response, _ = client.edit(
            image_paths=args.images,
            prompt=prompt,
            model=args.model,
            size=args.size,
            quality=args.quality,
            n=args.n,
            output_format=output_format,
            input_fidelity=None if args.input_fidelity == "omit" else args.input_fidelity,
            response_format=args.response_format,
            background=args.background,
            moderation=args.moderation,
            output_compression=args.output_compression,
            image_field=args.image_field,
        )
        saved = save_response_images(
            response,
            output_path,
            timeout_seconds=args.timeout,
            overwrite=args.overwrite,
        )
        metadata = {
            "schema_version": "uniapi-i2i-asset-v1",
            "request": request_summary,
            "response": {
                "created": response.get("created"),
                "usage": response.get("usage"),
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
        if not args.no_metadata:
            metadata_path = (
                args.metadata_output.expanduser().resolve()
                if args.metadata_output
                else output_path.resolve().with_suffix(output_path.suffix + ".json")
            )
            encoded_metadata = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            _atomic_write(metadata_path, encoded_metadata, overwrite=args.overwrite)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, UniAPIImageError) as exc:
        request_id = getattr(exc, "request_id", None)
        suffix = f" (request_id={request_id})" if request_id else ""
        print(f"error: {exc}{suffix}", file=sys.stderr)
        return 2
