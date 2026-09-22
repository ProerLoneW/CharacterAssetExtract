#!/usr/bin/env python3
"""OpenAI / UniAPI clothing-extraction counterpart to extract_character_clothing.

This module shares the prompt text and planning helpers with the Gemini version
but routes the request through the OpenAI ``/v1/images/edits`` endpoint via
:class:`UniAPIImageClient`. The output PNG is post-processed with the same
alpha-bake step to give downstream consumers a proper RGBA plate even when
the OpenAI image model returns an opaque-background JPG/PNG.

Programmatic entry points:

* :func:`extract_character_top_openai`
* :func:`extract_character_pants_openai`
* :func:`extract_character_combined_openai`

CLI usage::

    export UNIAPI_API_KEY='...'
    python i2i_tools/extract_character_clothing_openai.py \\
        --images character_a.jpg character_b.jpg character_c.jpg \\
        --output outputs/character_clothing.png \\
        --model gpt-image-2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

try:
    from .uniapi_image_client import (
        OPENAI_IMAGE_MODEL_PROFILES,
        RestorationResult,
        UniAPIImageClient,
        UniAPIImageError,
        _atomic_write,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
        resolve_openai_model_size,
        restore_reference,
        save_response_images,
    )
    from .uniapi_gemini_image_client import UniAPIGeminiImageClient  # noqa: F401  (re-exported)
    from .extract_character_clothing import (
        COMBINED_INSTRUCTIONS,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        ClothingExtractionPlan,
        bake_saved_image,
        plan_clothing_extraction,
    )
except ImportError:  # pragma: no cover - script import path
    from uniapi_image_client import (
        OPENAI_IMAGE_MODEL_PROFILES,
        RestorationResult,
        UniAPIImageClient,
        UniAPIImageError,
        _atomic_write,
        _validate_output_targets,
        compose_prompt,
        inspect_images,
        resolve_openai_model_size,
        restore_reference,
        save_response_images,
    )
    from uniapi_gemini_image_client import UniAPIGeminiImageClient  # noqa: F401
    from extract_character_clothing import (
        COMBINED_INSTRUCTIONS,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        ClothingExtractionPlan,
        bake_saved_image,
        plan_clothing_extraction,
    )


DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2"
SUPPORTED_OPENAI_MODELS = tuple(sorted(OPENAI_IMAGE_MODEL_PROFILES))


# Aspect-ratio mapping used to translate the Gemini 1:1 / 2:3 requests into the
# closest OpenAI size that the proxy accepts. OpenAI images/edits returns
# square or 3:2/2:3 canvas sizes; combined plates use the 2:3 portrait size.
_ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "2:3": "1024x1536",
    "3:2": "1536x1024",
    "3:4": "1024x1536",
    "4:3": "1536x1024",
    "4:5": "1024x1536",
    "5:4": "1536x1024",
    "9:16": "1024x1536",
    "16:9": "1536x1024",
    "21:9": "1536x1024",
}


def _size_for_aspect(aspect_ratio: str, model: str, requested_size: str | None) -> str:
    """Map a Gemini-style aspect ratio to the closest OpenAI image size.

    Honors ``requested_size`` when set to a non-empty value; otherwise uses the
    per-model default. Always returns a size accepted by ``model``.
    """
    profile = OPENAI_IMAGE_MODEL_PROFILES.get(model)
    if requested_size and requested_size != "auto":
        return resolve_openai_model_size(model, requested_size)
    fallback = _ASPECT_RATIO_TO_SIZE.get(aspect_ratio, "1024x1024")
    if profile and fallback not in profile.supported_sizes:
        return profile.default_size
    return fallback


def extract_character_top_openai(
    client: UniAPIImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    size: str = "auto",
    aspect_ratio: str = "1:1",
    quality: str = "high",
    background: str | None = "transparent",
    output_format: str = "png",
    input_fidelity: str | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
    response_format: str | None = None,
    moderation: str | None = "low",
) -> RestorationResult:
    """Run the upper-body garment extraction via OpenAI /v1/images/edits."""

    resolved_size = _size_for_aspect(aspect_ratio, model, size)
    result = restore_reference(
        client,
        image_paths=image_paths,
        user_prompt=(
            "Lay out this character's upper-body garment flat, fully spread "
            "and unoccluded, on a fully transparent background (alpha = 0). "
            "Show only the top — no body, no pants, no shoes, no mannequin, "
            "no props, no card, no shadow. Preserve exact fabric, color, "
            "pattern, and condition."
        ),
        output_path=output_path,
        task_instructions=TOP_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        edit_options={
            "model": model,
            "size": resolved_size,
            "quality": quality,
            "n": 1,
            "output_format": output_format,
            "input_fidelity": input_fidelity,
            "response_format": response_format,
            "background": background,
            "moderation": moderation,
        },
    )
    if bake_alpha:
        bake_saved_image(
            Path(output_path).expanduser(),
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    return result


def extract_character_pants_openai(
    client: UniAPIImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    size: str = "auto",
    aspect_ratio: str = "1:1",
    quality: str = "high",
    background: str | None = "transparent",
    output_format: str = "png",
    input_fidelity: str | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
    response_format: str | None = None,
    moderation: str | None = "low",
) -> RestorationResult:
    """Run the lower-body garment extraction via OpenAI /v1/images/edits."""

    resolved_size = _size_for_aspect(aspect_ratio, model, size)
    result = restore_reference(
        client,
        image_paths=image_paths,
        user_prompt=(
            "Lay out this character's lower-body garment flat, fully spread "
            "and unoccluded, on a fully transparent background (alpha = 0). "
            "Show only the pants — no body, no top, no shoes, no mannequin, "
            "no props, no card, no shadow. Waistband at the top of the "
            "canvas, hem at the bottom. Preserve exact fabric, color, "
            "pattern, and condition."
        ),
        output_path=output_path,
        task_instructions=PANTS_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        edit_options={
            "model": model,
            "size": resolved_size,
            "quality": quality,
            "n": 1,
            "output_format": output_format,
            "input_fidelity": input_fidelity,
            "response_format": response_format,
            "background": background,
            "moderation": moderation,
        },
    )
    if bake_alpha:
        bake_saved_image(
            Path(output_path).expanduser(),
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    return result


def extract_character_combined_openai(
    client: UniAPIImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    size: str = "auto",
    aspect_ratio: str = "2:3",
    quality: str = "high",
    background: str | None = "transparent",
    output_format: str = "png",
    input_fidelity: str | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
    response_format: str | None = None,
    moderation: str | None = "low",
) -> RestorationResult:
    """Run the combined top+pants extraction via OpenAI /v1/images/edits."""

    resolved_size = _size_for_aspect(aspect_ratio, model, size)
    result = restore_reference(
        client,
        image_paths=image_paths,
        user_prompt=(
            "Produce a two-half plate: the top half shows the upper-body "
            "garment laid flat, the bottom half shows the lower-body "
            "garment laid flat (waistband up, hem down). Each half is on "
            "its OWN fully transparent background (alpha = 0); do not draw "
            "a shared card, table, ground, or shadow. Preserve exact "
            "fabrics, colors, patterns, and condition for both garments."
        ),
        output_path=output_path,
        task_instructions=COMBINED_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        edit_options={
            "model": model,
            "size": resolved_size,
            "quality": quality,
            "n": 1,
            "output_format": output_format,
            "input_fidelity": input_fidelity,
            "response_format": response_format,
            "background": background,
            "moderation": moderation,
        },
    )
    if bake_alpha:
        bake_saved_image(
            Path(output_path).expanduser(),
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    return result


# ---------------------------------------------------------------------------
# Plan executor that wires the OpenAI client to the existing ClothingExtractionPlan
# ---------------------------------------------------------------------------


def _metadata_for_openai(
    plan: ClothingExtractionPlan,
    result: RestorationResult,
    *,
    task: str,
    model: str,
    requested_size: str,
    resolved_size: str,
    quality: str,
    background: str | None,
    bake_alpha: bool,
    bake_alpha_threshold: int,
    bake_alpha_softness: int,
) -> dict:
    return {
        "schema_version": "character-clothing-extraction-openai-v1",
        "provider": "openai",
        "request": {
            "task": task,
            "endpoint": result.response.get("_endpoint"),
            "model": model,
            "requested_size": requested_size,
            "resolved_size": resolved_size,
            "quality": quality,
            "background": background,
            "character_label": plan.character_label,
            "requested_parts": sorted(plan.requested_parts()),
            "images": [
                {
                    "index": index,
                    "path": str(image.path),
                    "size_bytes": image.size_bytes,
                    "sha256": image.sha256,
                }
                for index, image in enumerate(result.input_images, start=1)
            ],
            "prompt": result.final_prompt,
            "bake_alpha": bake_alpha,
            "bake_alpha_threshold": bake_alpha_threshold,
            "bake_alpha_softness": bake_alpha_softness,
        },
        "response": result.response,
        "outputs": [
            {
                "path": str(item.path),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "revised_prompt": item.revised_prompt,
            }
            for item in result.saved_images
        ],
    }


def execute_plan_openai(
    plan: ClothingExtractionPlan,
    *,
    client: UniAPIImageClient,
    overwrite: bool = False,
    model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    size: str = "auto",
    quality: str = "high",
    background: str | None = "transparent",
    output_format: str = "png",
    input_fidelity: str | None = "high",
    response_format: str | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> dict[str, RestorationResult]:
    """Run all requested parts of a clothing extraction plan via the OpenAI client.

    Returns a mapping ``{'top': RestorationResult, 'pants': ..., 'combined': ...}``
    with only the keys that were requested. Each part writes a PNG next to the
    plan output paths and a JSON sidecar with the same naming.
    """

    results: dict[str, RestorationResult] = {}
    common = {
        "model": model,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "input_fidelity": input_fidelity,
        "response_format": response_format,
        "overwrite": overwrite,
        "bake_alpha": bake_alpha,
        "bake_alpha_threshold": bake_alpha_threshold,
        "bake_alpha_softness": bake_alpha_softness,
    }
    if plan.top_output is not None:
        results["top"] = extract_character_top_openai(
            client,
            image_paths=plan.image_paths,
            output_path=plan.top_output,
            asset_description=plan.character_label,
            aspect_ratio="1:1",
            **common,
        )
        if plan.top_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_openai(
                        plan,
                        results["top"],
                        task="character_clothing_top",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_aspect("1:1", model, size),
                        quality=quality,
                        background=background,
                        bake_alpha=bake_alpha,
                        bake_alpha_threshold=bake_alpha_threshold,
                        bake_alpha_softness=bake_alpha_softness,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            _atomic_write(plan.top_metadata, encoded, overwrite=overwrite)
    if plan.pants_output is not None:
        results["pants"] = extract_character_pants_openai(
            client,
            image_paths=plan.image_paths,
            output_path=plan.pants_output,
            asset_description=plan.character_label,
            aspect_ratio="1:1",
            **common,
        )
        if plan.pants_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_openai(
                        plan,
                        results["pants"],
                        task="character_clothing_pants",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_aspect("1:1", model, size),
                        quality=quality,
                        background=background,
                        bake_alpha=bake_alpha,
                        bake_alpha_threshold=bake_alpha_threshold,
                        bake_alpha_softness=bake_alpha_softness,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            _atomic_write(plan.pants_metadata, encoded, overwrite=overwrite)
    if plan.combined_output is not None:
        results["combined"] = extract_character_combined_openai(
            client,
            image_paths=plan.image_paths,
            output_path=plan.combined_output,
            asset_description=plan.character_label,
            aspect_ratio="2:3",
            **common,
        )
        if plan.combined_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_openai(
                        plan,
                        results["combined"],
                        task="character_clothing_combined",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_aspect("2:3", model, size),
                        quality=quality,
                        background=background,
                        bake_alpha=bake_alpha,
                        bake_alpha_threshold=bake_alpha_threshold,
                        bake_alpha_softness=bake_alpha_softness,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            _atomic_write(plan.combined_metadata, encoded, overwrite=overwrite)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the character's top and/or pants as transparent-background "
            "PNG plates using the OpenAI /v1/images/edits endpoint."
        )
    )
    parser.add_argument("--images", nargs="+", required=True, metavar="IMAGE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parts",
        default="top,pants",
        help="Comma-separated list of parts to generate: top, pants, combined.",
    )
    parser.add_argument("--character-label", default="character")
    parser.add_argument(
        "--model",
        default=os.getenv("UNIAPI_OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_IMAGE_MODEL),
        choices=SUPPORTED_OPENAI_MODELS,
        help="OpenAI image model on the UniAPI proxy.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("UNIAPI_OPENAI_BASE_URL", "https://api.uniapi.io/v1"),
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output size; e.g. 1024x1024, 1024x1536, auto. Default: derived from aspect ratio.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=None,
        help="Used to derive the default output size when --size=auto.",
    )
    parser.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high"])
    parser.add_argument("--background", default="transparent", choices=["auto", "opaque", "transparent"])
    parser.add_argument("--output-format", default="png", choices=["png", "jpeg", "webp"])
    parser.add_argument("--input-fidelity", default="high", choices=["high", "low", "omit"])
    parser.add_argument("--response-format", default=None, choices=["b64_json", "url"])
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bake-alpha", action="store_true")
    parser.add_argument("--bake-alpha-threshold", type=int, default=18)
    parser.add_argument("--bake-alpha-softness", type=int, default=32)
    return parser


def _coerce_parts(value: str) -> list[str]:
    parts = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--parts must be a non-empty comma-separated list")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        parts = _coerce_parts(args.parts)
        plan = plan_clothing_extraction(
            args.images,
            output=args.output,
            parts=parts,
            character_label=args.character_label,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    request_summary = {
        "task": "character_clothing_extraction_openai",
        "provider": "openai",
        "model": args.model,
        "base_url": args.base_url,
        "requested_size": args.size,
        "resolved_size_top": _size_for_aspect("1:1", args.model, args.size),
        "resolved_size_combined": _size_for_aspect("2:3", args.model, args.size),
        "quality": args.quality,
        "background": args.background,
        "output_format": args.output_format,
        "input_fidelity": args.input_fidelity,
        "parts_requested": sorted(plan.requested_parts()),
        "bake_alpha": args.bake_alpha,
        "outputs": {
            "top": str(plan.top_output) if plan.top_output else None,
            "pants": str(plan.pants_output) if plan.pants_output else None,
            "combined": str(plan.combined_output) if plan.combined_output else None,
        },
    }
    print(json.dumps(request_summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(
            f"error: environment variable {args.api_key_env} is not set",
            file=sys.stderr,
        )
        return 2

    client = UniAPIImageClient(
        api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )

    fidelity = None if args.input_fidelity == "omit" else args.input_fidelity
    try:
        results = execute_plan_openai(
            plan,
            client=client,
            overwrite=args.overwrite,
            model=args.model,
            size=args.size,
            quality=args.quality,
            background=args.background,
            output_format=args.output_format,
            input_fidelity=fidelity,
            response_format=args.response_format,
            bake_alpha=args.bake_alpha,
            bake_alpha_threshold=args.bake_alpha_threshold,
            bake_alpha_softness=args.bake_alpha_softness,
        )
    except (FileNotFoundError, FileExistsError, ValueError, UniAPIImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    saved: list[dict] = []
    for part_name, sub in results.items():
        for item in sub.saved_images:
            saved.append({"part": part_name, "path": str(item.path), "sha256": item.sha256})
    print(json.dumps({"saved": saved}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
