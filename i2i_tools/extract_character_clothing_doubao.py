#!/usr/bin/env python3
"""Doubao (ARK / Seedream) clothing-extraction wrapper.

Mirrors :mod:`extract_character_clothing_openai` but routes through the
``UniAPIDoubaoImageClient`` (JSON ``/v1/images/generations`` endpoint with
reference images as base64 data URIs). All three task prompts
(``TOP_INSTRUCTIONS`` / ``PANTS_INSTRUCTIONS`` / ``COMBINED_INSTRUCTIONS``)
and the alpha-bake step are shared with the Gemini and OpenAI paths.

Programmatic entry points:

* :func:`extract_character_top_doubao`
* :func:`extract_character_pants_doubao`
* :func:`extract_character_combined_doubao`
* :func:`execute_plan_doubao`

CLI usage::

    export UNIAPI_API_KEY='...'
    python i2i_tools/extract_character_clothing_doubao.py \\
        --images character_a.jpg character_b.jpg character_c.jpg \\
        --output outputs/CHR004.png \\
        --model doubao-seedream-5-0-260128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

try:
    from .uniapi_doubao_image_client import (
        DEFAULT_DOUBEO_MODEL,
        DOUBAO_IMAGE_MODEL_PROFILES,
        DoubaoRestorationResult,
        SUPPORTED_DOUBEAO_MODELS,
        UniAPIDoubaoImageClient,
        UniAPIImageError,
        _atomic_write,
        resolve_doubao_model_size,
        restore_doubao_reference,
    )
    from .extract_character_clothing import (
        COMBINED_INSTRUCTIONS,
        ClothingExtractionPlan,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        bake_saved_image,
        plan_clothing_extraction,
    )
    from .uniapi_image_client import _atomic_write as _shared_atomic_write  # noqa: F401
except ImportError:  # pragma: no cover - script import path
    from uniapi_doubao_image_client import (
        DEFAULT_DOUBEO_MODEL,
        DOUBAO_IMAGE_MODEL_PROFILES,
        DoubaoRestorationResult,
        SUPPORTED_DOUBEAO_MODELS,
        UniAPIDoubaoImageClient,
        UniAPIImageError,
        _atomic_write,
        resolve_doubao_model_size,
        restore_doubao_reference,
    )
    from extract_character_clothing import (
        COMBINED_INSTRUCTIONS,
        ClothingExtractionPlan,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        bake_saved_image,
        plan_clothing_extraction,
    )


# Per-part Doubao sizes. Doubao requires at least 3686400 pixels (~1920x1920)
# for its image generation endpoint, so the default sizes are bumped to 2K
# presets which both meet the floor and produce quality output. Use the
# aspect-ratio presets "2K" (square-ish) and "3K" (larger) or pass an explicit
# WxH that satisfies the floor.
_DEFAULT_TOP_SIZE = "2048x2048"
_DEFAULT_PANTS_SIZE = "2048x2048"
_DEFAULT_COMBINED_SIZE = "2048x2048"


def _size_for_part(model: str, part: str, requested_size: str | None) -> str:
    if requested_size and requested_size != "auto":
        resolved = resolve_doubao_model_size(model, requested_size)
        from uniapi_doubao_image_client import _assert_meets_doubao_pixel_floor

        _assert_meets_doubao_pixel_floor(resolved)
        return resolved
    if part == "combined":
        fallback = _DEFAULT_COMBINED_SIZE
    else:
        fallback = _DEFAULT_TOP_SIZE
    profile = DOUBAO_IMAGE_MODEL_PROFILES.get(model)
    if profile and fallback not in profile.supported_sizes:
        return profile.default_size
    return fallback


def extract_character_top_doubao(
    client: UniAPIDoubaoImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_DOUBEO_MODEL,
    size: str = "auto",
    output_format: str = "png",
    watermark: bool = False,
    seed: int | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> DoubaoRestorationResult:
    """Run the upper-body garment extraction via Doubao /v1/images/generations."""

    resolved_size = _size_for_part(model, "top", size)
    result = restore_doubao_reference(
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
        generation_options={
            "model": model,
            "size": resolved_size,
            "output_format": output_format,
            "watermark": watermark,
            "seed": seed,
        },
    )
    if bake_alpha:
        bake_saved_image(
            Path(output_path).expanduser(),
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    return result


def extract_character_pants_doubao(
    client: UniAPIDoubaoImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_DOUBEO_MODEL,
    size: str = "auto",
    output_format: str = "png",
    watermark: bool = False,
    seed: int | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> DoubaoRestorationResult:
    """Run the lower-body garment extraction via Doubao /v1/images/generations."""

    resolved_size = _size_for_part(model, "pants", size)
    result = restore_doubao_reference(
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
        generation_options={
            "model": model,
            "size": resolved_size,
            "output_format": output_format,
            "watermark": watermark,
            "seed": seed,
        },
    )
    if bake_alpha:
        bake_saved_image(
            Path(output_path).expanduser(),
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    return result


def extract_character_combined_doubao(
    client: UniAPIDoubaoImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    model: str = DEFAULT_DOUBEO_MODEL,
    size: str = "auto",
    output_format: str = "png",
    watermark: bool = False,
    seed: int | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> DoubaoRestorationResult:
    """Run the combined top+pants extraction via Doubao /v1/images/generations."""

    resolved_size = _size_for_part(model, "combined", size)
    result = restore_doubao_reference(
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
        generation_options={
            "model": model,
            "size": resolved_size,
            "output_format": output_format,
            "watermark": watermark,
            "seed": seed,
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
# Plan executor
# ---------------------------------------------------------------------------


def _metadata_for_doubao(
    plan: ClothingExtractionPlan,
    result: DoubaoRestorationResult,
    *,
    task: str,
    model: str,
    requested_size: str,
    resolved_size: str,
    output_format: str,
    watermark: bool,
    bake_alpha: bool,
    bake_alpha_threshold: int,
    bake_alpha_softness: int,
) -> dict:
    return {
        "schema_version": "character-clothing-extraction-doubao-v1",
        "provider": "doubao",
        "request": {
            "task": task,
            "model": model,
            "requested_size": requested_size,
            "resolved_size": resolved_size,
            "output_format": output_format,
            "watermark": watermark,
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


def execute_plan_doubao(
    plan: ClothingExtractionPlan,
    *,
    client: UniAPIDoubaoImageClient,
    overwrite: bool = False,
    model: str = DEFAULT_DOUBEO_MODEL,
    size: str = "auto",
    output_format: str = "png",
    watermark: bool = False,
    seed: int | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> dict[str, DoubaoRestorationResult]:
    """Run all requested parts of a clothing extraction plan via the Doubao client."""

    results: dict[str, DoubaoRestorationResult] = {}
    common = {
        "model": model,
        "size": size,
        "output_format": output_format,
        "watermark": watermark,
        "seed": seed,
        "overwrite": overwrite,
        "bake_alpha": bake_alpha,
        "bake_alpha_threshold": bake_alpha_threshold,
        "bake_alpha_softness": bake_alpha_softness,
    }
    if plan.top_output is not None:
        results["top"] = extract_character_top_doubao(
            client,
            image_paths=plan.image_paths,
            output_path=plan.top_output,
            asset_description=plan.character_label,
            **common,
        )
        if plan.top_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_doubao(
                        plan,
                        results["top"],
                        task="character_clothing_top",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_part(model, "top", size),
                        output_format=output_format,
                        watermark=watermark,
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
        results["pants"] = extract_character_pants_doubao(
            client,
            image_paths=plan.image_paths,
            output_path=plan.pants_output,
            asset_description=plan.character_label,
            **common,
        )
        if plan.pants_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_doubao(
                        plan,
                        results["pants"],
                        task="character_clothing_pants",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_part(model, "pants", size),
                        output_format=output_format,
                        watermark=watermark,
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
        results["combined"] = extract_character_combined_doubao(
            client,
            image_paths=plan.image_paths,
            output_path=plan.combined_output,
            asset_description=plan.character_label,
            **common,
        )
        if plan.combined_metadata is not None:
            encoded = (
                json.dumps(
                    _metadata_for_doubao(
                        plan,
                        results["combined"],
                        task="character_clothing_combined",
                        model=model,
                        requested_size=size,
                        resolved_size=_size_for_part(model, "combined", size),
                        output_format=output_format,
                        watermark=watermark,
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
            "PNG plates using Doubao Seedream on the ARK endpoint via UniAPI."
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
        default=os.getenv("UNIAPI_DOUBEO_IMAGE_MODEL", DEFAULT_DOUBEO_MODEL),
        choices=SUPPORTED_DOUBEAO_MODELS,
        help="Doubao model id on the UniAPI proxy.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("UNIAPI_DOUBEO_BASE_URL", "https://api.uniapi.io/v1"),
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output size; e.g. 2K, 3K, 1024x1024, 1024x1536, auto. Default: auto.",
    )
    parser.add_argument("--output-format", choices=["png", "jpeg"], default="png")
    parser.add_argument("--watermark", choices=["yes", "no"], default="no")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
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
        "task": "character_clothing_extraction_doubao",
        "provider": "doubao",
        "model": args.model,
        "base_url": args.base_url,
        "requested_size": args.size,
        "resolved_size_top": _size_for_part(args.model, "top", args.size),
        "resolved_size_combined": _size_for_part(args.model, "combined", args.size),
        "watermark": args.watermark == "yes",
        "output_format": args.output_format,
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

    client = UniAPIDoubaoImageClient(
        api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    try:
        results = execute_plan_doubao(
            plan,
            client=client,
            overwrite=args.overwrite,
            model=args.model,
            size=args.size,
            output_format=args.output_format,
            watermark=(args.watermark == "yes"),
            seed=args.seed,
            bake_alpha=args.bake_alpha,
            bake_alpha_threshold=args.bake_alpha_threshold,
            bake_alpha_softness=args.bake_alpha_softness,
        )
    except (FileNotFoundError, FileExistsError, ValueError, UniAPIImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    saved: list[dict] = []
    for label, sub in results.items():
        for item in sub.saved_images:
            saved.append({"part": label, "path": str(item.path), "sha256": item.sha256})
    print(json.dumps({"saved": saved}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
