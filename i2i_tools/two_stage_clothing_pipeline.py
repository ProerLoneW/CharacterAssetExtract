#!/usr/bin/env python3
"""Two-stage character clothing extraction pipeline.

Stage 1 — given one or more full-body crops of the SAME character from the
original video, generate a high-quality character turnaround/three-view sheet
with the requested model (Gemini or OpenAI family).

Stage 2 — feed that high-quality turnaround back into the model along with the
original full-body crops, and ask for the isolated top + pants garments as
transparent-background PNG plates.

The same script drives both providers so we can compare the same prompt
template against Gemini-3-pro-image / 3.1-flash-image / 3.1-flash-lite-image
(over the UniAPI Gemini endpoint) and gpt-image-2 / gpt-image-2.5-sunburst
(over the UniAPI OpenAI endpoint).

Usage::

    export UNIAPI_API_KEY='...'
    python i2i_tools/two_stage_clothing_pipeline.py \\
        --images crops/CHR004_0007.jpg crops/CHR004_0009.jpg crops/CHR004_0010.jpg \\
        --character-label CHR004 \\
        --output-root workspace/two_stage_clothing/CHR004 \\
        --model gemini-3-pro-image

The orchestrator writes:

    <output-root>/<model>/<character>_three_view.png
    <output-root>/<model>/<character>_three_view.png.json
    <output-root>/<model>/<character>_top.png
    <output-root>/<model>/<character>_top.png.json
    <output-root>/<model>/<character>_pants.png
    <output-root>/<model>/<character>_pants.png.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from .uniapi_gemini_image_client import (
        DEFAULT_GEMINI_BASE_URL,
        DEFAULT_GEMINI_IMAGE_MODEL,
        GEMINI_IMAGE_MODEL_PROFILES,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
    )
    from .uniapi_image_client import (
        DEFAULT_BASE_URL,
        OPENAI_IMAGE_MODEL_PROFILES,
        RestorationResult,
        UniAPIImageClient,
    )
    from .extract_character_clothing import (
        ClothingExtractionPlan,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        bake_saved_image,
        plan_clothing_extraction,
    )
    from .extract_character_clothing_openai import (
        DEFAULT_OPENAI_IMAGE_MODEL,
        SUPPORTED_OPENAI_MODELS,
        _size_for_aspect,
        extract_character_pants_openai,
        extract_character_top_openai,
    )
    from .restore_character_three_view_and_costume import THREE_VIEW_INSTRUCTIONS
except ImportError:  # pragma: no cover - script import path
    from uniapi_gemini_image_client import (
        DEFAULT_GEMINI_BASE_URL,
        DEFAULT_GEMINI_IMAGE_MODEL,
        GEMINI_IMAGE_MODEL_PROFILES,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
    )
    from uniapi_image_client import (
        DEFAULT_BASE_URL,
        OPENAI_IMAGE_MODEL_PROFILES,
        RestorationResult,
        UniAPIImageClient,
    )
    from extract_character_clothing import (
        ClothingExtractionPlan,
        PANTS_INSTRUCTIONS,
        TOP_INSTRUCTIONS,
        bake_saved_image,
        plan_clothing_extraction,
    )
    from extract_character_clothing_openai import (
        DEFAULT_OPENAI_IMAGE_MODEL,
        SUPPORTED_OPENAI_MODELS,
        _size_for_aspect,
        extract_character_pants_openai,
        extract_character_top_openai,
    )
    from extract_character_clothing_doubao import (
        DEFAULT_DOUBEO_MODEL as _DOUBAO_DEFAULT,
        extract_character_pants_doubao,
        extract_character_top_doubao,
        _size_for_part as _doubao_size_for_part,
    )
    from uniapi_doubao_image_client import (
        DEFAULT_DOUBEO_BASE_URL as _DOUBEO_BASE,
        DEFAULT_DOUBEO_BASE_URL,
        DEFAULT_DOUBEO_MODEL,
        DOUBAO_IMAGE_MODEL_PROFILES,
        DoubaoRestorationResult,
        SUPPORTED_DOUBEAO_MODELS,
        UniAPIDoubaoImageClient,
        restore_doubao_reference,
    )
    from restore_character_three_view_and_costume import THREE_VIEW_INSTRUCTIONS


# Models available for cross-provider testing in this pipeline.
ALL_MODELS: tuple[str, ...] = (
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gpt-image-2",
    "gpt-image-2.5-sunburst",
    "doubao-seedream-5-0-260128",
)


def _classify_model(model: str) -> str:
    if model in GEMINI_IMAGE_MODEL_PROFILES:
        return "gemini"
    if model in OPENAI_IMAGE_MODEL_PROFILES:
        return "openai"
    if model in DOUBAO_IMAGE_MODEL_PROFILES:
        return "doubao"
    raise ValueError(
        f"Unknown model {model!r}; expected one of: {', '.join(ALL_MODELS)}"
    )


def _stage1_prompt_user() -> str:
    return (
        "Produce a clean character turnaround/three-view reference sheet for "
        "this character. Keep identity, hairstyle, body proportions and "
        "complete outfit consistent across all poses. The image is intended "
        "as the single highest-quality reference that downstream pipelines "
        "can use to extract individual garments."
    )


def _stage2_top_user_prompt() -> str:
    return (
        "Lay out this character's upper-body garment flat, fully spread and "
        "unoccluded, on a fully transparent background (alpha = 0). Show "
        "only the top — no body, no pants, no shoes, no mannequin, no "
        "props, no card, no shadow. Preserve exact fabric, color, pattern, "
        "and condition."
    )


def _stage2_pants_user_prompt() -> str:
    return (
        "Lay out this character's lower-body garment flat, fully spread and "
        "unoccluded, on a fully transparent background (alpha = 0). Show "
        "only the pants — no body, no top, no shoes, no mannequin, no "
        "props, no card, no shadow. Waistband at the top of the canvas, "
        "hem at the bottom. Preserve exact fabric, color, pattern, and "
        "condition."
    )


# ---------------------------------------------------------------------------
# Provider-specific stage executors
# ---------------------------------------------------------------------------


def _run_stage1_gemini(
    *,
    client: UniAPIGeminiImageClient,
    image_paths: Sequence[Path],
    output_path: Path,
    model: str,
    image_size: str = "2K",
    overwrite: bool = False,
) -> GeminiRestorationResult:
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=_stage1_prompt_user(),
        output_path=output_path,
        task_instructions=THREE_VIEW_INSTRUCTIONS,
        asset_description=(
            "Stage 1: building a high-quality three-view turnaround reference "
            f"for the character from {len(image_paths)} full-body crops."
        ),
        overwrite=overwrite,
        generation_options={
            "model": model,
            "aspect_ratio": "3:4",
            "image_size": image_size,
        },
    )


def _run_stage1_openai(
    *,
    client: UniAPIImageClient,
    image_paths: Sequence[Path],
    output_path: Path,
    model: str,
    size: str = "auto",
    quality: str = "high",
    overwrite: bool = False,
) -> RestorationResult:
    from uniapi_image_client import restore_reference

    return restore_reference(
        client,
        image_paths=image_paths,
        user_prompt=_stage1_prompt_user(),
        output_path=output_path,
        task_instructions=THREE_VIEW_INSTRUCTIONS,
        asset_description=(
            "Stage 1: building a high-quality three-view turnaround reference "
            f"for the character from {len(image_paths)} full-body crops."
        ),
        overwrite=overwrite,
        edit_options={
            "model": model,
            "size": _size_for_aspect("3:4", model, size),
            "quality": quality,
            "n": 1,
            "output_format": "png",
            "input_fidelity": "high",
            "background": "auto",
        },
    )


def _run_stage2_gemini(
    *,
    client: UniAPIGeminiImageClient,
    stage1_three_view: Path,
    original_crops: Sequence[Path],
    output_top: Path,
    output_pants: Path,
    model: str,
    image_size: str = "2K",
    overwrite: bool = False,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> tuple[GeminiRestorationResult, GeminiRestorationResult]:
    # Stage 2 input order: the high-quality three-view first so it anchors
    # identity/wardrobe, then the original crops as supporting evidence.
    image_paths = [stage1_three_view, *original_crops]
    top_result = restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=_stage2_top_user_prompt(),
        output_path=output_top,
        task_instructions=TOP_INSTRUCTIONS,
        asset_description=(
            "Stage 2: extracting the upper-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        generation_options={
            "model": model,
            "aspect_ratio": "1:1",
            "image_size": image_size,
        },
    )
    pants_result = restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=_stage2_pants_user_prompt(),
        output_path=output_pants,
        task_instructions=PANTS_INSTRUCTIONS,
        asset_description=(
            "Stage 2: extracting the lower-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        generation_options={
            "model": model,
            "aspect_ratio": "1:1",
            "image_size": image_size,
        },
    )
    if bake_alpha:
        bake_saved_image(output_top, threshold=bake_alpha_threshold, softness=bake_alpha_softness)
        bake_saved_image(output_pants, threshold=bake_alpha_threshold, softness=bake_alpha_softness)
    return top_result, pants_result


def _run_stage2_openai(
    *,
    client: UniAPIImageClient,
    stage1_three_view: Path,
    original_crops: Sequence[Path],
    output_top: Path,
    output_pants: Path,
    model: str,
    size: str = "auto",
    quality: str = "high",
    overwrite: bool = False,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> tuple[RestorationResult, RestorationResult]:
    image_paths = [stage1_three_view, *original_crops]
    top_result = extract_character_top_openai(
        client,
        image_paths=image_paths,
        output_path=output_top,
        asset_description=(
            "Stage 2: extracting the upper-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        model=model,
        size=size,
        aspect_ratio="1:1",
        quality=quality,
        background="transparent",
        output_format="png",
        input_fidelity="high",
        bake_alpha=bake_alpha,
        bake_alpha_threshold=bake_alpha_threshold,
        bake_alpha_softness=bake_alpha_softness,
    )
    pants_result = extract_character_pants_openai(
        client,
        image_paths=image_paths,
        output_path=output_pants,
        asset_description=(
            "Stage 2: extracting the lower-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        model=model,
        size=size,
        aspect_ratio="1:1",
        quality=quality,
        background="transparent",
        output_format="png",
        input_fidelity="high",
        bake_alpha=bake_alpha,
        bake_alpha_threshold=bake_alpha_threshold,
        bake_alpha_softness=bake_alpha_softness,
    )
    return top_result, pants_result


def _run_stage1_doubao(
    *,
    client: UniAPIDoubaoImageClient,
    image_paths: Sequence[Path],
    output_path: Path,
    model: str,
    size: str = "auto",
    watermark: bool = False,
    overwrite: bool = False,
) -> DoubaoRestorationResult:
    # Stage 1 produces a 3:4 turnaround; Doubao requires at least 3686400
    # pixels per request, so we use 2048x2048 (square) for the turnaround.
    # 1536x2048 = ~3.14M is below the floor.
    return restore_doubao_reference(
        client,
        image_paths=image_paths,
        user_prompt=_stage1_prompt_user(),
        output_path=output_path,
        task_instructions=THREE_VIEW_INSTRUCTIONS,
        asset_description=(
            "Stage 1: building a high-quality three-view turnaround reference "
            f"for the character from {len(image_paths)} full-body crops."
        ),
        overwrite=overwrite,
        generation_options={
            "model": model,
            "size": "2048x2048" if size == "auto" else size,
            "output_format": "png",
            "watermark": watermark,
        },
    )


def _run_stage2_doubao(
    *,
    client: UniAPIDoubaoImageClient,
    stage1_three_view: Path,
    original_crops: Sequence[Path],
    output_top: Path,
    output_pants: Path,
    model: str,
    size: str = "auto",
    watermark: bool = False,
    overwrite: bool = False,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> tuple[DoubaoRestorationResult, DoubaoRestorationResult]:
    image_paths = [stage1_three_view, *original_crops]
    top_result = extract_character_top_doubao(
        client,
        image_paths=image_paths,
        output_path=output_top,
        asset_description=(
            "Stage 2: extracting the upper-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        model=model,
        size=size,
        output_format="png",
        watermark=watermark,
        bake_alpha=bake_alpha,
        bake_alpha_threshold=bake_alpha_threshold,
        bake_alpha_softness=bake_alpha_softness,
    )
    pants_result = extract_character_pants_doubao(
        client,
        image_paths=image_paths,
        output_path=output_pants,
        asset_description=(
            "Stage 2: extracting the lower-body garment using the high-"
            "quality turnaround plus the original full-body crops."
        ),
        overwrite=overwrite,
        model=model,
        size=size,
        output_format="png",
        watermark=watermark,
        bake_alpha=bake_alpha,
        bake_alpha_threshold=bake_alpha_threshold,
        bake_alpha_softness=bake_alpha_softness,
    )
    return top_result, pants_result


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageOutputs:
    model: str
    provider: str
    three_view: Path
    three_view_metadata: Path
    top: Path
    top_metadata: Path
    pants: Path
    pants_metadata: Path


def _atomic_write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    from uniapi_image_client import _atomic_write

    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, encoded, overwrite=overwrite)


def _stage1_metadata(
    *,
    model: str,
    provider: str,
    inputs: Sequence[Path],
    result,
    image_size_requested: str,
    image_size_resolved: str | None,
    quality: str,
    aspect_ratio: str,
) -> dict:
    base = {
        "schema_version": "two-stage-clothing-v1-stage1",
        "provider": provider,
        "model": model,
        "image_size_requested": image_size_requested,
        "image_size_resolved": image_size_resolved,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "inputs": [{"path": str(p)} for p in inputs],
        "prompt": result.final_prompt,
        "outputs": [
            {"path": str(item.path), "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in result.saved_images
        ],
    }
    if isinstance(result, GeminiRestorationResult):
        base["response_model_version"] = result.response.get("modelVersion")
        base["response_usage"] = result.response.get("usageMetadata")
        base["response_text"] = result.response_text
    else:
        base["response"] = {
            "created": result.response.get("created"),
            "usage": result.response.get("usage"),
        }
    return base


def _stage2_metadata(
    *,
    model: str,
    provider: str,
    inputs: Sequence[Path],
    result,
    part: str,
    image_size_requested: str,
    image_size_resolved: str | None,
    quality: str,
    bake_alpha: bool,
) -> dict:
    base = {
        "schema_version": "two-stage-clothing-v1-stage2",
        "provider": provider,
        "model": model,
        "part": part,
        "image_size_requested": image_size_requested,
        "image_size_resolved": image_size_resolved,
        "quality": quality,
        "bake_alpha": bake_alpha,
        "inputs": [{"path": str(p)} for p in inputs],
        "prompt": result.final_prompt,
        "outputs": [
            {"path": str(item.path), "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in result.saved_images
        ],
    }
    if isinstance(result, GeminiRestorationResult):
        base["response_model_version"] = result.response.get("modelVersion")
        base["response_usage"] = result.response.get("usageMetadata")
        base["response_text"] = result.response_text
    else:
        base["response"] = {
            "created": result.response.get("created"),
            "usage": result.response.get("usage"),
        }
        if result.saved_images and result.saved_images[0].revised_prompt:
            base["revised_prompt"] = result.saved_images[0].revised_prompt
    return base


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoStagePlan:
    model: str
    provider: str
    output_root: Path
    outputs: StageOutputs
    image_size_requested: str
    image_size_resolved: str | None
    quality: str
    bake_alpha: bool


def plan_two_stage(
    *,
    image_paths: Sequence[Path],
    output_root: Path,
    character_label: str,
    model: str,
    image_size: str = "auto",
    quality: str = "high",
) -> TwoStagePlan:
    provider = _classify_model(model)
    model_dir = output_root / model
    model_dir.mkdir(parents=True, exist_ok=True)
    image_size_resolved: str | None
    if provider == "gemini":
        from uniapi_gemini_image_client import resolve_model_image_size

        image_size_resolved = resolve_model_image_size(model, image_size)
    elif provider == "openai":
        image_size_resolved = _size_for_aspect("3:4", model, image_size)
    else:  # doubao
        from extract_character_clothing_doubao import _size_for_part

        image_size_resolved = _size_for_part(model, "combined", image_size)
    three_view = model_dir / f"{character_label}_three_view.png"
    top = model_dir / f"{character_label}_top.png"
    pants = model_dir / f"{character_label}_pants.png"
    return TwoStagePlan(
        model=model,
        provider=provider,
        output_root=output_root,
        outputs=StageOutputs(
            model=model,
            provider=provider,
            three_view=three_view,
            three_view_metadata=three_view.with_suffix(three_view.suffix + ".json"),
            top=top,
            top_metadata=top.with_suffix(top.suffix + ".json"),
            pants=pants,
            pants_metadata=pants.with_suffix(pants.suffix + ".json"),
        ),
        image_size_requested=image_size,
        image_size_resolved=image_size_resolved,
        quality=quality,
        bake_alpha=False,
    )


def run_two_stage(
    plan: TwoStagePlan,
    *,
    original_crops: Sequence[Path],
    gemini_client: UniAPIGeminiImageClient | None = None,
    openai_client: UniAPIImageClient | None = None,
    doubao_client: UniAPIDoubaoImageClient | None = None,
    overwrite: bool = False,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> dict:
    """Execute the two-stage pipeline for a single model.

    Returns a small dict describing the steps that ran (paths, providers, model
    versions, usages). Raises if either stage fails.
    """

    summary: dict = {
        "model": plan.model,
        "provider": plan.provider,
        "stage1": {},
        "stage2": {"top": {}, "pants": {}},
    }

    if plan.provider == "gemini":
        assert gemini_client is not None, "gemini_client is required for Gemini models"
        stage1 = _run_stage1_gemini(
            client=gemini_client,
            image_paths=original_crops,
            output_path=plan.outputs.three_view,
            model=plan.model,
            image_size=plan.image_size_resolved or "2K",
            overwrite=overwrite,
        )
        top_result, pants_result = _run_stage2_gemini(
            client=gemini_client,
            stage1_three_view=plan.outputs.three_view,
            original_crops=original_crops,
            output_top=plan.outputs.top,
            output_pants=plan.outputs.pants,
            model=plan.model,
            image_size=plan.image_size_resolved or "2K",
            overwrite=overwrite,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        aspect = "3:4"
    elif plan.provider == "openai":
        assert openai_client is not None, "openai_client is required for OpenAI models"
        stage1 = _run_stage1_openai(
            client=openai_client,
            image_paths=original_crops,
            output_path=plan.outputs.three_view,
            model=plan.model,
            size=plan.image_size_requested,
            quality=plan.quality,
            overwrite=overwrite,
        )
        top_result, pants_result = _run_stage2_openai(
            client=openai_client,
            stage1_three_view=plan.outputs.three_view,
            original_crops=original_crops,
            output_top=plan.outputs.top,
            output_pants=plan.outputs.pants,
            model=plan.model,
            size=plan.image_size_requested,
            quality=plan.quality,
            overwrite=overwrite,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        aspect = "3:4"
    else:  # doubao
        assert doubao_client is not None, "doubao_client is required for Doubao models"
        stage1 = _run_stage1_doubao(
            client=doubao_client,
            image_paths=original_crops,
            output_path=plan.outputs.three_view,
            model=plan.model,
            size=plan.image_size_requested,
            overwrite=overwrite,
        )
        top_result, pants_result = _run_stage2_doubao(
            client=doubao_client,
            stage1_three_view=plan.outputs.three_view,
            original_crops=original_crops,
            output_top=plan.outputs.top,
            output_pants=plan.outputs.pants,
            model=plan.model,
            size=plan.image_size_requested,
            overwrite=overwrite,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        aspect = "3:4"

    _atomic_write_json(
        plan.outputs.three_view_metadata,
        _stage1_metadata(
            model=plan.model,
            provider=plan.provider,
            inputs=list(original_crops),
            result=stage1,
            image_size_requested=plan.image_size_requested,
            image_size_resolved=plan.image_size_resolved,
            quality=plan.quality,
            aspect_ratio=aspect,
        ),
        overwrite=overwrite,
    )
    _atomic_write_json(
        plan.outputs.top_metadata,
        _stage2_metadata(
            model=plan.model,
            provider=plan.provider,
            inputs=[plan.outputs.three_view, *original_crops],
            result=top_result,
            part="top",
            image_size_requested=plan.image_size_requested,
            image_size_resolved=plan.image_size_resolved,
            quality=plan.quality,
            bake_alpha=bake_alpha,
        ),
        overwrite=overwrite,
    )
    _atomic_write_json(
        plan.outputs.pants_metadata,
        _stage2_metadata(
            model=plan.model,
            provider=plan.provider,
            inputs=[plan.outputs.three_view, *original_crops],
            result=pants_result,
            part="pants",
            image_size_requested=plan.image_size_requested,
            image_size_resolved=plan.image_size_resolved,
            quality=plan.quality,
            bake_alpha=bake_alpha,
        ),
        overwrite=overwrite,
    )

    summary["stage1"]["path"] = str(plan.outputs.three_view)
    summary["stage1"]["metadata"] = str(plan.outputs.three_view_metadata)
    summary["stage2"]["top"]["path"] = str(plan.outputs.top)
    summary["stage2"]["top"]["metadata"] = str(plan.outputs.top_metadata)
    summary["stage2"]["pants"]["path"] = str(plan.outputs.pants)
    summary["stage2"]["pants"]["metadata"] = str(plan.outputs.pants_metadata)
    summary["outputs"] = {
        "three_view": str(plan.outputs.three_view),
        "top": str(plan.outputs.top),
        "pants": str(plan.outputs.pants),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-stage character clothing extraction pipeline "
            "(high-quality turnaround → garment extraction) using a chosen "
            "Gemini or OpenAI image model on UniAPI."
        )
    )
    parser.add_argument("--images", nargs="+", required=True, metavar="IMAGE")
    parser.add_argument("--character-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=ALL_MODELS,
        help=(
            "Model(s) to run; may be repeated to run several models in "
            "sequence (parallel via the orchestrator script). Default: all "
            "five."
        ),
    )
    parser.add_argument(
        "--gemini-base-url",
        default=os.getenv("UNIAPI_GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("UNIAPI_OPENAI_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--doubao-base-url",
        default=os.getenv("UNIAPI_DOUBEO_BASE_URL", DEFAULT_DOUBEO_BASE_URL),
    )
    parser.add_argument("--api-key-env", default="UNIAPI_API_KEY")
    parser.add_argument("--image-size", default="auto")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bake-alpha", action="store_true")
    parser.add_argument("--bake-alpha-threshold", type=int, default=18)
    parser.add_argument("--bake-alpha-softness", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the plan for each requested model and print it.",
    )
    return parser


def _resolve_inputs(image_args: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in image_args:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        resolved.append(path)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        inputs = _resolve_inputs(args.images)
        models = args.model or list(ALL_MODELS)
        for model in models:
            _classify_model(model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.dry_run:
        print(
            f"error: environment variable {args.api_key_env} is not set",
            file=sys.stderr,
        )
        return 2

    plans = [
        plan_two_stage(
            image_paths=inputs,
            output_root=args.output_root,
            character_label=args.character_label,
            model=model,
            image_size=args.image_size,
            quality=args.quality,
        )
        for model in models
    ]

    request_summary = {
        "character_label": args.character_label,
        "output_root": str(args.output_root),
        "input_images": [str(p) for p in inputs],
        "models": [
            {
                "model": p.model,
                "provider": p.provider,
                "image_size_requested": p.image_size_requested,
                "image_size_resolved": p.image_size_resolved,
                "outputs": {
                    "three_view": str(p.outputs.three_view),
                    "top": str(p.outputs.top),
                    "pants": str(p.outputs.pants),
                },
            }
            for p in plans
        ],
    }
    print(json.dumps(request_summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    # Build one client per provider and reuse it across models.
    gemini_client = UniAPIGeminiImageClient(
        api_key,
        base_url=args.gemini_base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    openai_client = UniAPIImageClient(
        api_key,
        base_url=args.openai_base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    doubao_client = UniAPIDoubaoImageClient(
        api_key,
        base_url=getattr(args, "doubao_base_url", None) or DEFAULT_DOUBEO_BASE_URL,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )

    summaries: list[dict] = []
    for plan in plans:
        try:
            summaries.append(
                run_two_stage(
                    plan,
                    original_crops=inputs,
                    gemini_client=gemini_client,
                    openai_client=openai_client,
                    doubao_client=doubao_client,
                    overwrite=args.overwrite,
                    bake_alpha=args.bake_alpha,
                    bake_alpha_threshold=args.bake_alpha_threshold,
                    bake_alpha_softness=args.bake_alpha_softness,
                )
            )
        except Exception as exc:  # pragma: no cover - surface to CLI caller
            summaries.append({"model": plan.model, "error": str(exc)})
            print(
                f"error: model {plan.model} failed: {exc}",
                file=sys.stderr,
            )

    print(json.dumps({"models": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
