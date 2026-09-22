#!/usr/bin/env python3
"""Extract isolated top / pants garments from one or more character images.

Given one or more input photos of a single character, this tool asks a
text-to-image / image-to-image model to produce two clean, transparent
-background garment plates:

* **top**    — the upper-body garment (jacket, shirt, sweater, coat, vest, …)
* **pants**  — the lower-body garment (trousers, jeans, skirt-as-pants, …)

The two garments are generated in two independent requests (or in a single
combined 2-panel request when ``--mode both-in-one`` is selected). Each output
is a PNG with a fully transparent background so the garment can be dropped
straight into downstream composites.

The model used is the same Gemini-native multimodal generation pipeline as the
other ``i2i_tools`` scripts (``UniAPIGeminiImageClient``), so prompts are sent
as ordered ``inlineData`` parts plus a text part, and ``responseModalities``
requests ``IMAGE`` only.

Programmatic entry points:

* :func:`extract_character_top`
* :func:`extract_character_pants`
* :func:`extract_character_clothing`

CLI usage::

    export UNIAPI_API_KEY='...'
    python i2i_tools/extract_character_clothing.py \\
        --images character_a.jpg character_b.jpg character_c.jpg \\
        --output outputs/character_clothing

Each garment is written next to the base path:

    outputs/character_clothing_top.png
    outputs/character_clothing_pants.png
    outputs/character_clothing_top.png.json
    outputs/character_clothing_pants.png.json

Add ``--dry-run`` to inspect selected inputs and final prompts without calling
the API; add ``--overwrite`` to replace existing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .uniapi_gemini_image_client import (
        DEFAULT_GEMINI_IMAGE_MODEL,
        GEMINI_IMAGE_MODEL_PROFILES,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        inspect_images,
        restore_gemini_reference,
    )
    from .uniapi_image_client import _atomic_write
except ImportError:  # Allow running as a plain script from i2i_tools/.
    from uniapi_gemini_image_client import (
        DEFAULT_GEMINI_IMAGE_MODEL,
        GEMINI_IMAGE_MODEL_PROFILES,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        inspect_images,
        restore_gemini_reference,
    )
    from uniapi_image_client import _atomic_write

try:  # Pillow is the standard image library; we use it to bake the alpha
    # channel that the Gemini image models do not natively produce.
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is expected to be installed.
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Task instructions
# ---------------------------------------------------------------------------

# Shared guard rails for both garment tasks. We isolate the garment on a
# *fully transparent* background (alpha = 0) so downstream pipelines can
# composite the plate onto any scene without re-matting. The garment must
# preserve its true silhouette, fabric, color, pattern, layering, and story
# -relevant wear or damage; the model must never invent accessories or
# redesign the wardrobe.
_CLOTHING_GUARDRAILS = """
Hard requirements for the output PNG:

* The garment fills the central ~85% of the canvas; it is fully visible with a
  small even margin on all four sides.
* The background MUST be fully transparent (alpha = 0 everywhere except the
  garment pixels). No checkerboard, white card, grey card, gradient, shadow,
  ground, table, mannequin, or environmental detail.
* The garment keeps its exact silhouette, fabric texture, color, pattern,
  print, embroidery, seam, hem, fold, layering, and any story-relevant wear
  (dirt, tear, blood, scorch, repair).
* Recover fine detail lost to blur, compression, low resolution, motion,
  partial occlusion, and harsh lighting — but never redesign, restyle,
  recolor, add accessories, add a second identical garment, or invent text
  or logos that cannot be verified from the references.
* The output is a single garment asset, not a collage, contact sheet,
  fashion editorial, or worn-by-character photo. Do not include the
  character's face, hair, skin, hands, feet, body, mannequin, props, labels,
  borders, or watermarks.
""".strip()


TOP_INSTRUCTIONS = f"""
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME upper-body garment. The first
image is the primary garment reference; later images contribute complementary
fabric, color, pattern, layering, and accessory evidence.

The output must be a single production garment plate showing ONLY the
upper-body garment the character is wearing in the references — typically a
jacket, coat, shirt, sweater, vest, or hoodie. The garment is shown laid out
flat, fully spread and unoccluded, as if prepared for a costume reference.

{_CLOTHING_GUARDRAILS}
""".strip()


PANTS_INSTRUCTIONS = f"""
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME lower-body garment. The first
image is the primary garment reference; later images contribute complementary
fabric, color, pattern, layering, belt-loop, pocket, and footwear-above-the
-hem evidence.

The output must be a single production garment plate showing ONLY the
lower-body garment the character is wearing in the references — typically
trousers, jeans, cargo pants, shorts (treated as pants for this task), or a
skirt. The garment is shown laid out flat, fully spread and unoccluded, as
if prepared for a costume reference. The waistband is at the top of the
canvas, the hem at the bottom, and any belt, drawstring, or waist tie is
preserved.

{_CLOTHING_GUARDRAILS}
""".strip()


COMBINED_INSTRUCTIONS = f"""
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME complete outfit. The first
image is the primary reference; later images contribute complementary fabric,
color, pattern, and layering evidence.

The output must be a single composite production plate showing ONLY the two
requested garments, both isolated and both with fully transparent backgrounds
(alpha = 0 everywhere except the garment pixels). The plate is split into two
halves:

* Top half — the upper-body garment (jacket / coat / shirt / sweater / vest
  / hoodie) laid out flat and fully spread.
* Bottom half — the lower-body garment (trousers / jeans / cargo pants /
  shorts / skirt) laid out flat and fully spread, waistband at the top of its
  half, hem at the bottom.

Both halves share the same canvas but each garment is on its OWN transparent
background; the two halves do not overlap and there is no shared card, table,
ground, or shadow between them. The plate is intended for downstream
compositing, not as a worn-by-character photo.

{_CLOTHING_GUARDRAILS}
""".strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClothingExtractionPlan:
    """Planned inputs / outputs for a single character clothing extraction."""

    character_label: str
    image_paths: list[Path] = field(default_factory=list)
    top_output: Path | None = None
    top_metadata: Path | None = None
    pants_output: Path | None = None
    pants_metadata: Path | None = None
    combined_output: Path | None = None
    combined_metadata: Path | None = None

    def requested_parts(self) -> set[str]:
        parts: set[str] = set()
        if self.top_output is not None:
            parts.add("top")
        if self.pants_output is not None:
            parts.add("pants")
        if self.combined_output is not None:
            parts.add("combined")
        return parts


@dataclass(frozen=True)
class ClothingExtractionResult:
    """Bundle of per-part restoration results returned to programmatic callers."""

    plan: ClothingExtractionPlan
    top: GeminiRestorationResult | None = None
    pants: GeminiRestorationResult | None = None
    combined: GeminiRestorationResult | None = None

    def all_saved_paths(self) -> list[Path]:
        paths: list[Path] = []
        for result in (self.top, self.pants, self.combined):
            if result is None:
                continue
            for saved in result.saved_images:
                paths.append(saved.path)
        return paths


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _resolve_base_paths(
    output: Path,
    *,
    request_top: bool,
    request_pants: bool,
    request_combined: bool,
) -> tuple[Path | None, Path | None, Path | None]:
    """Return the (top, pants, combined) output paths derived from ``output``.

    ``output`` may point at a file (``.png``) — in that case the per-part
    paths are derived by appending ``_top`` / ``_pants`` before the extension.
    If it points at a directory or has no suffix, the per-part files are
    written inside it as ``top.png`` / ``pants.png``.
    """

    suffix = output.suffix.lower()
    if suffix == ".png":
        stem = output.with_suffix("")
        base_dir = output.parent
        top_path = (base_dir / f"{stem.name}_top.png").resolve() if request_top else None
        pants_path = (base_dir / f"{stem.name}_pants.png").resolve() if request_pants else None
        combined_path = (base_dir / f"{stem.name}_combined.png").resolve() if request_combined else None
    else:
        base_dir = output
        top_path = (base_dir / "top.png").resolve() if request_top else None
        pants_path = (base_dir / "pants.png").resolve() if request_pants else None
        combined_path = (base_dir / "combined.png").resolve() if request_combined else None
    return top_path, pants_path, combined_path


def _metadata_for(image_path: Path | None) -> Path | None:
    if image_path is None:
        return None
    return image_path.with_suffix(image_path.suffix + ".json")


def plan_clothing_extraction(
    image_paths: Sequence[str | Path],
    *,
    output: str | Path,
    parts: Iterable[str] = ("top", "pants"),
    character_label: str = "character",
) -> ClothingExtractionPlan:
    """Validate inputs and resolve output paths for a clothing extraction run.

    ``parts`` accepts any non-empty subset of ``{"top", "pants", "combined"}``.
    ``combined`` requests a single plate with both garments stacked; ``top``
    and ``pants`` request independent transparent-background plates.
    """

    normalized_parts = {str(part).strip().lower() for part in parts}
    if not normalized_parts:
        raise ValueError("At least one of 'top', 'pants', or 'combined' must be requested")
    allowed = {"top", "pants", "combined"}
    unknown = normalized_parts - allowed
    if unknown:
        raise ValueError(
            f"Unknown part(s) requested: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )

    inspected = inspect_images(image_paths)
    resolved_image_paths = [image.path for image in inspected]

    top_out, pants_out, combined_out = _resolve_base_paths(
        Path(output).expanduser(),
        request_top="top" in normalized_parts,
        request_pants="pants" in normalized_parts,
        request_combined="combined" in normalized_parts,
    )
    return ClothingExtractionPlan(
        character_label=character_label.strip() or "character",
        image_paths=resolved_image_paths,
        top_output=top_out,
        top_metadata=_metadata_for(top_out),
        pants_output=pants_out,
        pants_metadata=_metadata_for(pants_out),
        combined_output=combined_out,
        combined_metadata=_metadata_for(combined_out),
    )


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def extract_character_top(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: object,
) -> GeminiRestorationResult:
    """Extract the character's upper-body garment as a transparent PNG."""

    generation_options.setdefault("aspect_ratio", "1:1")
    return restore_gemini_reference(
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
        generation_options=generation_options,
    )


def extract_character_pants(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: object,
) -> GeminiRestorationResult:
    """Extract the character's lower-body garment as a transparent PNG."""

    generation_options.setdefault("aspect_ratio", "1:1")
    return restore_gemini_reference(
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
        generation_options=generation_options,
    )


def _extract_and_save_part(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    user_prompt: str,
    task_instructions: str,
    output_path: str | Path,
    asset_description: str | None,
    overwrite: bool,
    bake_alpha_enabled: bool,
    bake_alpha_threshold: int,
    bake_alpha_softness: int,
    generation_options: dict,
) -> GeminiRestorationResult:
    """Run one part of the clothing extraction, including the alpha-bake step.

    The Gemini image models return ``image/jpeg`` (3.x) or ``image/png`` as
    ``image/RGB`` (2.5), neither of which carries an alpha channel even
    though the model agreed to a transparent background. This helper calls
    the model, accepts any ``image/*`` mime, saves the bytes, and then bakes
    the alpha channel via :func:`bake_saved_image` so the final file is a
    proper RGBA PNG ready for compositing.
    """

    try:
        from .uniapi_gemini_image_client import (  # type: ignore[no-redef]
            _normalize_output_path,
            compose_prompt,
            extract_gemini_outputs,
        )
    except ImportError:  # Plain script: fall back to absolute imports.
        from uniapi_gemini_image_client import (  # type: ignore[no-redef]
            _normalize_output_path,
            compose_prompt,
            extract_gemini_outputs,
        )

    resolved_output = _normalize_output_path(output_path)
    inspected = inspect_images(image_paths)
    _validate_part_outputs(inspected, [resolved_output])
    final_prompt = compose_prompt(
        task_instructions,
        user_prompt,
        asset_description=asset_description,
    )
    response, _ = client.generate(
        image_paths=image_paths,
        prompt=final_prompt,
        **generation_options,
    )
    images, texts = extract_gemini_outputs(response)
    if not images:
        from uniapi_gemini_image_client import UniAPIImageError
        raise UniAPIImageError("Gemini response contains no images for clothing part")
    (content, mime_type), path = images[0], resolved_output
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    # Save raw bytes first; the alpha-bake step will re-encode as RGBA PNG.
    _atomic_write(path, content, overwrite=overwrite)
    bake_info: dict[str, object]
    if bake_alpha_enabled:
        bake_info = bake_saved_image(
            path,
            threshold=bake_alpha_threshold,
            softness=bake_alpha_softness,
        )
    else:
        bake_info = {"baked": False, "reason": "disabled"}
    raw_sha = hashlib.sha256(content).hexdigest()
    final_bytes = path.read_bytes()
    from uniapi_gemini_image_client import SavedImage, GeminiRestorationResult
    saved = [
        SavedImage(
            path=path,
            sha256=hashlib.sha256(final_bytes).hexdigest(),
            size_bytes=len(final_bytes),
            revised_prompt=None,
        )
    ]
    return GeminiRestorationResult(
        final_prompt=final_prompt,
        input_images=inspected,
        saved_images=saved,
        response_text=texts,
        response={
            "modelVersion": response.get("modelVersion"),
            "usageMetadata": response.get("usageMetadata"),
            "_raw_mime": mime_type,
            "_raw_sha256": raw_sha,
            "_alpha_bake": bake_info,
            "_response": response,
        },
    )


def _validate_part_outputs(
    inputs: Sequence,
    outputs: Sequence[Path],
) -> None:
    """Local stand-in for ``_validate_output_targets`` that does not need it
    for our simple use case (one output path, no metadata clash).
    """

    try:
        from .uniapi_image_client import _validate_output_targets
    except ImportError:
        from uniapi_image_client import _validate_output_targets
    _validate_output_targets([i.path for i in inputs], outputs)


def extract_character_combined(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: object,
) -> GeminiRestorationResult:
    """Generate a single plate with both top and pants, each on transparent bg."""

    generation_options.setdefault("aspect_ratio", "2:3")
    return restore_gemini_reference(
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
        generation_options=generation_options,
    )


def estimate_background_color(
    img: "Image.Image",
    *,
    border_fraction: float = 0.02,
) -> tuple[int, int, int] | None:
    """Estimate the background color from image borders.

    Returns the median of all border pixels (top/bottom rows + left/right
    columns) after skipping the outermost ``border_fraction`` of each side
    so garment pixels that touch the edge don't poison the estimate.
    Returns ``None`` if the image is too small to sample.
    """

    if Image is None:
        return None
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < 8 or h < 8:
        return None
    skip_x = max(1, int(w * border_fraction))
    skip_y = max(1, int(h * border_fraction))
    border_px: list[tuple[int, int, int]] = []
    for x in range(skip_x, w - skip_x):
        border_px.append(rgb.getpixel((x, skip_y)))  # type: ignore[arg-type]
        border_px.append(rgb.getpixel((x, h - 1 - skip_y)))  # type: ignore[arg-type]
    for y in range(skip_y, h - skip_y):
        border_px.append(rgb.getpixel((skip_x, y)))  # type: ignore[arg-type]
        border_px.append(rgb.getpixel((w - 1 - skip_x, y)))  # type: ignore[arg-type]
    if not border_px:
        return None
    r = sorted(p[0] for p in border_px)[len(border_px) // 2]
    g = sorted(p[1] for p in border_px)[len(border_px) // 2]
    b = sorted(p[2] for p in border_px)[len(border_px) // 2]
    return (r, g, b)


def bake_alpha(
    img: "Image.Image",
    *,
    threshold: int = 18,
    softness: int = 32,
) -> "Image.Image":
    """Replace near-background pixels with full transparency.

    ``threshold`` is the per-channel color distance below which alpha is
    forced to 0; ``softness`` is the additional distance over which alpha
    ramps linearly from 0 to 255. The L-inf (max channel) distance is used
    so that neutral gray card pixels are removed while saturated garment
    pixels stay fully opaque.
    """

    if Image is None:  # pragma: no cover - Pillow is expected to be installed.
        raise RuntimeError("Pillow is required for bake_alpha; install with `pip install pillow`")
    rgb = img.convert("RGB")
    bg = estimate_background_color(rgb) or (255, 255, 255)
    w, h = rgb.size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    src_px = rgb.load()
    dst_px = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b = src_px[x, y]
            dist = max(abs(r - bg[0]), abs(g - bg[1]), abs(b - bg[2]))
            if dist <= threshold:
                alpha = 0
            elif dist >= threshold + softness:
                alpha = 255
            else:
                alpha = int(255 * (dist - threshold) / softness)
            dst_px[x, y] = (r, g, b, alpha)
    return out


def bake_saved_image(
    image_path: Path,
    *,
    threshold: int = 18,
    softness: int = 32,
) -> dict[str, object]:
    """Open ``image_path`` (any PIL-readable format), bake alpha, re-save as RGBA PNG.

    Returns a small dict describing the operation so callers can include it
    in the metadata sidecar. If Pillow is unavailable the original file is
    left untouched and ``{"baked": False, "reason": "pillow_unavailable"}``
    is returned.
    """

    if Image is None:
        return {"baked": False, "reason": "pillow_unavailable"}
    with Image.open(image_path) as raw:
        bg = estimate_background_color(raw)
        rgba = bake_alpha(raw, threshold=threshold, softness=softness)
    rgba.save(image_path, format="PNG")
    return {
        "baked": True,
        "estimated_background_color": bg,
        "threshold": threshold,
        "softness": softness,
        "final_mode": "RGBA",
    }


def _build_common_options(
    *,
    model: str,
    image_size: str | None,
    temperature: float | None,
) -> dict:
    options: dict = {"model": model}
    if image_size is not None:
        options["image_size"] = image_size
    if temperature is not None:
        options["temperature"] = temperature
    return options


def _build_asset_description(plan: ClothingExtractionPlan) -> str:
    parts = sorted(plan.requested_parts())
    return (
        f"{plan.character_label}: extracting {', '.join(parts)} from "
        f"{len(plan.image_paths)} input image(s)."
    )


def _write_clothing_metadata(
    plan: ClothingExtractionPlan,
    result: GeminiRestorationResult,
    metadata_path: Path,
    *,
    task: str,
    overwrite: bool,
) -> None:
    response = result.response
    raw_mime = response.get("_raw_mime")
    raw_sha = response.get("_raw_sha256")
    alpha_bake = response.get("_alpha_bake")
    full_response = response.get("_response", response)
    metadata = {
        "schema_version": "character-clothing-extraction-v1",
        "request": {
            "task": task,
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
        },
        "response": {
            "model_version": full_response.get("modelVersion"),
            "usage": full_response.get("usageMetadata"),
            "text": result.response_text,
            "finish_reasons": [
                candidate.get("finishReason")
                for candidate in full_response.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get("finishReason")
            ],
            "raw_mime": raw_mime,
            "raw_sha256": raw_sha,
        },
        "alpha_bake": alpha_bake,
        "outputs": [
            {
                "path": str(item.path),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in result.saved_images
        ],
    }
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(metadata_path, encoded, overwrite=overwrite)


def execute_plan(
    plan: ClothingExtractionPlan,
    *,
    client: UniAPIGeminiImageClient,
    overwrite: bool = False,
    model: str = DEFAULT_GEMINI_IMAGE_MODEL,
    image_size: str | None = "auto",
    temperature: float | None = None,
    bake_alpha: bool = False,
    bake_alpha_threshold: int = 18,
    bake_alpha_softness: int = 32,
) -> ClothingExtractionResult:
    """Run all requested parts of a clothing extraction plan.

    Parts are generated sequentially, top first, then pants, then combined.
    Each part is independent and uses a fresh API call so a failure on one
    part does not block the others. When ``bake_alpha`` is true, each
    saved PNG is post-processed to derive an alpha channel from the
    gray card background the model leaves behind. The default is
    ``False``: callers who want a real transparent background should
    pass ``bake_alpha=True`` (or use the ``--bake-alpha`` CLI flag).
    """

    common_options = _build_common_options(
        model=model,
        image_size=image_size,
        temperature=temperature,
    )
    asset_description = _build_asset_description(plan)

    top_result: GeminiRestorationResult | None = None
    pants_result: GeminiRestorationResult | None = None
    combined_result: GeminiRestorationResult | None = None

    if plan.top_output is not None:
        top_result = _run_part(
            part="top",
            plan=plan,
            client=client,
            asset_description=asset_description,
            overwrite=overwrite,
            common_options=common_options,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        if plan.top_metadata is not None:
            _write_clothing_metadata(
                plan,
                top_result,
                plan.top_metadata,
                task="character_clothing_top",
                overwrite=overwrite,
            )

    if plan.pants_output is not None:
        pants_result = _run_part(
            part="pants",
            plan=plan,
            client=client,
            asset_description=asset_description,
            overwrite=overwrite,
            common_options=common_options,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        if plan.pants_metadata is not None:
            _write_clothing_metadata(
                plan,
                pants_result,
                plan.pants_metadata,
                task="character_clothing_pants",
                overwrite=overwrite,
            )

    if plan.combined_output is not None:
        combined_result = _run_part(
            part="combined",
            plan=plan,
            client=client,
            asset_description=asset_description,
            overwrite=overwrite,
            common_options=common_options,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
        )
        if plan.combined_metadata is not None:
            _write_clothing_metadata(
                plan,
                combined_result,
                plan.combined_metadata,
                task="character_clothing_combined",
                overwrite=overwrite,
            )

    return ClothingExtractionResult(
        plan=plan,
        top=top_result,
        pants=pants_result,
        combined=combined_result,
    )


def _run_part(
    *,
    part: str,
    plan: ClothingExtractionPlan,
    client: UniAPIGeminiImageClient,
    asset_description: str,
    overwrite: bool,
    common_options: dict,
    bake_alpha: bool,
    bake_alpha_threshold: int,
    bake_alpha_softness: int,
) -> GeminiRestorationResult:
    """Dispatch to the right per-part extractor and return its result."""

    if part == "top":
        output_path = plan.top_output
        return _extract_and_save_part(
            client,
            image_paths=plan.image_paths,
            user_prompt=(
                "Lay out this character's upper-body garment flat, fully "
                "spread and unoccluded, on a fully transparent background "
                "(alpha = 0). Show only the top - no body, no pants, no "
                "shoes, no mannequin, no props, no card, no shadow. "
                "Preserve exact fabric, color, pattern, and condition."
            ),
            task_instructions=TOP_INSTRUCTIONS,
            output_path=output_path,
            asset_description=asset_description,
            overwrite=overwrite,
            bake_alpha_enabled=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
            generation_options=common_options,
        )
    if part == "pants":
        output_path = plan.pants_output
        return _extract_and_save_part(
            client,
            image_paths=plan.image_paths,
            user_prompt=(
                "Lay out this character's lower-body garment flat, fully "
                "spread and unoccluded, on a fully transparent background "
                "(alpha = 0). Show only the pants - no body, no top, no "
                "shoes, no mannequin, no props, no card, no shadow. "
                "Waistband at the top of the canvas, hem at the bottom. "
                "Preserve exact fabric, color, pattern, and condition."
            ),
            task_instructions=PANTS_INSTRUCTIONS,
            output_path=output_path,
            asset_description=asset_description,
            overwrite=overwrite,
            bake_alpha_enabled=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
            generation_options=common_options,
        )
    if part == "combined":
        output_path = plan.combined_output
        return _extract_and_save_part(
            client,
            image_paths=plan.image_paths,
            user_prompt=(
                "Produce a two-half plate: the top half shows the "
                "upper-body garment laid flat, the bottom half shows the "
                "lower-body garment laid flat (waistband up, hem down). "
                "Each half is on its OWN fully transparent background "
                "(alpha = 0); do not draw a shared card, table, ground, or "
                "shadow. Preserve exact fabrics, colors, patterns, and "
                "condition for both garments."
            ),
            task_instructions=COMBINED_INSTRUCTIONS,
            output_path=output_path,
            asset_description=asset_description,
            overwrite=overwrite,
            bake_alpha_enabled=bake_alpha,
            bake_alpha_threshold=bake_alpha_threshold,
            bake_alpha_softness=bake_alpha_softness,
            generation_options=common_options,
        )
    raise ValueError(f"Unknown part: {part}")


# ---------------------------------------------------------------------------
# Discovery (best-effort grouping for batch runs)
# ---------------------------------------------------------------------------


def discover_character_image_groups(
    root: Path,
    *,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> list[list[Path]]:
    """Group images under ``root`` by their parent directory.

    Each subdirectory of ``root`` (or ``root`` itself if it directly contains
    images) is treated as one character group. Returns a list of image
    batches ready to feed into :func:`plan_clothing_extraction`.
    """

    if not root.is_dir():
        return []
    groups: list[list[Path]] = []
    # If the root itself contains images, treat it as a single group.
    direct = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if direct:
        groups.append(direct)
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            path
            for path in subdir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if images:
            groups.append(images)
    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _coerce_parts(value: str) -> list[str]:
    parts = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--parts must be a comma-separated list with at least one of: top, pants, combined"
        )
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the character's top and/or pants as transparent-background "
            "PNG plates using Gemini multi-image generation."
        )
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="IMAGE",
        help=(
            "One or more reference photos of the SAME character wearing the "
            "outfit to extract. Put the clearest, most complete reference first."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path. If it ends in .png, per-part files are written as "
            "<stem>_top.png / <stem>_pants.png / <stem>_combined.png beside it. "
            "Otherwise it is treated as a directory and outputs are written as "
            "<output>/top.png / <output>/pants.png / <output>/combined.png."
        ),
    )
    parser.add_argument(
        "--parts",
        type=_coerce_parts,
        default=["top", "pants"],
        help=(
            "Comma-separated list of parts to generate. Allowed: top, pants, "
            "combined. Default: top,pants. 'combined' produces a single plate "
            "with both garments stacked."
        ),
    )
    parser.add_argument(
        "--character-label",
        default="character",
        help="Free-text label used in metadata to identify the character.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("UNIAPI_IMAGE_PROVIDER", "gemini"),
        choices=["gemini", "openai", "doubao"],
        help=(
            "Image provider family: 'gemini' (UniAPI Gemini), "
            "'openai' (UniAPI OpenAI Images), or "
            "'doubao' (UniAPI proxy for Volcengine ARK Seedream)."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("UNIAPI_GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL),
        help=(
            "Image model ID. Gemini choices: "
            + ", ".join(GEMINI_IMAGE_MODEL_PROFILES)
            + "; OpenAI choices: gpt-image-2, gpt-image-2.5-sunburst, gpt-image-1"
            + "; Doubao choices: doubao-seedream-5-0-260128, doubao-seedream-5-0-lite-260128, "
            "doubao-seedream-4-5-251128, doubao-seedream-4-0-250828."
        ),
    )
    parser.add_argument(
        "--image-size",
        default="auto",
        help=(
            "Gemini: one of auto/1K/2K/4K. OpenAI: a literal size such as "
            "1024x1024, 1024x1536, or auto. Doubao: 2K / 3K / 1024x1024 etc. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL (default: UniAPI gemini or v1 endpoint).",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=None,
        choices=sorted({"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}),
        help=(
            "Override the default aspect ratio. Default: 1:1 for top and pants, "
            "2:3 for combined. Doubao ignores this and uses --image-size directly."
        ),
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=["auto", "low", "medium", "high"],
        help="OpenAI-only: image generation quality.",
    )
    parser.add_argument(
        "--input-fidelity",
        default="high",
        choices=["high", "low", "omit"],
        help="OpenAI-only: input fidelity passed through /v1/images/edits.",
    )
    parser.add_argument(
        "--watermark",
        choices=["yes", "no"],
        default="no",
        help="Doubao-only: add the Volcengine AI watermark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Doubao-only: deterministic seed.",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "jpeg"],
        default="png",
        help="Doubao-only: output container format.",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--api-key-env",
        default="UNIAPI_API_KEY",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned inputs, prompts, and outputs without calling the API.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Request IMAGE-only responses (no accompanying text) for faster turns.",
    )
    parser.add_argument(
        "--bake-alpha",
        action="store_true",
        help=(
            "Enable the local post-processing step that bakes a real alpha "
            "channel into each output PNG. By default the script leaves "
            "the model's output untouched, so the saved files reflect "
            "exactly what the API returned. Use this flag if you need "
            "true RGBA transparency in the final PNGs."
        ),
    )
    parser.add_argument(
        "--bake-alpha-threshold",
        type=int,
        default=18,
        help=(
            "Per-channel color distance below which a pixel is treated as "
            "background (alpha = 0). Default: 18."
        ),
    )
    parser.add_argument(
        "--bake-alpha-softness",
        type=int,
        default=32,
        help=(
            "Additional color distance over which alpha ramps from 0 to "
            "255. Default: 32."
        ),
    )
    return parser


def _summarise_inputs(image_paths: Sequence[Path]) -> list[dict]:
    summary: list[dict] = []
    for index, path in enumerate(image_paths, start=1):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        summary.append({"index": index, "path": str(path), "size_bytes": size_bytes})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        plan = plan_clothing_extraction(
            args.images,
            output=args.output,
            parts=args.parts,
            character_label=args.character_label,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Provider dispatch: OpenAI uses /v1/images/edits via UniAPIImageClient;
    # Gemini uses generateContent via UniAPIGeminiImageClient;
    # Doubao uses /v1/images/generations (JSON) via UniAPIDoubaoImageClient.
    if args.provider == "openai":
        return _main_openai(args, plan)
    if args.provider == "doubao":
        return _main_doubao(args, plan)

    # If the user gave an override aspect ratio, push it into the plan by
    # baking it into a local copy of the generation options at execute time.
    # We resolve conflicts up-front so dry-run shows the final values.
    aspect_override = args.aspect_ratio
    common_options_preview: dict = {"model": args.model, "image_size": args.image_size}
    if aspect_override is not None:
        common_options_preview["aspect_ratio"] = aspect_override
    if args.temperature is not None:
        common_options_preview["temperature"] = args.temperature

    outputs: dict[str, Path | None] = {
        "top": plan.top_output,
        "pants": plan.pants_output,
        "combined": plan.combined_output,
    }
    output_preview = {part: (str(path) if path else None) for part, path in outputs.items()}

    request_summary = {
        "task": "character_clothing_extraction",
        "provider": "gemini",
        "model": args.model,
        "image_size_requested": args.image_size,
        "parts_requested": sorted(plan.requested_parts()),
        "inputs": _summarise_inputs(plan.image_paths),
        "outputs": output_preview,
        "common_generation_options": common_options_preview,
        "bake_alpha": args.bake_alpha,
        "bake_alpha_threshold": args.bake_alpha_threshold,
        "bake_alpha_softness": args.bake_alpha_softness,
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

    client = UniAPIGeminiImageClient(
        api_key,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )

    # Re-attach the override to the call by passing per-part generation_options.
    # We rebuild a small wrapper plan here so execute_plan can stay simple.
    bake_alpha = args.bake_alpha
    if aspect_override is not None or args.temperature is not None:
        # Patch the per-part extract functions via generation_options in
        # execute_plan. To do that cleanly, we just rerun a custom branch.
        common_options = _build_common_options(
            model=args.model,
            image_size=args.image_size,
            temperature=args.temperature,
        )
        if aspect_override is not None:
            common_options["aspect_ratio"] = aspect_override
        asset_description = _build_asset_description(plan)

        top_result = pants_result = combined_result = None
        for part_name in ("top", "pants", "combined"):
            output_path = getattr(plan, f"{part_name}_output")
            metadata_path = getattr(plan, f"{part_name}_metadata")
            if output_path is None:
                continue
            sub_result = _run_part(
                part=part_name,
                plan=plan,
                client=client,
                asset_description=asset_description,
                overwrite=args.overwrite,
                common_options=common_options,
                bake_alpha=bake_alpha,
                bake_alpha_threshold=args.bake_alpha_threshold,
                bake_alpha_softness=args.bake_alpha_softness,
            )
            if part_name == "top":
                top_result = sub_result
            elif part_name == "pants":
                pants_result = sub_result
            else:
                combined_result = sub_result
            if metadata_path is not None:
                _write_clothing_metadata(
                    plan,
                    sub_result,
                    metadata_path,
                    task=f"character_clothing_{part_name}",
                    overwrite=args.overwrite,
                )
        result = ClothingExtractionResult(
            plan=plan,
            top=top_result,
            pants=pants_result,
            combined=combined_result,
        )
    else:
        result = execute_plan(
            plan,
            client=client,
            overwrite=args.overwrite,
            model=args.model,
            image_size=args.image_size,
            temperature=args.temperature,
            bake_alpha=bake_alpha,
            bake_alpha_threshold=args.bake_alpha_threshold,
            bake_alpha_softness=args.bake_alpha_softness,
        )

    saved_summary: list[dict] = []
    for label, sub in (
        ("top", result.top),
        ("pants", result.pants),
        ("combined", result.combined),
    ):
        if sub is None:
            continue
        for item in sub.saved_images:
            saved_summary.append(
                {"part": label, "path": str(item.path), "sha256": item.sha256}
            )
    print(json.dumps({"saved": saved_summary}, ensure_ascii=False, indent=2))
    return 0


def _main_openai(args: argparse.Namespace, plan: ClothingExtractionPlan) -> int:
    """OpenAI branch of the CLI: forwards to :mod:`extract_character_clothing_openai`."""
    from extract_character_clothing_openai import (
        OPENAI_IMAGE_MODEL_PROFILES,
        _size_for_aspect,
        execute_plan_openai,
    )
    from uniapi_image_client import UniAPIImageClient, UniAPIImageError

    if args.model not in OPENAI_IMAGE_MODEL_PROFILES:
        print(
            f"error: model {args.model!r} is not a known OpenAI image model; "
            f"expected one of: {', '.join(sorted(OPENAI_IMAGE_MODEL_PROFILES))}",
            file=sys.stderr,
        )
        return 2

    request_summary = {
        "task": "character_clothing_extraction",
        "provider": "openai",
        "model": args.model,
        "requested_size": args.image_size,
        "resolved_size_top": _size_for_aspect("1:1", args.model, args.image_size),
        "resolved_size_combined": _size_for_aspect("2:3", args.model, args.image_size),
        "quality": args.quality,
        "input_fidelity": args.input_fidelity,
        "aspect_ratio": args.aspect_ratio,
        "parts_requested": sorted(plan.requested_parts()),
        "inputs": _summarise_inputs(plan.image_paths),
        "outputs": {
            "top": str(plan.top_output) if plan.top_output else None,
            "pants": str(plan.pants_output) if plan.pants_output else None,
            "combined": str(plan.combined_output) if plan.combined_output else None,
        },
        "bake_alpha": args.bake_alpha,
        "bake_alpha_threshold": args.bake_alpha_threshold,
        "bake_alpha_softness": args.bake_alpha_softness,
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

    base_url = args.base_url or "https://api.uniapi.io/v1"
    client = UniAPIImageClient(
        api_key,
        base_url=base_url,
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
            size=args.image_size,
            quality=args.quality,
            background="transparent",
            output_format="png",
            input_fidelity=fidelity,
            response_format=None,
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


def _main_doubao(args: argparse.Namespace, plan: ClothingExtractionPlan) -> int:
    """Doubao branch of the CLI: forwards to :mod:`extract_character_clothing_doubao`."""
    from extract_character_clothing_doubao import (
        DOUBAO_IMAGE_MODEL_PROFILES,
        UniAPIDoubaoImageClient,
        _size_for_part,
        execute_plan_doubao,
    )
    from uniapi_doubao_image_client import UniAPIImageError

    if args.model not in DOUBAO_IMAGE_MODEL_PROFILES:
        print(
            f"error: model {args.model!r} is not a known Doubao image model; "
            f"expected one of: {', '.join(sorted(DOUBAO_IMAGE_MODEL_PROFILES))}",
            file=sys.stderr,
        )
        return 2

    request_summary = {
        "task": "character_clothing_extraction",
        "provider": "doubao",
        "model": args.model,
        "requested_size": args.image_size,
        "resolved_size_top": _size_for_part(args.model, "top", args.image_size),
        "resolved_size_combined": _size_for_part(args.model, "combined", args.image_size),
        "watermark": args.watermark == "yes",
        "output_format": args.output_format,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "parts_requested": sorted(plan.requested_parts()),
        "inputs": _summarise_inputs(plan.image_paths),
        "outputs": {
            "top": str(plan.top_output) if plan.top_output else None,
            "pants": str(plan.pants_output) if plan.pants_output else None,
            "combined": str(plan.combined_output) if plan.combined_output else None,
        },
        "bake_alpha": args.bake_alpha,
        "bake_alpha_threshold": args.bake_alpha_threshold,
        "bake_alpha_softness": args.bake_alpha_softness,
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

    base_url = args.base_url or "https://api.uniapi.io/v1"
    client = UniAPIDoubaoImageClient(
        api_key,
        base_url=base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    try:
        results = execute_plan_doubao(
            plan,
            client=client,
            overwrite=args.overwrite,
            model=args.model,
            size=args.image_size,
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
