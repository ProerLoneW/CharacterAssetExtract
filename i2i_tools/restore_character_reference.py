#!/usr/bin/env python3
"""Restore one high-quality character reference from multiple extracted views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    from .uniapi_gemini_image_client import (
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
        run_gemini_task_cli,
    )
except ImportError:  # Allow: python i2i_tools/restore_character_reference.py
    from uniapi_gemini_image_client import (
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
        run_gemini_task_cli,
    )


CHARACTER_INSTRUCTIONS = """
Create one high-quality, photorealistic character reference image by reconciling
all supplied images as observations of the SAME person. Image 1 is the primary
identity reference; later images provide complementary evidence about the face,
hair, body proportions, clothing, accessories, materials, and colors.

Preserve identity and distinctive traits exactly. Recover plausible fine detail
lost to blur, compression, low resolution, crop boundaries, motion, lighting, or
partial occlusion, but do not beautify, redesign, change age, change ethnicity,
or invent unsupported accessories. Resolve conflicts in favor of repeated visual
evidence and the primary image. Show exactly one person, naturally posed, with a
complete readable silhouette, neutral expression, even studio-like illumination,
plain unobtrusive background, realistic anatomy, sharp facial detail, and faithful
wardrobe. The result is a production reference asset, not a collage, contact sheet,
turnaround, poster, or dramatic scene. Do not add text, labels, logos, borders,
watermarks, extra people, or duplicated body parts.
"""


def restore_character_reference(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    prompt: str,
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: Any,
) -> GeminiRestorationResult:
    """Programmatic multi-image character restoration entry point."""
    generation_options.setdefault("aspect_ratio", "2:3")
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=prompt,
        output_path=output_path,
        task_instructions=CHARACTER_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        generation_options=generation_options,
    )


def main() -> int:
    return run_gemini_task_cli(
        task_name="character",
        task_instructions=CHARACTER_INSTRUCTIONS,
        default_aspect_ratio="2:3",
    )


if __name__ == "__main__":
    raise SystemExit(main())
