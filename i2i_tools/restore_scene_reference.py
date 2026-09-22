#!/usr/bin/env python3
"""Restore one high-quality clean scene reference from multiple extracted frames."""

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
except ImportError:  # Allow direct script execution.
    from uniapi_gemini_image_client import (
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
        run_gemini_task_cli,
    )


SCENE_INSTRUCTIONS = """
Create one high-quality, photorealistic clean scene reference by reconciling all
supplied images as observations of the SAME physical location. Image 1 defines the
main viewpoint and composition; later images reveal occluded structure, materials,
objects, spatial layout, and lighting.

Preserve the location's architecture, geometry, object placement, proportions,
palette, period, weather, and time-of-day evidence. Recover detail lost to blur,
compression, low resolution, foreground obstruction, and incomplete framing. Remove
people and transient foreground occluders, filling exposed areas only with background
that is consistent with the auxiliary views. Do not redesign, redecorate, widen the
room, move structural objects, or merge incompatible viewpoints. Use natural,
balanced illumination and a clear production-design reference look. The output must
be one coherent scene image, not a collage, panorama montage, floor plan, or scene
board. Do not add text, labels, logos, borders, watermarks, characters, or invented
story action.
"""


def restore_scene_reference(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    prompt: str,
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: Any,
) -> GeminiRestorationResult:
    """Programmatic multi-image scene restoration entry point."""
    generation_options.setdefault("aspect_ratio", "3:2")
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=prompt,
        output_path=output_path,
        task_instructions=SCENE_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        generation_options=generation_options,
    )


def main() -> int:
    return run_gemini_task_cli(
        task_name="scene",
        task_instructions=SCENE_INSTRUCTIONS,
        default_aspect_ratio="3:2",
    )


if __name__ == "__main__":
    raise SystemExit(main())
