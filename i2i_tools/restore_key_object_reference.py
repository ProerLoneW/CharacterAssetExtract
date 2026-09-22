#!/usr/bin/env python3
"""Restore one high-quality key-object reference from multiple extracted crops."""

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


OBJECT_INSTRUCTIONS = """
Create one high-quality, photorealistic key-object reference by reconciling all
supplied images as observations of the SAME physical object. Image 1 is the primary
shape and appearance reference; later images provide complementary views of geometry,
materials, surface wear, construction, colors, patterns, and functional details.

Preserve the object's exact category, proportions, distinctive details, state, and
story-relevant markings. Recover plausible fine detail lost to blur, compression,
low resolution, crop boundaries, motion, glare, and partial occlusion, but do not
redesign, repair damage, add parts, or invent text that cannot be verified. Resolve
conflicts using repeated evidence and the primary image. Show exactly one complete
object, fully visible, centered, sharply focused, with neutral product-reference
lighting and a plain unobtrusive background. The output is a single asset image,
not a collage, exploded diagram, turntable sheet, or scene. Do not add hands, people,
props, labels, logos, borders, watermarks, or duplicate objects unless explicitly
required by the additional user direction.
"""


def restore_key_object_reference(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[str | Path],
    prompt: str,
    output_path: str | Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options: Any,
) -> GeminiRestorationResult:
    """Programmatic multi-image key-object restoration entry point."""
    generation_options.setdefault("aspect_ratio", "1:1")
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=prompt,
        output_path=output_path,
        task_instructions=OBJECT_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        generation_options=generation_options,
    )


def main() -> int:
    return run_gemini_task_cli(
        task_name="key object",
        task_instructions=OBJECT_INSTRUCTIONS,
        default_aspect_ratio="1:1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
