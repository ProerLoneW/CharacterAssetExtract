#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 entry point for atomic, temporally precise key-object proposals.

This intentionally reuses V2 validation and request orchestration while using
the new V5 prompt. Existing V1/V2 code and prompts remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mimo_shot_captioner_v2 as v2


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "shot_caption_prompt_en_semantic_roles_objects_v5.md"
_V2_ANNOTATE_ONE = v2.annotate_one


async def annotate_one(*args: Any, **kwargs: Any) -> dict:
    result = await _V2_ANNOTATE_ONE(*args, **kwargs)
    result["schema_version"] = "shot-caption-atomic-key-objects-v3"
    return result


def main() -> int:
    v2.DEFAULT_PROMPT = DEFAULT_PROMPT
    v2.annotate_one = annotate_one
    return v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
