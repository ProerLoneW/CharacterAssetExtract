#!/usr/bin/env python3
"""Restore per-character three-view + costume reference sheets from character assets.

The character visual bank produced by the Stage 4 pipeline stores per-character
observations under::

    workspace/.../characters/characters/<semantic_scope>/<subject_id>/visual/
        representative/{primary_face,primary_appearance}.jpg
        face/<view>/*.jpg
        body/<view>/<crop_type>/<APP_ID>/CHR*_BODY_*.jpg
        character.json | appearance_states.json | assets.json

Those observations are cropped, partial, and often share a single appearance
state (costume). This script consumes them and produces two reference assets
per character appearance state via Gemini multi-image restoration:

* ``three_view`` — a clean character turnaround with front, 3/4, side and back
  views of the SAME identity and SAME costume, suitable for downstream image /
  video generation. Default aspect 3:4.
* ``costume`` — the costume isolated on a neutral background, presented as a
  production costume reference (no person / no mannequin face). Default aspect
  1:1.

Inputs are grouped by ``appearance_state_id`` so a character with multiple
outfits produces one pair of references per outfit. Output filenames follow
the existing bank conventions::

    reference/<CHR_ID>_APP<n>/<CHR_ID>_APP<n>_three_view.png
    reference/<CHR_ID>_APP<n>/<CHR_ID>_APP<n>_costume.png

The module also exposes programmatic helpers (``plan_character_restorations``,
``restore_character_three_view`` and ``restore_character_costume``) so the same
logic can be wired into a larger pipeline.

Usage::

    export UNIAPI_API_KEY='...'
    python i2i_tools/restore_character_three_view_and_costume.py \\
        --character-root workspace/.../characters/characters \\
        --character CHR002 --character CHR005 \\
        --output-root workspace/.../characters/reference

Add ``--dry-run`` to inspect selected inputs and final prompts without calling
the API; add ``--overwrite`` to replace existing outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .uniapi_gemini_image_client import (
        GEMINI_IMAGE_MODEL_PROFILES,
        DEFAULT_GEMINI_IMAGE_MODEL,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
        inspect_images,
    )
    from .uniapi_image_client import _atomic_write
except ImportError:  # Allow running as a plain script from i2i_tools/.
    from uniapi_gemini_image_client import (
        GEMINI_IMAGE_MODEL_PROFILES,
        DEFAULT_GEMINI_IMAGE_MODEL,
        GeminiRestorationResult,
        UniAPIGeminiImageClient,
        restore_gemini_reference,
        inspect_images,
    )
    from uniapi_image_client import _atomic_write


# ---------------------------------------------------------------------------
# Task instructions
# ---------------------------------------------------------------------------

THREE_VIEW_INSTRUCTIONS = """
Create one high-quality, photorealistic character turnaround sheet by
reconciling all supplied images as observations of the SAME person wearing the
SAME costume. The first image defines the primary identity reference; later
images contribute complementary face, hair, body proportion, wardrobe, and
material evidence.

The output must be a single, production-grade character reference showing
EXACTLY this one person in EXACTLY this costume, rendered as a clean
three-quarter / side / back / front turnaround on a plain neutral background
with even, studio-like illumination. The four poses must read as the same
person: identical identity (face, skin, hair, age, ethnicity), identical
body type and posture proportions, identical wardrobe (every garment,
accessory, color, pattern, texture, and footwear). Recover fine detail lost
to blur, compression, low resolution, motion, crop, occlusion, and harsh
lighting — but never redesign, restyle, swap outfits, change age, change
ethnicity, or invent accessories. The reference is not a collage, storyboard,
poster, or dramatic scene; do not add text, labels, logos, borders,
watermarks, extra people, or duplicated limbs.
""".strip()


COSTUME_INSTRUCTIONS = """
Create one high-quality, photorealistic costume reference sheet by
reconciling all supplied images as observations of the SAME wardrobe worn by
the SAME character. The first image is the primary costume reference;
later images contribute complementary garment, accessory, fabric, color,
pattern, layering, and footwear evidence.

The output must be a single production costume plate showing the complete
outfit — every garment, layering piece, accessory, and footwear — arranged
neatly on a plain neutral background, as if laid out for a costume
department reference: jacket / top centered, shirt underneath visible,
trousers / skirt folded beneath, belt / accessories in a tidy row, shoes at
the bottom. Preserve the costume's exact category, color, fabric texture,
pattern, condition, and story-relevant wear or damage. Recover detail lost
to blur, compression, low resolution, motion, partial occlusion, and
harsh lighting — but do not redesign, recolor, repair damage, add
accessories, or invent text that cannot be verified from the references.
The reference is not a worn-by-character photo, a collage, mood board, or
fashion editorial; do not include the character's face, hair, skin, hands,
body, mannequin head, props, labels, logos, borders, or watermarks.
""".strip()


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterAsset:
    """A single reference image inside a character's visual bank."""

    path: Path
    kind: str  # "face" or "body"
    view: str  # "front" / "side" / "back" / "other" / "unknown"
    crop_type: str | None  # "partial" / "upper_body" / "full_body" or None
    appearance_state_id: str | None
    asset_id: str | None
    quality_score: float


@dataclass(frozen=True)
class CharacterVisualBank:
    """Resolved per-character visual inputs grouped by appearance state."""

    character_id: str
    semantic_scope_id: str
    subject_id: str
    visual_root: Path
    representative_face: Path | None
    representative_appearance: Path | None
    appearance_state_ids: list[str] = field(default_factory=list)
    assets_by_state: dict[str, list[CharacterAsset]] = field(default_factory=dict)

    def asset_count(self, state_id: str) -> int:
        return len(self.assets_by_state.get(state_id, []))


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _coerce_view(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _coerce_crop_type(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _coerce_score(components: dict, fallback: float = 0.0) -> float:
    for key in ("quality_score", "score"):
        value = components.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return fallback


def _assets_from_gallery(
    gallery: dict,
    root: Path,
    *,
    kind: str,
) -> list[CharacterAsset]:
    """Flatten ``character.json`` gallery entries into ``CharacterAsset`` rows."""

    rows: list[CharacterAsset] = []
    for view_key, items in (gallery or {}).items():
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            registry_path = entry.get("registry_path") or entry.get("path")
            if not registry_path:
                continue
            path = Path(registry_path)
            if not path.is_absolute():
                path = (root / registry_path).resolve()
            components = entry.get("quality_components") or {}
            rows.append(
                CharacterAsset(
                    path=path,
                    kind=kind,
                    view=_coerce_view(entry.get("view") or view_key),
                    crop_type=_coerce_crop_type(entry.get("crop_type")),
                    appearance_state_id=entry.get("appearance_state_id"),
                    asset_id=entry.get("asset_id"),
                    quality_score=_coerce_score(
                        {**entry, "quality_score": entry.get("quality_score")},
                        components.get("sharpness", 0.0),
                    ),
                )
            )
    return rows


def _assets_from_appearance_states(
    appearance_payload: dict,
    character_payload: dict,
    visual_root: Path,
) -> tuple[list[str], dict[str, list[CharacterAsset]]]:
    """Build appearance_state_id -> [CharacterAsset] from appearance_states.json.

    ``appearance_states.json`` records per-state ``body_assets`` and
    ``face_assets`` paths. We cross-reference with character.json's gallery so
    we get view / crop_type / quality metadata for each path. Paths that exist
    in appearance_states but not in the gallery still pass through, with view
    ``unknown`` and quality 0.0.
    """

    by_path: dict[Path, CharacterAsset] = {}
    for gallery in (character_payload.get("face_gallery"), character_payload.get("body_gallery")):
        if not isinstance(gallery, dict):
            continue
        kind = "face" if gallery is character_payload.get("face_gallery") else "body"
        for asset in _assets_from_gallery(gallery, visual_root, kind=kind):
            by_path[asset.path.resolve()] = asset

    states = appearance_payload.get("appearance_states") or []
    state_ids: list[str] = []
    grouped: dict[str, list[CharacterAsset]] = {}
    for state in states:
        if not isinstance(state, dict):
            continue
        state_id = state.get("state_id")
        if not state_id:
            continue
        state_ids.append(state_id)
        rows: list[CharacterAsset] = []
        for kind_key, kind_label in (("body_assets", "body"), ("face_assets", "face")):
            for raw_path in state.get(kind_key) or []:
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (visual_root / raw_path).resolve()
                key = path.resolve()
                if key in by_path:
                    rows.append(by_path[key])
                else:
                    rows.append(
                        CharacterAsset(
                            path=path,
                            kind=kind_label,
                            view="unknown",
                            crop_type=None,
                            appearance_state_id=state_id,
                            asset_id=None,
                            quality_score=0.0,
                        )
                    )
        grouped[state_id] = rows
    return state_ids, grouped


def load_character_visual_bank(visual_root: Path) -> CharacterVisualBank | None:
    """Resolve one character's ``visual/`` folder into a ``CharacterVisualBank``.

    Returns ``None`` if the directory does not look like a character visual
    root (missing ``character.json`` or ``appearance_states.json``)."""

    character_payload = _load_json(visual_root / "character.json")
    appearance_payload = _load_json(visual_root / "appearance_states.json")
    if not character_payload and not appearance_payload:
        return None

    character_id = (
        character_payload.get("character_id")
        or (visual_root.parent.parent.name.upper() if visual_root.parent else None)
        or visual_root.parent.name
        or "CHR_UNKNOWN"
    )

    state_ids, grouped = _assets_from_appearance_states(
        appearance_payload,
        character_payload,
        visual_root,
    )

    # Fallback: no appearance_states.json (older banks); synthesize one state
    # from the whole gallery so we still produce references.
    if not state_ids:
        fallback_id = f"{character_id}_APP001"
        state_ids = [fallback_id]
        body_rows = _assets_from_gallery(
            character_payload.get("body_gallery") or {},
            visual_root,
            kind="body",
        )
        face_rows = _assets_from_gallery(
            character_payload.get("face_gallery") or {},
            visual_root,
            kind="face",
        )
        for row in body_rows + face_rows:
            row.__dict__["appearance_state_id"] = fallback_id  # type: ignore[misc]
        grouped[fallback_id] = body_rows + face_rows

    rep_dir = visual_root / "representative"
    rep_face = rep_dir / "primary_face.jpg" if (rep_dir / "primary_face.jpg").is_file() else None
    rep_appearance = (
        rep_dir / "primary_appearance.jpg"
        if (rep_dir / "primary_appearance.jpg").is_file()
        else None
    )

    return CharacterVisualBank(
        character_id=str(character_id),
        semantic_scope_id=str(character_payload.get("semantic_scope_id") or visual_root.parent.parent.name),
        subject_id=str(character_payload.get("subject_id") or visual_root.parent.name),
        visual_root=visual_root,
        representative_face=rep_face,
        representative_appearance=rep_appearance,
        appearance_state_ids=state_ids,
        assets_by_state=grouped,
    )


# ---------------------------------------------------------------------------
# Asset selection
# ---------------------------------------------------------------------------

# Order matters: we put the most informative views first to anchor identity,
# then add complementary evidence.
_VIEW_PRIORITY = {
    "front": 0,
    "side": 1,
    "back": 2,
    "profile": 1,
    "other": 3,
    "unknown": 4,
}


def _sort_key(asset: CharacterAsset) -> tuple[int, float]:
    return (_VIEW_PRIORITY.get(asset.view, 5), -asset.quality_score)


def select_three_view_inputs(
    bank: CharacterVisualBank,
    state_id: str,
    *,
    max_images: int = 5,
) -> list[Path]:
    """Pick the strongest references for a character turnaround.

    Strategy: include the representative face (or the highest-quality face
    crop) as the identity anchor, then add up to ``max_images - 1`` body
    crops sorted by quality_score and view diversity. We always prefer at
    least one body crop whose view is not 'unknown' so the model has a
    clothing / posture cue."""

    assets = list(bank.assets_by_state.get(state_id, []))
    faces = sorted(
        (a for a in assets if a.kind == "face"),
        key=_sort_key,
    )
    bodies = sorted(
        (a for a in assets if a.kind == "body"),
        key=_sort_key,
    )

    chosen: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            return
        chosen.append(resolved)
        seen.add(resolved)

    if bank.representative_face and bank.representative_face.is_file():
        _add(bank.representative_face)
    elif faces:
        _add(faces[0].path)

    for asset in bodies:
        if len(chosen) >= max_images:
            break
        _add(asset.path)

    # Final fallback: if we still have <2 inputs, pull representative_appearance.
    if len(chosen) < 2 and bank.representative_appearance:
        _add(bank.representative_appearance)

    return chosen[:max_images]


def select_costume_inputs(
    bank: CharacterVisualBank,
    state_id: str,
    *,
    max_images: int = 4,
) -> list[Path]:
    """Pick the strongest references for an isolated costume plate.

    Strategy: prefer body crops that show as much garment as possible and
    exclude face crops so the model does not redraw the character."""

    assets = [
        a
        for a in bank.assets_by_state.get(state_id, [])
        if a.kind == "body"
    ]
    bodies = sorted(assets, key=_sort_key)
    chosen: list[Path] = []
    seen: set[Path] = set()
    for asset in bodies:
        if len(chosen) >= max_images:
            break
        resolved = asset.path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        chosen.append(resolved)
        seen.add(resolved)
    if not chosen and bank.representative_appearance and bank.representative_appearance.is_file():
        chosen.append(bank.representative_appearance.resolve())
    return chosen


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterRestorationPlan:
    """Planned inputs / outputs for one (character, appearance) pair."""

    bank: CharacterVisualBank
    state_id: str
    three_view_inputs: list[Path]
    costume_inputs: list[Path]
    three_view_output: Path
    costume_output: Path
    three_view_metadata: Path
    costume_metadata: Path


def _resolve_output_root(output_root: Path, character_id: str) -> Path:
    return output_root / character_id


def plan_character_restorations(
    bank: CharacterVisualBank,
    *,
    output_root: Path,
    max_three_view_inputs: int = 5,
    max_costume_inputs: int = 4,
) -> list[CharacterRestorationPlan]:
    """Build one plan per appearance state for a single character."""

    char_dir = _resolve_output_root(output_root, bank.character_id)
    plans: list[CharacterRestorationPlan] = []
    for state_id in bank.appearance_state_ids:
        three_view_inputs = select_three_view_inputs(
            bank,
            state_id,
            max_images=max_three_view_inputs,
        )
        costume_inputs = select_costume_inputs(
            bank,
            state_id,
            max_images=max_costume_inputs,
        )
        if not three_view_inputs or not costume_inputs:
            # Skip empty states silently — they cannot produce anything useful.
            continue
        plans.append(
            CharacterRestorationPlan(
                bank=bank,
                state_id=state_id,
                three_view_inputs=three_view_inputs,
                costume_inputs=costume_inputs,
                three_view_output=(char_dir / state_id / f"{state_id}_three_view.png").resolve(),
                costume_output=(char_dir / state_id / f"{state_id}_costume.png").resolve(),
                three_view_metadata=(char_dir / state_id / f"{state_id}_three_view.png.json").resolve(),
                costume_metadata=(char_dir / state_id / f"{state_id}_costume.png.json").resolve(),
            )
        )
    return plans


def _summarise_inputs(image_paths: Sequence[Path]) -> list[dict]:
    summary: list[dict] = []
    for index, path in enumerate(image_paths, start=1):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        summary.append(
            {
                "index": index,
                "path": str(path),
                "size_bytes": size_bytes,
            }
        )
    return summary


def restore_character_three_view(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[Path],
    output_path: Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options,
) -> GeminiRestorationResult:
    """Restore a three-view character turnaround sheet."""

    generation_options.setdefault("aspect_ratio", "3:4")
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=(
            "Compose a clean three-quarter / side / back / front turnaround of this "
            "single character in this single costume. Same identity, same wardrobe "
            "in every pose. Plain neutral background, even studio lighting."
        ),
        output_path=output_path,
        task_instructions=THREE_VIEW_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        generation_options=generation_options,
    )


def restore_character_costume(
    client: UniAPIGeminiImageClient,
    *,
    image_paths: Sequence[Path],
    output_path: Path,
    asset_description: str | None = None,
    overwrite: bool = False,
    **generation_options,
) -> GeminiRestorationResult:
    """Restore an isolated costume reference plate."""

    generation_options.setdefault("aspect_ratio", "1:1")
    return restore_gemini_reference(
        client,
        image_paths=image_paths,
        user_prompt=(
            "Lay out this complete costume neatly on a plain neutral background, "
            "as if presented to a costume department. Top centered, shirt "
            "underneath visible, trousers / skirt folded beneath, accessories in "
            "a tidy row, shoes at the bottom. Preserve exact colors, fabrics, "
            "patterns, and condition."
        ),
        output_path=output_path,
        task_instructions=COSTUME_INSTRUCTIONS,
        asset_description=asset_description,
        overwrite=overwrite,
        generation_options=generation_options,
    )


def _write_metadata(
    plan: CharacterRestorationPlan,
    result: GeminiRestorationResult,
    metadata_path: Path,
    *,
    task: str,
    overwrite: bool,
) -> None:
    metadata = {
        "schema_version": "character-reference-restore-v1",
        "request": {
            "task": task,
            "character_id": plan.bank.character_id,
            "appearance_state_id": plan.state_id,
            "semantic_scope_id": plan.bank.semantic_scope_id,
            "subject_id": plan.bank.subject_id,
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
            "model_version": result.response.get("modelVersion"),
            "usage": result.response.get("usageMetadata"),
            "text": result.response_text,
            "finish_reasons": [
                candidate.get("finishReason")
                for candidate in result.response.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get("finishReason")
            ],
        },
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
    plan: CharacterRestorationPlan,
    *,
    client: UniAPIGeminiImageClient,
    overwrite: bool = False,
    model: str = DEFAULT_GEMINI_IMAGE_MODEL,
    image_size: str | None = "auto",
    temperature: float | None = None,
) -> tuple[GeminiRestorationResult, GeminiRestorationResult]:
    """Run both restorations for one plan. Returns (three_view_result, costume_result)."""

    common_options: dict = {"model": model}
    if image_size is not None:
        common_options["image_size"] = image_size
    if temperature is not None:
        common_options["temperature"] = temperature

    asset_description = (
        f"Character {plan.bank.character_id}, appearance state {plan.state_id}. "
        f"Reconciled from {len(plan.three_view_inputs)} identity / body crops and "
        f"{len(plan.costume_inputs)} costume crops."
    )

    three_view_result = restore_character_three_view(
        client,
        image_paths=plan.three_view_inputs,
        output_path=plan.three_view_output,
        asset_description=asset_description,
        overwrite=overwrite,
        **common_options,
    )
    _write_metadata(
        plan,
        three_view_result,
        plan.three_view_metadata,
        task="character_three_view",
        overwrite=overwrite,
    )
    costume_result = restore_character_costume(
        client,
        image_paths=plan.costume_inputs,
        output_path=plan.costume_output,
        asset_description=asset_description,
        overwrite=overwrite,
        **common_options,
    )
    _write_metadata(
        plan,
        costume_result,
        plan.costume_metadata,
        task="character_costume",
        overwrite=overwrite,
    )
    return three_view_result, costume_result


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_character_banks(
    character_root: Path,
    *,
    character_filter: Iterable[str] | None = None,
) -> list[CharacterVisualBank]:
    """Walk ``character_root/characters/<scope>/<subject>/visual`` and load banks.

    ``character_root`` is expected to point at the directory that contains
    ``<scope>/<subject>/visual`` siblings, i.e. one level above ``visual/``."""

    wanted: set[str] | None = (
        {c.upper() for c in character_filter} if character_filter else None
    )
    banks: list[CharacterVisualBank] = []
    if not character_root.is_dir():
        return banks
    for visual_dir in sorted(character_root.glob("*/*/visual")):
        bank = load_character_visual_bank(visual_dir)
        if bank is None:
            continue
        if wanted is not None and bank.character_id.upper() not in wanted:
            continue
        banks.append(bank)
    return banks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore per-character three-view and costume reference sheets from the "
            "character visual bank using Gemini multi-image generation."
        )
    )
    parser.add_argument(
        "--character-root",
        type=Path,
        required=True,
        help=(
            "Directory that contains <semantic_scope>/<subject_id>/visual subtrees "
            "(e.g. workspace/.../characters/characters)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where to write reference/<CHR>/<APP>/<APP>_three_view.png and <APP>_costume.png.",
    )
    parser.add_argument(
        "--character",
        action="append",
        default=[],
        help="Restrict to one or more character IDs (e.g. CHR002). Repeatable.",
    )
    parser.add_argument(
        "--max-three-view-inputs",
        type=int,
        default=5,
        help="Maximum reference images to send for the three-view task (default 5).",
    )
    parser.add_argument(
        "--max-costume-inputs",
        type=int,
        default=4,
        help="Maximum reference images to send for the costume task (default 4).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("UNIAPI_GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL),
        choices=list(GEMINI_IMAGE_MODEL_PROFILES),
    )
    parser.add_argument(
        "--image-size",
        default="auto",
        choices=["auto", "1K", "2K", "4K"],
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned inputs/prompts and exit without calling the API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="UNIAPI_API_KEY",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    character_filter = args.character or None
    banks = discover_character_banks(
        args.character_root,
        character_filter=character_filter,
    )
    if not banks:
        print(
            f"error: no character visual banks found under {args.character_root}",
            file=sys.stderr,
        )
        return 2

    all_plans: list[CharacterRestorationPlan] = []
    for bank in banks:
        plans = plan_character_restorations(
            bank,
            output_root=args.output_root,
            max_three_view_inputs=args.max_three_view_inputs,
            max_costume_inputs=args.max_costume_inputs,
        )
        all_plans.extend(plans)

    if not all_plans:
        print(
            "error: character banks were found but none have enough inputs to "
            "restore",
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "character_root": str(args.character_root),
                "output_root": str(args.output_root),
                "character_count": len(banks),
                "plan_count": len(all_plans),
                "plans": [
                    {
                        "character_id": plan.bank.character_id,
                        "appearance_state_id": plan.state_id,
                        "three_view_inputs": _summarise_inputs(plan.three_view_inputs),
                        "costume_inputs": _summarise_inputs(plan.costume_inputs),
                        "three_view_output": str(plan.three_view_output),
                        "costume_output": str(plan.costume_output),
                    }
                    for plan in all_plans
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

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

    failures = 0
    for plan in all_plans:
        try:
            three_view_result, costume_result = execute_plan(
                plan,
                client=client,
                overwrite=args.overwrite,
                model=args.model,
                image_size=args.image_size,
                temperature=args.temperature,
            )
            print(
                json.dumps(
                    {
                        "character_id": plan.bank.character_id,
                        "appearance_state_id": plan.state_id,
                        "three_view": {
                            "path": str(three_view_result.saved_images[0].path),
                            "sha256": three_view_result.saved_images[0].sha256,
                        },
                        "costume": {
                            "path": str(costume_result.saved_images[0].path),
                            "sha256": costume_result.saved_images[0].sha256,
                        },
                    },
                    ensure_ascii=False,
                )
            )
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
            failures += 1
            print(
                f"error: failed to restore {plan.bank.character_id}/{plan.state_id}: {exc}",
                file=sys.stderr,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
