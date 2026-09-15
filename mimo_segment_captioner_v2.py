#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 segment captioner with hard duration checks and object consolidation.

The original :mod:`mimo_segment_captioner` remains untouched. This executable
reuses its request/retry/persistence machinery while replacing the prompt,
input boundary guide, and deterministic response validator.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import mimo_segment_captioner as v1
from mimo_caption_client import load_dotenv, resolve_api_key, resolve_base_url, resolve_model


LOGGER = logging.getLogger("mimo_segment_captioner_v2")
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "segment_caption_prompt_en_semantic_roles_objects_v4.md"
SCHEMA_VERSION = "segment-caption-key-objects-v2"

MIN_SEGMENT_SECONDS = 15.0
TARGET_SEGMENT_SECONDS = 20.0
MAX_SEGMENT_SECONDS = 30.0
TIME_EPSILON = 0.25

OBJECT_PRIORITIES = {"must", "high", "optional", "ignore"}
OBJECT_REASON_CODES = {
    "character_interaction", "narrative_prop", "recurring", "state_change",
    "closeup", "distinctive", "text_information",
}
OBJECT_ROLES = {
    "narrative_prop", "interaction_prop", "recurring_prop", "emphasized_prop",
    "information_prop", "uncertain",
}
SPECIAL_SUBJECT_IDS = {"unknown", "new_subject_candidate"}

V1_BUILD_INPUT_TEXT = v1.build_input_text
V1_VALIDATE_AND_ENRICH = v1.validate_and_enrich


def _string_list(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _finite_confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return round(min(max(result, 0.0), 1.0), 6)


def _shot_object_candidates(shot: dict) -> list[dict]:
    caption = v1.unwrap_nested_caption(shot, "shot_caption")
    value = caption.get("key_object_candidates", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def object_candidate_inventory(shots: list[dict]) -> list[dict]:
    inventory: list[dict] = []
    for shot in shots:
        shot_id = int(shot["shot_id"])
        for index, candidate in enumerate(_shot_object_candidates(shot), 1):
            local_id = str(candidate.get("local_object_id") or f"OC{index}")
            inventory.append({
                "candidate_ref": f"SHOT_{shot_id:03d}:{local_id}",
                "shot_id": shot_id,
                "canonical_name": candidate.get("canonical_name"),
                "priority": candidate.get("priority"),
                "grounding_phrases": candidate.get("grounding_phrases", []),
                "reason_codes": candidate.get("reason_codes", []),
                "related_subject_ids": candidate.get("related_subject_ids", []),
                "interactions": candidate.get("interactions", []),
                "visible_time_range_sec": candidate.get("visible_time_range_sec"),
                "confidence": candidate.get("confidence", "unknown"),
            })
    return inventory


def valid_boundary_guide(shots: list[dict]) -> dict:
    """Enumerate timing-valid contiguous ranges for the model."""
    ranges: list[dict] = []
    for left in range(len(shots)):
        start, _ = v1.number_range(shots[left]["time_range_sec"], "shot start")
        for right in range(left, len(shots)):
            _, end = v1.number_range(shots[right]["time_range_sec"], "shot end")
            duration = end - start
            if duration > MAX_SEGMENT_SECONDS + TIME_EPSILON:
                break
            if duration >= MIN_SEGMENT_SECONDS - TIME_EPSILON:
                ranges.append({
                    "shot_ids": [int(item["shot_id"]) for item in shots[left:right + 1]],
                    "duration_sec": round(duration, 3),
                    "distance_from_target_sec": round(abs(duration - TARGET_SEGMENT_SECONDS), 3),
                })
    complete: list[dict] = []

    def visit(position: int, chosen: list[dict]) -> None:
        if position == len(shots):
            durations = [float(item["duration_sec"]) for item in chosen]
            complete.append({
                "shot_id_groups": [item["shot_ids"] for item in chosen],
                "durations_sec": durations,
                "target_distance_sum_sec": round(sum(
                    abs(value - TARGET_SEGMENT_SECONDS) for value in durations), 3),
            })
            return
        start, _ = v1.number_range(shots[position]["time_range_sec"], "partition start")
        for right in range(position, len(shots)):
            _, end = v1.number_range(shots[right]["time_range_sec"], "partition end")
            duration = end - start
            if duration > MAX_SEGMENT_SECONDS + TIME_EPSILON:
                break
            is_final = right == len(shots) - 1
            valid = duration >= MIN_SEGMENT_SECONDS - TIME_EPSILON
            if not valid and is_final and chosen:
                previous_first_id = int(chosen[-1]["shot_ids"][0])
                previous_first = next(
                    item for item in shots if int(item["shot_id"]) == previous_first_id)
                previous_start, _ = v1.number_range(
                    previous_first["time_range_sec"], "previous partition start")
                valid = end - previous_start > MAX_SEGMENT_SECONDS + TIME_EPSILON
            if valid:
                visit(right + 1, chosen + [{
                    "shot_ids": [int(item["shot_id"]) for item in shots[position:right + 1]],
                    "duration_sec": round(duration, 3),
                }])

    visit(0, [])
    complete.sort(key=lambda item: (
        item["target_distance_sum_sec"], len(item["shot_id_groups"]), item["shot_id_groups"]))
    for index, item in enumerate(complete[:40], 1):
        item["partition_id"] = f"P{index:03d}"

    return {
        "min_segment_seconds": MIN_SEGMENT_SECONDS,
        "target_segment_seconds": TARGET_SEGMENT_SECONDS,
        "max_segment_seconds": MAX_SEGMENT_SECONDS,
        "short_final_remainder_rule": (
            "allowed only when merging with previous would exceed max_segment_seconds"
        ),
        "valid_non_short_ranges": ranges,
        "valid_complete_partitions": complete[:40],
        "mandatory_instruction": (
            "Copy shot_id_groups from exactly one valid_complete_partitions entry; "
            "do not invent a different grouping."
        ),
    }


def build_input_text(prompt: str, global_record: dict, shots: list[dict], duration: float,
                     asr_timeline: Sequence[dict[str, Any]] = ()) -> str:
    base = V1_BUILD_INPUT_TEXT(prompt, global_record, shots, duration, asr_timeline)
    guide = valid_boundary_guide(shots)
    inventory = object_candidate_inventory(shots)
    return (
        base
        + "\n\nVALID BOUNDARY GUIDE\n"
        + json.dumps(guide, ensure_ascii=False, indent=2)
        + "\n\nSHOT OBJECT CANDIDATE INVENTORY\n"
        + json.dumps(inventory, ensure_ascii=False, indent=2)
        + "\n"
    )


def _segment_duration(segment: dict) -> float:
    start, end = v1.number_range(segment["time_range_sec"], "segment time_range_sec")
    return end - start


def validate_segment_durations(segments: list[dict]) -> None:
    for index, segment in enumerate(segments):
        duration = _segment_duration(segment)
        if duration > MAX_SEGMENT_SECONDS + TIME_EPSILON:
            raise ValueError(
                f"segment {index + 1} duration {duration:.3f}s exceeds "
                f"{MAX_SEGMENT_SECONDS:.3f}s")
        if duration >= MIN_SEGMENT_SECONDS - TIME_EPSILON:
            continue
        is_final = index == len(segments) - 1
        if not is_final:
            raise ValueError(
                f"non-final segment {index + 1} duration {duration:.3f}s is below "
                f"{MIN_SEGMENT_SECONDS:.3f}s")
        if index == 0:
            continue
        merged_duration = segments[index]["time_range_sec"][1] - segments[index - 1]["time_range_sec"][0]
        if merged_duration <= MAX_SEGMENT_SECONDS + TIME_EPSILON:
            raise ValueError(
                f"short final segment {index + 1} should merge with previous "
                f"({merged_duration:.3f}s <= {MAX_SEGMENT_SECONDS:.3f}s)")


def _candidate_maps(segment: dict) -> tuple[dict[str, dict], dict[str, int]]:
    by_ref: dict[str, dict] = {}
    shot_by_ref: dict[str, int] = {}
    for shot in segment.get("source_shots", []):
        shot_id = int(shot["shot_id"])
        for index, candidate in enumerate(_shot_object_candidates(shot), 1):
            local_id = str(candidate.get("local_object_id") or f"OC{index}")
            ref = f"SHOT_{shot_id:03d}:{local_id}"
            if ref in by_ref:
                raise ValueError(f"duplicate supplied object candidate reference: {ref}")
            by_ref[ref] = candidate
            shot_by_ref[ref] = shot_id
    return by_ref, shot_by_ref


def _fallback_candidate(ref: str, candidate: dict, shot_id: int) -> dict:
    name = str(candidate.get("canonical_name") or "unknown object").strip()
    priority = str(candidate.get("priority") or "optional").lower()
    if priority not in OBJECT_PRIORITIES:
        priority = "optional"
    attributes = candidate.get("visual_attributes")
    return {
        "segment_object_id": "",
        "source_candidate_refs": [ref],
        "canonical_name": name,
        "grounding_phrases": _string_list(candidate.get("grounding_phrases"), limit=4) or [name],
        "priority": priority,
        "reason_codes": [
            item for item in _string_list(candidate.get("reason_codes"))
            if item in OBJECT_REASON_CODES
        ],
        "visual_attributes": attributes if isinstance(attributes, dict) else {},
        "related_subject_ids": _string_list(candidate.get("related_subject_ids")),
        "interactions": copy.deepcopy(candidate.get("interactions", []))
        if isinstance(candidate.get("interactions"), list) else [],
        "occurrence_shot_ids": [shot_id],
        "possible_states": _string_list(candidate.get("possible_states"), limit=8),
        "states_by_shot": ([{
            "shot_id": shot_id,
            "state": str(candidate.get("observed_state") or "unknown"),
        }] if candidate.get("observed_state") else []),
        "recurring_within_segment": False,
        "stateful": bool(candidate.get("possible_states")),
        "segment_role": "uncertain",
        "confidence": 0.0,
        "proposal_status": "shot_only_unconsolidated",
    }


def normalize_segment_objects(segment: dict, registered_ids: set[str]) -> list[str]:
    supplied, shot_by_ref = _candidate_maps(segment)
    raw_rows = segment.get("key_object_candidates", [])
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, list):
        raise ValueError("segment key_object_candidates must be a list")
    rejected = _string_list(segment.get("rejected_object_candidate_refs"))
    unknown_rejected = [ref for ref in rejected if ref not in supplied]
    if unknown_rejected:
        raise ValueError(f"unknown rejected object candidate refs: {unknown_rejected}")
    rejected_set = set(rejected)
    used: set[str] = set()
    normalized: list[dict] = []
    issues: list[str] = []

    for row_index, raw in enumerate(raw_rows, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"key_object_candidates[{row_index}] must be an object")
        refs = _string_list(raw.get("source_candidate_refs"))
        if not refs:
            raise ValueError(f"key_object_candidates[{row_index}] has no source_candidate_refs")
        invalid_refs = [ref for ref in refs if ref not in supplied]
        if invalid_refs:
            raise ValueError(f"unknown source object candidate refs: {invalid_refs}")
        duplicate_refs = [ref for ref in refs if ref in used]
        if duplicate_refs:
            raise ValueError(f"object candidate refs used more than once: {duplicate_refs}")
        if rejected_set.intersection(refs):
            raise ValueError(f"object candidate refs both accepted and rejected: {refs}")
        used.update(refs)
        seed = supplied[refs[0]]
        name = str(raw.get("canonical_name") or seed.get("canonical_name") or "").strip()
        if not name:
            raise ValueError(f"key_object_candidates[{row_index}] missing canonical_name")
        priority = str(raw.get("priority") or seed.get("priority") or "optional").lower()
        if priority not in OBJECT_PRIORITIES:
            raise ValueError(f"invalid segment object priority: {priority}")
        subjects = _string_list(raw.get("related_subject_ids"))
        invalid_subjects = [
            subject for subject in subjects
            if subject not in registered_ids and subject not in SPECIAL_SUBJECT_IDS
        ]
        if invalid_subjects:
            raise ValueError(f"unregistered segment object subject IDs: {invalid_subjects}")
        occurrence_shots = sorted({shot_by_ref[ref] for ref in refs})
        role = str(raw.get("segment_role") or "uncertain")
        if role not in OBJECT_ROLES:
            role = "uncertain"
        attributes = raw.get("visual_attributes", seed.get("visual_attributes", {}))
        possible_states = _string_list(
            raw.get("possible_states", seed.get("possible_states", [])), limit=8)
        normalized.append({
            "segment_object_id": "",
            "source_candidate_refs": refs,
            "canonical_name": name,
            "grounding_phrases": _string_list(raw.get("grounding_phrases"), limit=4)
            or _string_list(seed.get("grounding_phrases"), limit=4) or [name],
            "priority": priority,
            "reason_codes": [
                item for item in _string_list(raw.get("reason_codes", seed.get("reason_codes", [])))
                if item in OBJECT_REASON_CODES
            ],
            "visual_attributes": attributes if isinstance(attributes, dict) else {},
            "related_subject_ids": subjects,
            "interactions": copy.deepcopy(raw.get("interactions", []))
            if isinstance(raw.get("interactions"), list) else [],
            "occurrence_shot_ids": occurrence_shots,
            "possible_states": possible_states,
            "states_by_shot": copy.deepcopy(raw.get("states_by_shot", []))
            if isinstance(raw.get("states_by_shot"), list) else [],
            "recurring_within_segment": len(occurrence_shots) > 1,
            "stateful": bool(raw.get("stateful", possible_states)),
            "segment_role": role,
            "confidence": _finite_confidence(raw.get("confidence")),
            "proposal_status": "unverified",
        })

    missing = set(supplied) - used - rejected_set
    for ref in supplied:
        if ref not in missing:
            continue
        normalized.append(_fallback_candidate(ref, supplied[ref], shot_by_ref[ref]))
        issues.append(f"shot_candidate_preserved_without_segment_review:{ref}")

    segment_id = int(segment["segment_id"])
    for index, row in enumerate(normalized, 1):
        row["segment_object_id"] = f"SEG_{segment_id:03d}_OC{index:03d}"
    segment["key_object_candidates"] = normalized
    segment["rejected_object_candidate_refs"] = rejected
    continuity = segment.get("object_continuity_to_next", [])
    segment["object_continuity_to_next"] = continuity if isinstance(continuity, list) else []
    return issues


def validate_and_enrich(payload: dict, shots: list[dict], global_record: dict, duration: float,
                        asr_timeline: Sequence[dict[str, Any]] = ()) -> dict:
    result = V1_VALIDATE_AND_ENRICH(payload, shots, global_record, duration, asr_timeline)
    validate_segment_durations(result["segments"])
    global_payload = v1.unwrap_nested_caption(global_record, "global_caption")
    registered_ids = {
        str(item.get("subject_id"))
        for item in global_payload.get("subject_registry", [])
        if isinstance(item, dict) and item.get("subject_id")
    }
    issues: list[str] = []
    candidate_index: dict[str, str] = {}
    for segment in result["segments"]:
        issues.extend(normalize_segment_objects(segment, registered_ids))
        for row in segment["key_object_candidates"]:
            for ref in row["source_candidate_refs"]:
                candidate_index[ref] = row["segment_object_id"]
    result["schema_version"] = SCHEMA_VERSION
    result["object_candidate_index"] = candidate_index
    result["object_quality_issues"] = issues
    result["duration_contract"] = {
        "min_segment_seconds": MIN_SEGMENT_SECONDS,
        "target_segment_seconds": TARGET_SEGMENT_SECONDS,
        "max_segment_seconds": MAX_SEGMENT_SECONDS,
        "epsilon_seconds": TIME_EPSILON,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create duration-safe segment captions with key-object consolidation (V2).")
    parser.add_argument("--global-input", required=True, type=Path)
    parser.add_argument("--shot-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录")
    parser.add_argument("--asr-input", type=Path,
                        help="ASR JSON/JSONL file or directory injected into MiMo")
    parser.add_argument("--asr-timebase", choices=("auto", "source", "global-local"), default="auto")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=("video_url", "input_video", "auto"),
                        default="video_url")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-segment-seconds", type=float, default=15.0)
    parser.add_argument("--target-segment-seconds", type=float, default=20.0)
    parser.add_argument("--max-segment-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    global MIN_SEGMENT_SECONDS, TARGET_SEGMENT_SECONDS, MAX_SEGMENT_SECONDS
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries, and timeout must be positive")
    if args.max_api_seconds is not None and args.max_api_seconds <= 0:
        parser.error("--max-api-seconds must be positive")
    if not (0 < args.min_segment_seconds <= args.target_segment_seconds <= args.max_segment_seconds):
        parser.error("require 0 < min <= target <= max segment seconds")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    MIN_SEGMENT_SECONDS = args.min_segment_seconds
    TARGET_SEGMENT_SECONDS = args.target_segment_seconds
    MAX_SEGMENT_SECONDS = args.max_segment_seconds
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).disabled = True

    # V1 orchestration resolves these callbacks at runtime. Keep all old source
    # files unchanged while this executable owns the V2 contract.
    v1.build_input_text = build_input_text
    v1.validate_and_enrich = validate_and_enrich
    return asyncio.run(v1.main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
