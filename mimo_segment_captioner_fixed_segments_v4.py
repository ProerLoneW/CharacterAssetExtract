#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-A V4: caption immutable upstream segments without MiMo planning.

The upstream pipeline already owns shot splitting and shot-to-segment/global
aggregation.  This executable sends those fixed segments to MiMo for caption
content only, then reconstructs the established downstream segment schema
locally.  Speaker attribution remains read-only from global captioning.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import mimo_segment_captioner as v1
import mimo_segment_captioner_global_speaker_v3 as speaker_v3
import mimo_segment_captioner_v2 as object_v2
from mimo_caption_client import (
    load_dotenv,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)

LOGGER = logging.getLogger("mimo_segment_captioner_fixed_segments_v4")
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "segment_caption_prompt_en_fixed_segments_global_speakers_v6.md"
SCHEMA_VERSION = "segment-caption-fixed-upstream-v4"

UPSTREAM_SEGMENTS: list[dict[str, Any]] = []
V1_ANNOTATE_ONE = v1.annotate_one
V1_VALIDATE_AND_ENRICH = v1.validate_and_enrich


def load_fixed_segments(path: Path) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    if source.is_dir():
        candidates = sorted(source.rglob("segments.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No segments.jsonl found under {source}")
        if len(candidates) > 1:
            raise ValueError(f"Multiple segments.jsonl files under {source}; pass one file explicitly")
        source = candidates[0]
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("segment_id"):
            raise ValueError(f"Missing segment_id at {source}:{line_no}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Fixed segment manifest is empty: {source}")
    return rows


def _range_from_row(row: dict[str, Any], global_start: float, global_end: float) -> tuple[float, float]:
    if isinstance(row.get("source_time_range_sec"), (list, tuple)):
        return v1.number_range(row["source_time_range_sec"], "segment source_time_range_sec")
    if row.get("start_time") is not None and row.get("end_time") is not None:
        return v1.number_range([row["start_time"], row["end_time"]], "segment start/end")
    value = row.get("time_range_sec")
    start, end = v1.number_range(value, "segment time_range_sec")
    duration = global_end - global_start
    looks_local = start >= -0.25 and end <= duration + 0.25
    looks_source = start >= global_start - 0.25 and end <= global_end + 0.25
    if looks_local and not looks_source:
        return start + global_start, end + global_start
    return start, end


def _numeric_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def fixed_plan_for_global(global_record: dict[str, Any]) -> list[dict[str, Any]]:
    global_start, global_end = v1.number_range(global_record["time_range_sec"], "global time_range_sec")
    expected = [int(value) for value in global_record.get("shot_ids", [])]
    expected_set = set(expected)
    selected = []
    for row in UPSTREAM_SEGMENTS:
        source_start, source_end = _range_from_row(row, global_start, global_end)
        if min(source_end, global_end) <= max(source_start, global_start):
            continue
        raw_ids = row.get("shot_ids")
        if not isinstance(raw_ids, list):
            raw_ids = row.get("clip_ids", [])
        shot_ids = [value for item in raw_ids if (value := _numeric_id(item)) in expected_set]
        if not shot_ids:
            raise ValueError(
                f"fixed segment {row['segment_id']} overlaps global {global_record.get('global_id')} "
                "but has no matching shot_ids/clip_ids"
            )
        selected.append({
            "upstream_segment_id": str(row["segment_id"]),
            "shot_ids": shot_ids,
            "time_range_sec": [
                round(max(source_start, global_start) - global_start, 6),
                round(min(source_end, global_end) - global_start, 6),
            ],
            "source_time_range_sec": [
                round(max(source_start, global_start), 6),
                round(min(source_end, global_end), 6),
            ],
        })
    selected.sort(key=lambda row: (row["source_time_range_sec"][0], row["upstream_segment_id"]))
    actual = [shot_id for row in selected for shot_id in row["shot_ids"]]
    if actual != expected:
        raise ValueError(
            f"fixed segments must cover global shots exactly once: expected={expected}, actual={actual}"
        )
    ids = [row["upstream_segment_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate upstream segment IDs in global {global_record.get('global_id')}")
    return selected


def prepare_caption_context(global_record, unused_index, unused_timebase="fixed-upstream"):
    return {
        "fixed_segment_plan": fixed_plan_for_global(global_record),
        "speaker_attributed_asr": speaker_v3.global_speaker_attributions(
            global_record, unused_index, unused_timebase
        ),
    }


def build_input_text(
    prompt: str,
    global_record: dict[str, Any],
    shots: list[dict[str, Any]],
    duration: float,
    caption_context: dict[str, Any],
) -> str:
    global_payload = v1.unwrap_nested_caption(global_record, "global_caption")
    global_payload.pop("asr_speaker_attributions", None)
    global_payload.pop("authoritative_asr_timeline", None)
    shot_map = {int(shot["shot_id"]): shot for shot in shots}
    fixed_segments = []
    for plan in caption_context["fixed_segment_plan"]:
        fixed_segments.append({
            **copy.deepcopy(plan),
            "shots": [{
                "shot_id": shot_id,
                "time_range_sec": shot_map[shot_id]["time_range_sec"],
                "source_time_range_sec": shot_map[shot_id]["source_time_range_sec"],
                "shot_caption": shot_map[shot_id]["shot_caption"],
            } for shot_id in plan["shot_ids"]],
        })
    return "\n".join([
        prompt.rstrip(),
        "",
        "INPUT DATA",
        "[Global Caption and Person Registry]",
        json.dumps({
            "global_id": global_record.get("global_id"),
            "video_duration_sec": round(duration, 6),
            "source_time_range_sec": global_record.get("time_range_sec"),
            "global_caption": global_payload,
        }, ensure_ascii=False, indent=2),
        "",
        "[FIXED UPSTREAM SEGMENTS — IMMUTABLE]",
        json.dumps(fixed_segments, ensure_ascii=False, indent=2),
        "",
        "[SHOT OBJECT CANDIDATE INVENTORY]",
        json.dumps(object_v2.object_candidate_inventory(shots), ensure_ascii=False, indent=2),
        "",
        "[READ-ONLY GLOBAL SPEAKER-ATTRIBUTED ASR]",
        json.dumps(caption_context["speaker_attributed_asr"], ensure_ascii=False, indent=2),
        "",
        "Caption exactly the supplied upstream_segment_ids. Do not plan segments.",
    ])


def _valid_model_objects(row: dict[str, Any], valid_refs: set[str]) -> tuple[list[dict], int]:
    output = []
    ignored = 0
    supplied = row.get("key_object_candidates")
    for candidate in supplied if isinstance(supplied, list) else []:
        refs = candidate.get("source_candidate_refs") if isinstance(candidate, dict) else None
        if isinstance(refs, list) and refs and all(str(ref) in valid_refs for ref in refs):
            output.append(candidate)
        else:
            ignored += 1
    return output, ignored


def validate_and_enrich(
    payload: dict[str, Any],
    shots: list[dict[str, Any]],
    global_record: dict[str, Any],
    duration: float,
    caption_context: dict[str, Any],
) -> dict[str, Any]:
    model_rows = payload.get("segment_captions")
    if not isinstance(model_rows, list):
        raise ValueError("response must contain segment_captions")
    plan = caption_context["fixed_segment_plan"]
    expected_ids = [row["upstream_segment_id"] for row in plan]
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(model_rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"segment_captions[{index}] must be an object")
        segment_id = str(row.get("upstream_segment_id") or "")
        if not segment_id or segment_id in by_id:
            raise ValueError(f"invalid or duplicate upstream_segment_id: {segment_id!r}")
        by_id[segment_id] = row
    if list(by_id) != expected_ids:
        raise ValueError(
            f"fixed segment ID/order mismatch: expected={expected_ids}, actual={list(by_id)}"
        )

    inventory_refs = {
        str(item["candidate_ref"]) for item in object_v2.object_candidate_inventory(shots)
    }
    ignored_dialogue = 0
    ignored_objects = 0
    synthesized = []
    for plan_row in plan:
        model = copy.deepcopy(by_id[plan_row["upstream_segment_id"]])
        dialogue = model.get("dialogue")
        ignored_dialogue += len(dialogue) if isinstance(dialogue, list) else 0
        objects, ignored = _valid_model_objects(model, inventory_refs)
        ignored_objects += ignored
        model.pop("upstream_segment_id", None)
        model.pop("segment_id", None)
        model.pop("time_range_sec", None)
        model.pop("source_time_range_sec", None)
        model.pop("boundary_reason", None)
        model["shot_ids"] = copy.deepcopy(plan_row["shot_ids"])
        model["dialogue"] = []
        model["key_object_candidates"] = objects
        rejected = model.get("rejected_object_candidate_refs")
        model["rejected_object_candidate_refs"] = [
            ref for ref in rejected if str(ref) in inventory_refs
        ] if isinstance(rejected, list) else []
        synthesized.append(model)

    compatible_payload = {
        "segments": synthesized,
        "semantic_role_summary": copy.deepcopy(
            payload.get("overall_semantic_role_summary", {})
        ),
        "annotation_confidence": payload.get("annotation_confidence", "unknown"),
    }
    result = V1_VALIDATE_AND_ENRICH(
        compatible_payload, shots, global_record, duration, ()
    )
    registered_ids = {
        str(item.get("subject_id"))
        for item in v1.unwrap_nested_caption(global_record, "global_caption").get("subject_registry", [])
        if isinstance(item, dict) and item.get("subject_id")
    }
    issues = []
    candidate_index = {}
    for output_segment, plan_row in zip(result["segments"], plan):
        output_segment["upstream_segment_id"] = plan_row["upstream_segment_id"]
        output_segment["time_range_sec"] = copy.deepcopy(plan_row["time_range_sec"])
        output_segment["source_time_range_sec"] = copy.deepcopy(plan_row["source_time_range_sec"])
        output_segment["dialogue"] = []
        issues.extend(object_v2.normalize_segment_objects(output_segment, registered_ids))
        for candidate in output_segment["key_object_candidates"]:
            for ref in candidate["source_candidate_refs"]:
                candidate_index[ref] = candidate["segment_object_id"]

    for source_row in caption_context["speaker_attributed_asr"]:
        row = copy.deepcopy(source_row)
        start, end = v1.number_range(row["time_range_sec"], "speaker ASR time_range_sec")
        target = max(result["segments"], key=lambda segment: (
            max(0.0, min(end, segment["time_range_sec"][1]) - max(start, segment["time_range_sec"][0])),
            -abs((start + end) / 2.0 - sum(segment["time_range_sec"]) / 2.0),
        ))
        row["segment_time_range_sec"] = [
            round(start - target["time_range_sec"][0], 6),
            round(end - target["time_range_sec"][0], 6),
        ]
        row["speaker_attribution_stage"] = "global_caption"
        target["dialogue"].append(row)
    for segment in result["segments"]:
        segment["dialogue"].sort(key=lambda row: (row["time_range_sec"][0], row["asr_id"]))

    result["schema_version"] = SCHEMA_VERSION
    result["fixed_segment_plan"] = copy.deepcopy(plan)
    result["segment_planning_source"] = "upstream_manifest"
    result["speaker_attribution_stage"] = "global_caption"
    result["authoritative_asr_timeline"] = copy.deepcopy(caption_context)
    result["object_candidate_index"] = candidate_index
    result["object_quality_issues"] = issues
    result["ignored_model_dialogue_count"] = ignored_dialogue
    result["ignored_unsourced_object_count"] = ignored_objects
    return result


async def annotate_one(client, semaphore, global_record, shots, prompt, args, caption_context):
    record = dict(global_record)
    if record.get("duration_sec") is None:
        start, end = v1.number_range(record["time_range_sec"], "global time_range_sec")
        record["duration_sec"] = round(end - start, 6)
    return await V1_ANNOTATE_ONE(
        client, semaphore, record, shots, prompt, args, caption_context
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Caption fixed upstream segments without MiMo segment planning."
    )
    parser.add_argument("--global-input", required=True, type=Path)
    parser.add_argument("--shot-input", required=True, type=Path)
    parser.add_argument("--segment-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=("video_url", "input_video", "auto"), default="video_url")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <global manifest父目录名>/ 子目录")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    global UPSTREAM_SEGMENTS
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries, and timeout must be positive")
    if args.max_api_seconds is not None and args.max_api_seconds <= 0:
        parser.error("--max-api-seconds must be positive")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    UPSTREAM_SEGMENTS = load_fixed_segments(args.segment_input)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).disabled = True

    v1.build_input_text = build_input_text
    v1.validate_and_enrich = validate_and_enrich
    v1.annotate_one = annotate_one
    v1.prepare_authoritative_asr = prepare_caption_context
    v1.load_asr_sidecars = lambda _path: {}
    # V1 compares this deterministic context with cached output. A changed
    # upstream plan or global speaker mapping therefore invalidates the cache.
    args.asr_input = args.segment_input
    args.asr_timebase = "fixed-upstream"
    return asyncio.run(v1.main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
