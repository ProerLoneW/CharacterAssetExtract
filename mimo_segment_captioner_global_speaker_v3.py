#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-A segment captioner using read-only global speaker attributions.

MiMo still performs the important segment semantic/object consolidation, but
speaker identity is decided only by the preceding global-caption call.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import mimo_segment_captioner as v1
import mimo_segment_captioner_v2 as v2
from mimo_caption_client import load_dotenv

LOGGER = logging.getLogger("mimo_segment_captioner_global_speaker_v3")
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "segment_caption_prompt_en_objects_global_speakers_v5.md"
SCHEMA_VERSION = "segment-caption-global-speakers-v3"

V2_BUILD_INPUT_TEXT = v2.build_input_text
V2_VALIDATE_AND_ENRICH = v2.validate_and_enrich
V1_ANNOTATE_ONE = v1.annotate_one


def global_speaker_attributions(
    global_record: dict[str, Any],
    _unused_index: dict[str, list[dict[str, Any]]],
    _unused_timebase: str = "auto",
) -> list[dict[str, Any]]:
    payload = v1.unwrap_nested_caption(global_record, "global_caption")
    rows = payload.get("asr_speaker_attributions")
    if not isinstance(rows, list):
        raise ValueError(
            f"global {global_record.get('global_id')} has no asr_speaker_attributions; "
            "run mimo_global_captioner_speaker_v2.py first"
        )
    result = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"global speaker attribution {index} is not an object")
        asr_id = str(row.get("asr_id") or "").strip()
        if not asr_id or asr_id in seen:
            raise ValueError(f"invalid or duplicate global ASR ID: {asr_id!r}")
        if not isinstance(row.get("time_range_sec"), list) or not isinstance(
            row.get("source_time_range_sec"), list
        ):
            raise ValueError(f"global attribution {asr_id} lacks authoritative time ranges")
        seen.add(asr_id)
        result.append(copy.deepcopy(row))
    return sorted(result, key=lambda row: (row["time_range_sec"][0], row["asr_id"]))


def _stripped_global_record(global_record: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(global_record)
    payload = result.get("global_caption")
    if isinstance(payload, dict):
        payload.pop("asr_speaker_attributions", None)
        payload.pop("authoritative_asr_timeline", None)
        nested = payload.get("global_caption")
        if isinstance(nested, dict):
            nested.pop("asr_speaker_attributions", None)
            nested.pop("authoritative_asr_timeline", None)
    return result


def build_input_text(
    prompt: str,
    global_record: dict[str, Any],
    shots: list[dict[str, Any]],
    duration: float,
    speaker_rows: Sequence[dict[str, Any]] = (),
) -> str:
    base = V2_BUILD_INPUT_TEXT(prompt, _stripped_global_record(global_record), shots, duration, ())
    return (
        base
        + "\n\nREAD-ONLY GLOBAL SPEAKER-ATTRIBUTED ASR\n"
        + json.dumps(list(speaker_rows), ensure_ascii=False, indent=2)
        + "\nThese rows are authoritative. Return dialogue: [] in every segment; "
          "local code will attach them after validation.\n"
    )


def validate_and_enrich(
    payload: dict[str, Any],
    shots: list[dict[str, Any]],
    global_record: dict[str, Any],
    duration: float,
    speaker_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Ignore model dialogue and attach global rows deterministically by time."""
    clean_payload = copy.deepcopy(payload)
    ignored_model_dialogue_count = 0
    ignored_unsourced_object_count = 0
    valid_candidate_refs = {
        str(row["candidate_ref"])
        for row in v2.object_candidate_inventory(shots)
    }
    for segment in clean_payload.get("segments") or []:
        dialogue = segment.get("dialogue")
        if isinstance(dialogue, list):
            ignored_model_dialogue_count += len(dialogue)
        segment["dialogue"] = []
        supplied_objects = segment.get("key_object_candidates")
        if isinstance(supplied_objects, list):
            kept_objects = []
            for row in supplied_objects:
                refs = row.get("source_candidate_refs") if isinstance(row, dict) else None
                if (isinstance(refs, list) and refs
                        and all(str(ref) in valid_candidate_refs for ref in refs)):
                    kept_objects.append(row)
                else:
                    ignored_unsourced_object_count += 1
            segment["key_object_candidates"] = kept_objects
        rejected = segment.get("rejected_object_candidate_refs")
        if isinstance(rejected, list):
            segment["rejected_object_candidate_refs"] = [
                ref for ref in rejected if str(ref) in valid_candidate_refs
            ]

    result = V2_VALIDATE_AND_ENRICH(clean_payload, shots, global_record, duration, ())
    for segment in result["segments"]:
        segment["dialogue"] = []

    def target_segment(row: dict[str, Any]) -> dict[str, Any]:
        start, end = v1.number_range(row["time_range_sec"], "global attribution time_range_sec")

        def score(segment: dict[str, Any]) -> tuple[float, float]:
            seg_start, seg_end = v1.number_range(segment["time_range_sec"], "segment time_range_sec")
            overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
            distance = abs((start + end) / 2.0 - (seg_start + seg_end) / 2.0)
            return overlap, -distance

        return max(result["segments"], key=score)

    for source_row in speaker_rows:
        row = copy.deepcopy(source_row)
        target = target_segment(row)
        segment_start = float(target["time_range_sec"][0])
        start, end = v1.number_range(row["time_range_sec"], "global attribution time_range_sec")
        row["segment_time_range_sec"] = [round(start - segment_start, 6), round(end - segment_start, 6)]
        row["speaker_attribution_stage"] = "global_caption"
        target["dialogue"].append(row)
    for segment in result["segments"]:
        segment["dialogue"].sort(key=lambda row: (row["time_range_sec"][0], row["asr_id"]))

    result["authoritative_asr_timeline"] = copy.deepcopy(list(speaker_rows))
    result["schema_version"] = SCHEMA_VERSION
    result["speaker_attribution_stage"] = "global_caption"
    result["ignored_model_dialogue_count"] = ignored_model_dialogue_count
    result["ignored_unsourced_object_count"] = ignored_unsourced_object_count
    return result


async def annotate_one(client, semaphore, global_record, shots, prompt, args, speaker_rows=()):
    """Accept legacy global manifests that omitted the redundant duration field."""
    record = dict(global_record)
    if record.get("duration_sec") is None:
        start, end = v1.number_range(record["time_range_sec"], "global time_range_sec")
        record["duration_sec"] = round(end - start, 6)
    return await V1_ANNOTATE_ONE(
        client, semaphore, record, shots, prompt, args, speaker_rows
    )


def build_parser():
    parser = v2.build_parser()
    parser.set_defaults(prompt_file=DEFAULT_PROMPT)
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.asr_input is not None:
        parser.error(
            "do not pass --asr-input here; Scheme A reads immutable speaker-attributed ASR "
            "from --global-input"
        )
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries, and timeout must be positive")
    if args.max_api_seconds is not None and args.max_api_seconds <= 0:
        parser.error("--max-api-seconds must be positive")
    if not (0 < args.min_segment_seconds <= args.target_segment_seconds <= args.max_segment_seconds):
        parser.error("require 0 < min <= target <= max segment seconds")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")

    v2.MIN_SEGMENT_SECONDS = args.min_segment_seconds
    v2.TARGET_SEGMENT_SECONDS = args.target_segment_seconds
    v2.MAX_SEGMENT_SECONDS = args.max_segment_seconds
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).disabled = True

    # V1's mature retry/persistence orchestration resolves these at runtime.
    v1.build_input_text = build_input_text
    v1.validate_and_enrich = validate_and_enrich
    v1.annotate_one = annotate_one
    v1.prepare_authoritative_asr = global_speaker_attributions
    v1.load_asr_sidecars = lambda _path: {}
    # Truthy sentinel enables V1 cache invalidation against the global rows.
    args.asr_input = args.global_input
    args.asr_timebase = "global-caption"
    return asyncio.run(v1.main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
