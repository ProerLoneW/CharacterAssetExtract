#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create 15-30 second segment captions from global and shot captions.

Atomic migration of ``lsj/video_process/scripts/annotate_segments_mimo.py``.
Reads a successful ``global_captions.jsonl`` and ``shot_captions.jsonl``,
sends each global clip together with the global caption context and ordered
shot captions to MiMo, validates that the returned segments are 15-30 seconds,
non-overlapping and exactly cover the supplied shots, then enriches each
segment with shot-relative timestamps and a deduplicated dialogue-tagged
description.

Outputs::

    <output>/<video>/segments/global_<id>.json
    <output>/<video>/raw_responses/global_<id>.txt
    <output>/<video>/segment_captions.jsonl
    <output>/<video>/summary.json

Usage::

    python mimo_segment_captioner.py \
        --global-input ./outputs/global_caption_outputs/<video> \
        --shot-input   ./outputs/shot_caption_outputs/<video> \
        --output-dir   ./outputs/segment_caption_outputs
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Sequence

from mimo_caption_client import (
    build_video_content,
    default_prompt_path,
    encode_video_b64,
    extract_response_text,
    find_global_caption_manifest,
    find_shot_caption_manifest,
    load_dotenv,
    load_jsonl,
    parse_json_response,
    persist_item,
    resolve_api_key,
    resolve_base_url,
    resolve_clip_path,
    resolve_model,
    response_diagnostic,
    unwrap_nested_caption,
    write_jsonl,
)

LOGGER = logging.getLogger("mimo_segment_captioner")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def find_manifest(input_path: Path, name: str) -> Path:
    if input_path.is_file():
        if input_path.name != name:
            raise ValueError(f"Input file must be {name}: {input_path}")
        return input_path
    candidates = sorted(input_path.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"No {name} found under {input_path}")
    if len(candidates) > 1:
        LOGGER.warning("Multiple %s files found; using %s", name, candidates[0])
    return candidates[0]


def number_range(value, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item numeric range")
    start, end = float(value[0]), float(value[1])
    if end < start:
        raise ValueError(f"{field} has end before start")
    return start, end


def prepare_shots(global_record: dict, shot_map: dict[int, dict]) -> tuple[list[dict], float]:
    shot_ids = [int(value) for value in global_record.get("shot_ids", [])]
    if not shot_ids:
        raise ValueError(f"Global clip {global_record.get('global_id')} has no shot_ids")
    global_start, global_end = number_range(global_record["time_range_sec"], "global time_range_sec")
    prepared = []
    for shot_id in shot_ids:
        if shot_id not in shot_map:
            raise ValueError(f"Missing shot caption for shot_id={shot_id}")
        source = copy.deepcopy(shot_map[shot_id])
        source_start, source_end = number_range(source["time_range_sec"], f"shot {shot_id} time_range_sec")
        local = dict(source)
        local["source_time_range_sec"] = [source_start, source_end]
        local["time_range_sec"] = [
            round(max(0.0, source_start - global_start), 6),
            round(max(0.0, source_end - global_start), 6),
        ]
        local["shot_caption"] = unwrap_nested_caption(source, "shot_caption")
        prepared.append(local)
    duration = max(0.0, global_end - global_start)
    return prepared, duration


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("segments", "asr_segments", "asr_timeline", "rows", "turns"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    return []


def _read_asr_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"ASR JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
        return rows
    return _payload_rows(json.loads(path.read_text(encoding="utf-8")))


def _global_scope(value: Any) -> str | None:
    match = re.search(r"global[_-]?(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return None


def load_asr_sidecars(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Load movie-level or per-global ASR JSON/JSONL without changing its timebase."""
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        files = [source]
    else:
        preferred = sorted(set(source.rglob("asr_timeline.json")) | set(source.rglob("asr_segments.jsonl")))
        files = preferred or sorted(
            file for file in source.rglob("*asr*.json*") if file.is_file()
        )
        if not files:
            raise FileNotFoundError(f"No ASR JSON/JSONL found under {source}")
    index: dict[str, list[dict[str, Any]]] = {}
    for file in files:
        scope = _global_scope(file) or ("*" if len(files) == 1 else None)
        if scope is None:
            LOGGER.warning("Skipping ASR file without a global scope: %s", file)
            continue
        index.setdefault(scope, []).extend(_read_asr_file(file))
    return index


def _asr_range(row: dict[str, Any]) -> tuple[float, float, str] | None:
    for key, timebase in (
        ("source_time_range_sec", "source"),
        ("absolute_source_time_range_sec", "source"),
        ("global_time_range_sec", "global-local"),
        ("local_time_range_sec", "global-local"),
        ("asr_time_range_sec", "unspecified"),
        ("time_range_sec", "unspecified"),
    ):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = float(value[0]), float(value[1])
            return (start, end, timebase) if end > start else None
    if row.get("start") is not None and row.get("end") is not None:
        start, end = float(row["start"]), float(row["end"])
        return (start, end, "unspecified") if end > start else None
    return None


def prepare_authoritative_asr(
    global_record: dict,
    index: dict[str, list[dict[str, Any]]],
    requested_timebase: str = "auto",
) -> list[dict[str, Any]]:
    """Select one global's ASR and expose both clip-local and source time ranges."""
    global_start, global_end = number_range(global_record["time_range_sec"], "global time_range_sec")
    duration = max(0.0, global_end - global_start)
    scope = _global_scope(global_record.get("global_id"))
    rows = [*index.get("*", []), *index.get(scope or "", [])]
    unspecified_ranges = [parsed for row in rows if (parsed := _asr_range(row)) is not None
                          and parsed[2] == "unspecified"]
    auto_default = None
    if index.get(scope or "") and not index.get("*"):
        auto_default = "global-local"
    elif any(end > duration + 1.0 for _, end, _ in unspecified_ranges):
        auto_default = "source"
    output = []
    seen_ids = set()
    for index_number, row in enumerate(rows, 1):
        parsed = _asr_range(row)
        if parsed is None:
            continue
        start, end, explicit_timebase = parsed
        timebase = explicit_timebase
        if timebase == "unspecified":
            timebase = requested_timebase
        if timebase == "auto":
            if auto_default:
                timebase = auto_default
            else:
                looks_source = start >= global_start - 1.0 and end <= global_end + 1.0
                looks_local = start >= -1.0 and end <= duration + 1.0
                timebase = "source" if looks_source and not looks_local else "global-local"
        if timebase == "global-local":
            local_start, local_end = start, end
            source_start, source_end = start + global_start, end + global_start
        elif timebase == "source":
            source_start, source_end = start, end
            local_start, local_end = start - global_start, end - global_start
        else:
            raise ValueError(f"Unsupported ASR timebase: {timebase}")
        if min(global_end, source_end) <= max(global_start, source_start):
            continue
        asr_id = str(row.get("asr_id") or f"ASR_{scope or 'GLOBAL'}_{index_number:04d}")
        if asr_id in seen_ids:
            raise ValueError(f"Duplicate ASR ID in global {global_record.get('global_id')}: {asr_id}")
        seen_ids.add(asr_id)
        output.append({
            "asr_id": asr_id,
            "time_range_sec": [round(local_start, 6), round(local_end, 6)],
            "source_time_range_sec": [round(source_start, 6), round(source_end, 6)],
            "text": str(row.get("text") or "").strip(),
            "language": str(row.get("language") or "unknown"),
            "confidence": row.get("confidence"),
        })
    return sorted(output, key=lambda row: (row["time_range_sec"][0], row["time_range_sec"][1], row["asr_id"]))


def build_input_text(prompt: str, global_record: dict, shots: list[dict], duration: float,
                     asr_timeline: Sequence[dict[str, Any]] = ()) -> str:
    global_payload = unwrap_nested_caption(global_record, "global_caption")
    global_context = {
        "global_id": global_record.get("global_id"),
        "video_duration_sec": round(duration, 6),
        "source_time_range_sec": global_record.get("time_range_sec"),
        "global_caption": global_payload,
    }
    parts = [prompt.rstrip(), "", "INPUT DATA",
             "[Global Caption and Event Evidence]",
             json.dumps(global_context, ensure_ascii=False, indent=2),
             "", "[Authoritative ASR Timeline]",
             "time_range_sec is relative to this global clip; source_time_range_sec is absolute in the source video.",
             "Copy every ASR row exactly once into segment dialogue. Do not edit its text or timestamps.",
             json.dumps(list(asr_timeline), ensure_ascii=False, indent=2)]
    for shot in shots:
        shot_id = int(shot["shot_id"])
        parts.extend(
            [
                "",
                f"[Shot {shot_id} | local_time: {shot['time_range_sec'][0]}-{shot['time_range_sec'][1]} sec]",
                json.dumps(
                    {
                        "shot_id": shot_id,
                        "time_range_sec": shot["time_range_sec"],
                        "source_time_range_sec": shot["source_time_range_sec"],
                        "shot_caption": shot["shot_caption"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Description normalisation helpers
# ---------------------------------------------------------------------------
def dialogue_tags(segment: dict, shot_map: dict[int, dict]) -> list[str]:
    tags = []
    seen = set()
    for shot_id in segment["shot_ids"]:
        caption = unwrap_nested_caption(shot_map[int(shot_id)], "shot_caption")
        for item in caption.get("dialogue", []):
            if not isinstance(item, dict) or not item.get("text"):
                continue
            language = str(item.get("language") or "unknown").strip()
            text = str(item["text"]).strip()
            tag = f"<d>[{language}]{text}</d>"
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def description_value(caption: dict, scoped_key: str) -> str:
    """Read the new stage-specific name while accepting legacy outputs."""
    value = caption.get(scoped_key)
    if isinstance(value, str) and value.strip():
        return value
    legacy = caption.get("integrated_multimodal_description")
    return legacy if isinstance(legacy, str) else ""


def add_dialogue_tags(description: str, tags: list[str]) -> str:
    """Ensure machine-readable dialogue spans exist without discarding model text."""
    result = description.strip()
    previous = None
    while previous != result:
        previous = result
        result = re.sub(r"<d>(\[[^\]]+\])<d>(\[[^\]]+\])", r"<d>\1", result)
        result = re.sub(r"</d></d>", "</d>", result)
    for tag in tags:
        if tag in result:
            continue
        text = tag.split("]", 1)[1].removesuffix("</d>") if "]" in tag else ""
        if text and text in result:
            result = result.replace(text, tag, 1)
        else:
            result = f"{result} {tag}".strip()
    return result


def deduplicate_dialogue_tags(description: str) -> str:
    """Keep each exact dialogue/ASR span once in shot-labeled description text."""
    seen = set()

    def keep_first(match: re.Match) -> str:
        tag = match.group(0)
        if tag in seen:
            return ""
        seen.add(tag)
        return tag

    result = re.sub(r"<d>.*?</d>", keep_first, description, flags=re.DOTALL)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\s+\.", ".", result)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r"\n\s+", "\n", result)
    return result.strip()


def remove_scenetrans_markers(description: str) -> str:
    """Strip legacy continuation markers from generated shot-labeled text."""
    result = re.sub(r"\(?\s*<scenetrans>\s*\)?", "", description, flags=re.IGNORECASE)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\s+\.", ".", result)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r"\n\s+", "\n", result)
    return result.strip()


def fallback_shot_labeled_description(segment: dict, shot_map: dict[int, dict]) -> str:
    lines = []
    seen_tags = set()
    for shot_id in segment["shot_ids"]:
        shot = shot_map[int(shot_id)]
        caption = unwrap_nested_caption(shot, "shot_caption")
        description = description_value(caption, "shot_description")
        if not description:
            description = json.dumps(caption, ensure_ascii=False)
        subjects = caption.get("subjects", [])
        labels = []
        for subject in subjects:
            if isinstance(subject, dict) and subject.get("id"):
                labels.append(f"{subject.get('description', 'subject')} ({subject['id']})")
        if labels:
            description = f"{description} Identified subject(s): {'; '.join(labels)}."
        tags = []
        for tag in dialogue_tags({"shot_ids": [shot_id]}, shot_map):
            if tag not in seen_tags:
                seen_tags.add(tag)
                tags.append(tag)
        description = add_dialogue_tags(description, tags)
        lines.append(f"[Shot {int(shot_id)}]: {description}")
    return "\n".join(lines)


def has_all_shot_labels(description: str, shot_ids: list[int]) -> bool:
    return all(f"[Shot {int(shot_id)}]" in description for shot_id in shot_ids)


def rewrite_shot_timestamps(description: str, segment: dict, shot_map: dict[int, dict]) -> str:
    """Rewrite [Shot N] markers to segment-relative [Shot N | SS.xx-SS.xx] form."""
    ids = [int(value) for value in segment.get("shot_ids", [])]
    if not ids:
        return description
    first, _ = number_range(shot_map[ids[0]]["time_range_sec"], "segment start")
    result = description
    for shot_id in ids:
        shot_start, shot_end = number_range(shot_map[shot_id]["time_range_sec"], f"shot {shot_id} range")
        relative_start = max(0.0, shot_start - first)
        relative_end = max(relative_start, shot_end - first)
        pattern = re.compile(
            rf"\[Shot\s+{re.escape(str(shot_id))}\]\s+"
            r"At\s+\d{2}:\d{2}\.\d{3},?\s*"
        )
        result = pattern.sub(
            lambda match: f"[Shot {shot_id} | {relative_start:.2f}-{relative_end:.2f}] ",
            result,
            count=1,
        )
        if f"[Shot {shot_id} |" not in result:
            result = re.sub(
                rf"\[Shot\s+{re.escape(str(shot_id))}\]",
                lambda match: f"[Shot {shot_id} | {relative_start:.2f}-{relative_end:.2f}]",
                result,
                count=1,
            )
    return result


def _attribution_confidence(value: Any) -> str:
    confidence = str(value or "unknown").strip().lower()
    return confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown"


DIRECT_ATTRIBUTION_EVIDENCE_TYPES = {
    "visible_lip_sync",
    "exclusive_audio_visual_cooccurrence",
    "repeated_voice_face_cooccurrence",
}
MAX_AUDIO_ASSET_ASR_SPAN_SECONDS = 10.0
MIN_AUDIO_ASSET_SPEECH_UNITS_PER_SECOND = 0.75


def _attribution_evidence_type(value: Any, evidence: str) -> str:
    """Normalize MiMo evidence and conservatively classify legacy responses."""
    allowed = DIRECT_ATTRIBUTION_EVIDENCE_TYPES | {
        "context_only", "voice_only", "none", "unknown",
    }
    result = str(value or "").strip().lower()
    if result in allowed:
        return result
    text = evidence.lower()
    if re.search(r"(lip|mouth).{0,40}(sync|mov|speak)|synchroni[sz]ed.{0,30}(lip|mouth)", text):
        return "visible_lip_sync"
    if re.search(r"visual (?:of|evidence).{0,20}s\d+.{0,20}speak|s\d+.{0,30}visible.{0,30}speak", text):
        return "visible_lip_sync"
    if "exclusive" in text and ("visible" in text or "on screen" in text):
        return "exclusive_audio_visual_cooccurrence"
    if "repeated" in text and ("voice" in text or "co-occurrence" in text):
        return "repeated_voice_face_cooccurrence"
    if "context" in text or "narrative" in text:
        return "context_only"
    if "voice" in text:
        return "voice_only"
    return "none" if not text else "unknown"


def _speech_unit_count(text: str) -> int:
    """Count coarse spoken units for Latin-token and CJK ASR sanity checks."""
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u3400-\u9fff]", text))


def align_authoritative_dialogue(
    segments: list[dict[str, Any]],
    asr_timeline: Sequence[dict[str, Any]],
    registered_ids: set[str],
) -> None:
    """Make ASR immutable while retaining only MiMo's speaker attribution judgment."""
    if not asr_timeline:
        return
    supplied = {str(row["asr_id"]): row for row in asr_timeline}
    model_rows: dict[str, dict[str, Any]] = {}
    for segment in segments:
        for row in segment.get("dialogue") or []:
            if not isinstance(row, dict):
                raise ValueError("dialogue rows must be objects when authoritative ASR is supplied")
            asr_id = str(row.get("asr_id") or "").strip()
            if asr_id not in supplied:
                raise ValueError(f"MiMo invented or omitted an ASR ID in dialogue: {asr_id!r}")
            if asr_id in model_rows:
                raise ValueError(f"MiMo returned duplicate dialogue for ASR ID: {asr_id}")
            model_rows[asr_id] = row
        segment["dialogue"] = []

    for asr in asr_timeline:
        asr_id = str(asr["asr_id"])
        start, end = number_range(asr["time_range_sec"], f"{asr_id}.time_range_sec")
        attribution = model_rows.get(asr_id, {})
        raw_speaker = str(attribution.get("speaker_id") or "unknown").strip()
        speaker_id = raw_speaker if raw_speaker in registered_ids else "unknown"
        confidence = _attribution_confidence(
            attribution.get("attribution_confidence", attribution.get("confidence"))
        )
        overlap_status = str(attribution.get("overlap_status") or "unknown").strip().lower()
        if overlap_status not in {"none", "possible", "overlap", "unknown"}:
            overlap_status = "unknown"
        is_off_screen = bool(attribution.get("is_off_screen", False))
        attribution_evidence = str(
            attribution.get("attribution_evidence") or attribution.get("evidence") or
            ("MiMo returned no attribution for this ASR row" if not attribution else "")
        ).strip()
        evidence_type = _attribution_evidence_type(
            attribution.get("attribution_evidence_type"), attribution_evidence
        )
        direct_identity_evidence = evidence_type in DIRECT_ATTRIBUTION_EVIDENCE_TYPES
        asr_text = str(asr.get("text") or "").strip()
        span_duration = max(0.0, end - start)
        speech_unit_count = _speech_unit_count(asr_text)
        speech_units_per_second = speech_unit_count / span_duration if span_duration > 0 else 0.0
        asr_timing_asset_safe = bool(
            span_duration <= MAX_AUDIO_ASSET_ASR_SPAN_SECONDS
            and speech_units_per_second >= MIN_AUDIO_ASSET_SPEECH_UNITS_PER_SECOND
        )
        attribution_accepted = bool(
            speaker_id != "unknown" and confidence == "high" and overlap_status == "none"
            and not is_off_screen and direct_identity_evidence
            and attribution_evidence and asr_text
        )
        eligible = attribution_accepted and asr_timing_asset_safe
        speech_clarity = str(attribution.get("speech_clarity") or "unknown").strip().lower()
        if speech_clarity not in {"clean", "usable", "poor", "unknown"}:
            speech_clarity = "unknown"
        background_interference = str(
            attribution.get("background_interference") or "unknown"
        ).strip().lower()
        if background_interference not in {"none", "mild", "strong", "unknown"}:
            background_interference = "unknown"
        rejection_reasons = []
        if speaker_id == "unknown":
            rejection_reasons.append("speaker_unresolved")
        if confidence != "high":
            rejection_reasons.append("speaker_attribution_not_high_confidence")
        if overlap_status != "none":
            rejection_reasons.append(f"overlap_status_{overlap_status}")
        if is_off_screen:
            rejection_reasons.append("speaker_off_screen")
        if not attribution_evidence:
            rejection_reasons.append("missing_attribution_evidence")
        if not direct_identity_evidence:
            rejection_reasons.append(f"non_direct_attribution_evidence_{evidence_type}")
        if span_duration > MAX_AUDIO_ASSET_ASR_SPAN_SECONDS:
            rejection_reasons.append("asr_span_too_long_for_direct_asset_cut")
        if speech_units_per_second < MIN_AUDIO_ASSET_SPEECH_UNITS_PER_SECOND:
            rejection_reasons.append("asr_speech_density_too_low_for_direct_asset_cut")
        if not asr_text:
            rejection_reasons.append("empty_asr_text")

        def overlap_score(segment: dict[str, Any]) -> tuple[float, float]:
            segment_start, segment_end = number_range(segment["time_range_sec"], "segment time_range_sec")
            overlap = max(0.0, min(end, segment_end) - max(start, segment_start))
            distance = abs((start + end) / 2.0 - (segment_start + segment_end) / 2.0)
            return overlap, -distance

        target = max(segments, key=overlap_score)
        segment_start = float(target["time_range_sec"][0])
        target["dialogue"].append({
            "asr_id": asr_id,
            "speaker_id": speaker_id,
            "text": asr["text"],
            "language": asr.get("language", "unknown"),
            "time_range_sec": [round(start, 6), round(end, 6)],
            "segment_time_range_sec": [round(start-segment_start, 6), round(end-segment_start, 6)],
            "source_time_range_sec": list(asr["source_time_range_sec"]),
            "asr_confidence": asr.get("confidence"),
            "asr_span_duration_sec": round(span_duration, 6),
            "asr_speech_unit_count": speech_unit_count,
            "asr_speech_units_per_second": round(speech_units_per_second, 6),
            "asr_timing_asset_safe": asr_timing_asset_safe,
            "attribution_confidence": confidence,
            "attribution_evidence": attribution_evidence,
            "attribution_evidence_type": evidence_type,
            "direct_identity_evidence": direct_identity_evidence,
            "is_off_screen": is_off_screen,
            "overlap_status": overlap_status,
            "speech_clarity": speech_clarity,
            "background_interference": background_interference,
            "speaker_attribution_accepted": attribution_accepted,
            "audio_asset_candidate": eligible,
            "audio_asset_admission_status": (
                "pending_audio_quality_validation" if eligible else "rejected_by_attribution_gate"
            ),
            "audio_asset_rejection_reasons": rejection_reasons,
            "speaker_attribution_source": "mimo" if attribution else "deterministic_unknown_fallback",
        })
    for segment in segments:
        segment["dialogue"].sort(key=lambda row: (row["time_range_sec"][0], row["asr_id"]))


def flatten_speaker_attributions(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the direct input contract for a later timestamp-based audio cutter."""
    rows = []
    for result in results:
        if result.get("success") is not True:
            continue
        payload = result.get("segment_caption") or {}
        for segment in payload.get("segments") or []:
            for dialogue in segment.get("dialogue") or []:
                rows.append({
                    "video_id": result.get("video_id"),
                    "global_id": result.get("global_id"),
                    "segment_id": segment.get("segment_id"),
                    "global_clip_path": result.get("clip_path"),
                    "source_video": result.get("source_video"),
                    **copy.deepcopy(dialogue),
                })
    return sorted(rows, key=lambda row: (
        str(row.get("video_id")), str(row.get("global_id")),
        float((row.get("source_time_range_sec") or [0.0])[0]), str(row.get("asr_id")),
    ))


def validate_and_enrich(payload: dict, shots: list[dict], global_record: dict, duration: float,
                        asr_timeline: Sequence[dict[str, Any]] = ()) -> dict:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("response segments must be a non-empty list")
    expected = [int(shot["shot_id"]) for shot in shots]
    actual: list[int] = []
    shot_map = {int(shot["shot_id"]): shot for shot in shots}
    global_payload = unwrap_nested_caption(global_record, "global_caption")
    registered_ids = {
        str(item.get("subject_id"))
        for item in global_payload.get("subject_registry", [])
        if isinstance(item, dict) and item.get("subject_id")
    }

    def normalize_unknown_id(value: str) -> str:
        lowered = value.lower().replace("-", "_")
        if "unknown" in lowered:
            return "unknown"
        if "new_subject_candidate" in lowered:
            return "new_subject_candidate"
        return value

    enriched = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} is not an object")
        ids = segment.get("shot_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"segment {index} has no shot_ids")
        ids = [int(value) for value in ids]
        if ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f"segment {index} shot_ids are not consecutive: {ids}")
        actual.extend(ids)
        description = segment.get("description", "")
        if not isinstance(description, str) or not description.strip():
            fallback_actions = []
            for shot_id in ids:
                shot_caption = unwrap_nested_caption(shot_map[shot_id], "shot_caption")
                shot_description = description_value(shot_caption, "shot_description")
                if shot_description:
                    fallback_actions.append(shot_description)
            description = " ".join(fallback_actions).strip() or "The segment contains the connected events shown in its referenced shots."
        scene_description = segment.get("scene_description", "")
        if not isinstance(scene_description, str) or not scene_description.strip():
            global_scene = unwrap_nested_caption(global_record, "global_caption").get("global_scene_description", "")
            shot_scenes = []
            for shot_id in ids:
                shot_scene = unwrap_nested_caption(shot_map[shot_id], "shot_caption").get("shot_scene_description", "")
                if isinstance(shot_scene, str) and shot_scene.strip() and shot_scene not in shot_scenes:
                    shot_scenes.append(shot_scene.strip())
            scene_description = str(global_scene).strip() or " ".join(shot_scenes) or "The scene remains as shown in the global clip."
        phases = {"introduction", "continuation", "escalation", "turning_point", "resolution", "transition"}
        event_phase = segment.get("event_phase")
        if event_phase is None:
            event_phase = "introduction" if index == 1 else "continuation"
        if event_phase not in phases:
            raise ValueError(f"segment {index} has invalid event_phase: {segment.get('event_phase')!r}")
        relation = segment.get("relation_to_previous", "N/A" if index == 1 else "")
        if not isinstance(relation, str):
            raise ValueError(f"segment {index} relation_to_previous must be a string")
        if not relation.strip():
            relation = "N/A" if index == 1 else "continues from the previous segment"
            segment["relation_to_previous"] = relation
        normalized_key_subjects = []
        for subject_id in segment.get("key_subject_ids", []):
            subject_id = normalize_unknown_id(str(subject_id))
            if subject_id != "unknown" and subject_id not in registered_ids:
                raise ValueError(f"segment {index} references unregistered key subject: {subject_id}")
            normalized_key_subjects.append(subject_id)
        segment["key_subject_ids"] = normalized_key_subjects
        for dialogue_index, dialogue in enumerate(segment.get("dialogue", []), start=1):
            if isinstance(dialogue, dict):
                raw_speaker_id = str(dialogue.get("speaker_id", "unknown"))
                speaker_id = normalize_unknown_id(raw_speaker_id).strip().strip("()[]{}").strip()
                speakers = [
                    normalize_unknown_id(part.strip())
                    for part in re.split(r"\s*[,;]\s*", speaker_id)
                ]
                if asr_timeline and len(speakers) != 1:
                    raise ValueError(
                        f"segment {index} dialogue[{dialogue_index}] must map one ASR row to one speaker"
                    )
                invalid = [
                    part for part in speakers
                    if part != "unknown" and part not in registered_ids
                ]
                if invalid:
                    raise ValueError(
                        f"segment {index} dialogue[{dialogue_index}] references unregistered speaker: {', '.join(invalid)}"
                    )
                dialogue["speaker_id"] = ", ".join(speakers)
        first = shot_map[ids[0]]
        last = shot_map[ids[-1]]
        local_start, _ = number_range(first["time_range_sec"], "segment start")
        _, local_end = number_range(last["time_range_sec"], "segment end")
        source_start, _ = number_range(first["source_time_range_sec"], "source segment start")
        _, source_end = number_range(last["source_time_range_sec"], "source segment end")
        result = copy.deepcopy(segment)
        result["segment_id"] = index
        result["shot_ids"] = ids
        result["description"] = description.strip()
        result["scene_description"] = scene_description.strip()
        result["event_phase"] = event_phase
        result["relation_to_previous"] = relation.strip() or "N/A"
        result.pop("semantic_unit", None)
        result["time_range_sec"] = [local_start, local_end]
        result["source_time_range_sec"] = [source_start, source_end]
        result["source_shots"] = [copy.deepcopy(shot_map[shot_id]) for shot_id in ids]
        result.setdefault("segment_caption", {})
        if not isinstance(result["segment_caption"], dict):
            result["segment_caption"] = {}
        model_description = description_value(result["segment_caption"], "segment_description")
        if not isinstance(model_description, str) or not has_all_shot_labels(model_description, ids):
            model_description = fallback_shot_labeled_description(result, shot_map)
        else:
            model_description = add_dialogue_tags(model_description, dialogue_tags(result, shot_map))
        model_description = deduplicate_dialogue_tags(model_description)
        model_description = remove_scenetrans_markers(model_description)
        model_description = rewrite_shot_timestamps(model_description, result, shot_map)
        result["segment_caption"].pop("integrated_multimodal_description", None)
        result["segment_caption"]["segment_description"] = model_description
        enriched.append(result)
    if actual != expected:
        raise ValueError(f"segments do not cover shots exactly once: expected {expected}, got {actual}")
    align_authoritative_dialogue(enriched, asr_timeline, registered_ids)
    output = copy.deepcopy(payload)
    output["video_duration_sec"] = round(duration, 6)
    output["segments"] = enriched
    output["global_id"] = global_record.get("global_id")
    output["global_caption"] = copy.deepcopy(global_record)
    output["authoritative_asr_timeline"] = list(asr_timeline)
    output.pop("subject_registry", None)
    return output


# ---------------------------------------------------------------------------
# Async annotation
# ---------------------------------------------------------------------------
async def annotate_one(client, semaphore, global_record, shots, prompt, args,
                       asr_timeline: Sequence[dict[str, Any]] = ()):
    started = time.perf_counter()
    duration = float(global_record["duration_sec"])
    request_text = build_input_text(prompt, global_record, shots, duration, asr_timeline)
    text_only = getattr(args, "text_only", False)
    if text_only:
        request_text += (
            "\nTEXT-ONLY AGGREGATION: No video is attached. Treat the supplied global "
            "and complete shot captions as the sole evidence. Do not invent missing "
            "visual or audio details. Preserve the required JSON schema, speaker IDs, "
            "and exact once-only coverage of every supplied shot.\n"
        )
    last_response = ""
    async with semaphore:
        encoded = "" if text_only else await asyncio.to_thread(encode_video_b64, Path(global_record["clip_path"]))
        formats = ["text_only"] if text_only else (
            [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
        )
        for current_format in formats:
            try:
                extra_body = {"chat_template_kwargs": {"enable_thinking": args.thinking}} if args.thinking else {}
                request = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "user", "content": (
                            [{"type": "text", "text": request_text}] if text_only
                            else build_video_content(request_text, encoded, current_format)
                        )}
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_completion_tokens,
                    extra_body=extra_body,
                )
                response = await asyncio.wait_for(request, timeout=args.max_api_seconds or args.timeout)
                content = extract_response_text(response)
                if not content:
                    last_response = response_diagnostic(response)
                    last_error = "response contains no textual content"
                    raise ValueError(last_error)
                last_response = content
                parsed, error = parse_json_response(content)
                if error is None and parsed is not None:
                    enriched = validate_and_enrich(parsed, shots, global_record, duration, asr_timeline)
                    return {
                        **global_record,
                        "success": True,
                        "model": args.model,
                        "request_format": current_format,
                        "segment_caption": enriched,
                        "response_seconds": round(time.perf_counter() - started, 3),
                        "_raw_response_text": content,
                    }
                last_error = error or "invalid segment response"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{current_format}: {type(exc).__name__}: {exc}"
    return {**global_record, "success": False, "model": args.model, "error": last_error,
            "response_text": last_response or None, "response_seconds": round(time.perf_counter() - started, 3)}


async def annotate_manifest(global_manifest, shot_manifest, args, client):
    global_records = [item for item in load_jsonl(global_manifest) if item.get("success") is True]
    shot_records = [item for item in load_jsonl(shot_manifest) if item.get("success") is True]
    shot_map = {int(item["shot_id"]): item for item in shot_records}
    asr_index = load_asr_sidecars(getattr(args, "asr_input", None))
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    output_dir = args.output_dir / global_manifest.parent.name
    segment_dir = output_dir / "segments"
    raw_dir = output_dir / "raw_responses"
    segment_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    jsonl_path = output_dir / "segment_captions.jsonl"

    all_results: dict[int, dict] = {}
    if args.skip_existing:
        for record in global_records:
            path = segment_dir / f"global_{int(record['global_id']):05d}.json"
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and existing.get("success") is True:
                        if getattr(args, "asr_input", None):
                            expected_asr = prepare_authoritative_asr(
                                record, asr_index, getattr(args, "asr_timebase", "auto")
                            )
                            existing_asr = (
                                existing.get("segment_caption") or {}
                            ).get("authoritative_asr_timeline")
                            if existing_asr != expected_asr:
                                LOGGER.info(
                                    "invalidate cached global %s: authoritative ASR changed",
                                    record.get("global_id"),
                                )
                                continue
                        all_results[int(record["global_id"])] = existing
                except (OSError, json.JSONDecodeError):
                    pass

    rounds = max(1, args.retries)
    base_temperature = args.temperature
    for round_no in range(1, rounds + 1):
        args.temperature = min(1.0, base_temperature + 0.2 * (round_no - 1))
        pending = [r for r in global_records if all_results.get(int(r["global_id"]), {}).get("success") is not True]
        if pending:
            tasks = []
            for record in pending:
                clip_path = resolve_clip_path(record.get("clip_path", ""), global_manifest)
                if not getattr(args, "text_only", False) and not clip_path.is_file():
                    raise FileNotFoundError(f"Global clip does not exist: {clip_path}")
                record = {**record, "clip_path": str(clip_path)}
                shots, _ = prepare_shots(record, shot_map)
                asr_timeline = prepare_authoritative_asr(
                    record, asr_index, getattr(args, "asr_timebase", "auto")
                )
                tasks.append(asyncio.create_task(
                    annotate_one(client, semaphore, record, shots, prompt, args, asr_timeline)
                ))
            progress = None
            if args.progress:
                try:
                    from tqdm import tqdm
                    progress = tqdm(
                        total=len(tasks),
                        desc=f"Segment caption {global_manifest.parent.name} (round {round_no}/{rounds})",
                        unit="clip",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass
            ok = failed = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                gid = int(result["global_id"])
                raw_response = result.pop("_raw_response_text", None)
                all_results[gid] = result
                persist_item(
                    segment_dir, raw_dir, "global", gid, result,
                    raw_response or result.get("response_text"),
                )
                ok += int(result.get("success") is True)
                failed += int(result.get("success") is not True)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(ok=ok, fail=failed, refresh=False)
            if progress is not None:
                progress.close()

        ordered = [all_results[int(record["global_id"])] for record in global_records]
        write_jsonl(jsonl_path, ordered)
        attribution_path = output_dir / "speaker_attributed_asr.jsonl"
        attribution_rows = flatten_speaker_attributions(ordered)
        write_jsonl(attribution_path, attribution_rows)
        total_success = sum(1 for result in all_results.values() if result.get("success"))
        total_failed = len(all_results) - total_success
        summary = {
            "global_manifest": str(global_manifest),
            "shot_manifest": str(shot_manifest),
            "global_clip_count": len(global_records),
            "success_count": total_success,
            "failed_count": total_failed,
            "rounds_used": round_no,
            "model": args.model,
            "asr_input": str(args.asr_input) if getattr(args, "asr_input", None) else None,
            "speaker_attribution_count": len(attribution_rows),
            "speaker_attribution_output": str(attribution_path),
            "output": str(jsonl_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        LOGGER.info(
            "%s: round %d/%d: %d/%d segment captions succeeded",
            global_manifest.parent.name, round_no, rounds, total_success, len(global_records),
        )
        if not pending or total_failed == 0:
            break
        LOGGER.warning("%s: round %d left %d failures, running next round", global_manifest.parent.name, round_no, total_failed)
        backoff = min(300.0, 60.0 * (2 ** (round_no - 1)))
        LOGGER.info("%s: sleeping %.0fs before retry round %d", global_manifest.parent.name, backoff, round_no + 1)
        await asyncio.sleep(backoff)
    return summary


async def main_async(args):
    from openai import AsyncOpenAI
    global_manifest = find_manifest(args.global_input, "global_captions.jsonl")
    shot_manifest = find_manifest(args.shot_input, "shot_captions.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    summary = await annotate_manifest(global_manifest, shot_manifest, args, client)
    return 0 if summary["failed_count"] == 0 else 1


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Create segment captions from global and shot captions.")
    parser.add_argument("--global-input", required=True, type=Path)
    parser.add_argument("--shot-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asr-input", type=Path,
                        help="ASR JSON/JSONL file or directory; precise text/timestamps are injected into MiMo")
    parser.add_argument("--asr-timebase", choices=("auto", "source", "global-local"), default="auto",
                        help="Timebase for ASR rows without an explicit source/local range field")
    parser.add_argument("--prompt-file", type=Path, default=default_prompt_path("segment"))
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=("video_url", "input_video", "auto"), default="video_url")
    parser.add_argument("--text-only", action="store_true",
                        help="Use only complete global/shot captions; do not encode or upload video.")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries, and timeout must be positive")
    if args.max_api_seconds is not None and args.max_api_seconds <= 0:
        parser.error("--max-api-seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).disabled = True
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
