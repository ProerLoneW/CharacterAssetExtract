#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1.5: align MiMo semantic characters to shot-local Tracklets.

This module has one responsibility: combine hierarchical MiMo captions with
the complete person proposals emitted by Stage 1, then produce conservative
Tracklet selections for Stage 3.  It never mutates or deletes Stage-1 output.

For every shot, MiMo receives the existing ``tracklet_overlay.mp4`` (whose
boxes/masks carry Tracklet IDs), compact global/segment/shot semantic context,
and a Tracklet inventory.  The validated output maps stable ``S`` IDs to
Tracklets and classifies each Tracklet as primary, supporting, contextual,
background, non-person, or unresolved.

Three downstream views are exported without copying image assets:

``stage3_inputs/primary``
    Only confidently aligned primary-character Tracklets.
``stage3_inputs/narrative``
    Primary and supporting-character Tracklets.
``stage3_inputs/safe``
    Fail-open view: removes only high-confidence background/non-person tracks.

Each view contains normal ``tracklet_signatures.json`` files and can therefore
be passed directly to ``visual_character_registry_builder.py --inputs``.
It also contains ``semantic_identity_decisions.json`` for the builder's
``--vlm-decisions`` input.  Those decisions only assert cross-shot identity
when MiMo assigned the same global-scoped subject ID to two high-confidence
person Tracklets.  Stage 3 still applies them solely to visually plausible
``POSSIBLE_LINK`` edges, and its cannot-link constraints remain authoritative.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import sys as _sys

# EFP 布局：本文件位于 assets/visual_person/，mimo_caption_client 在 EFP 根目录
_EFP_ROOT = Path(__file__).resolve().parents[2]
if str(_EFP_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_EFP_ROOT))

from mimo_caption_client import (
    build_video_content,
    encode_video_b64,
    extract_response_text,
    load_dotenv,
    parse_json_response,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)


LOGGER = logging.getLogger("semantic_role_tracklet_refiner")
SCHEMA_VERSION = "semantic-role-tracklet-alignment-v1"
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "tracklet_semantic_alignment_prompt_en_v1.md"

IMPORTANCE_TIERS = {"primary", "supporting", "incidental", "background", "unknown"}
ENTITY_TYPES = {"person", "non_person", "uncertain"}
DECISIONS = {
    "keep_primary", "keep_supporting", "keep_context",
    "reject_background", "reject_non_person", "review",
}
KEEP_BY_POLICY = {
    "primary": {"keep_primary"},
    "narrative": {"keep_primary", "keep_supporting"},
}
SUBJECT_PATTERN = re.compile(r"S[1-9][0-9]*$")
SHOT_NUMBER_PATTERN = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ShotInput:
    shot_id: str
    signature_path: Path
    overlay_path: Path
    signature_payload: Dict[str, Any]

    @property
    def tracklets(self) -> list[dict]:
        value = self.signature_payload.get("tracklet_signatures", [])
        return value if isinstance(value, list) else []


class AsyncRateGate:
    """Serialize request starts at the service-safe rate used by m06."""

    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self.lock = asyncio.Lock()
        self.last_send = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self.lock:
            now = asyncio.get_running_loop().time()
            delay = self.interval - (now - self.last_send)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last_send = asyncio.get_running_loop().time()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _find_manifest(path: Optional[Path], expected_name: str) -> Optional[Path]:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != expected_name:
            raise ValueError(f"expected {expected_name}, got {path}")
        return path
    candidates = sorted(path.rglob(expected_name))
    if not candidates:
        raise FileNotFoundError(f"no {expected_name} found under {path}")
    if len(candidates) > 1:
        raise ValueError(f"multiple {expected_name} files under {path}; pass one explicitly")
    return candidates[0]


def _unwrap(record: dict, key: str) -> dict:
    value = record.get(key)
    if not isinstance(value, dict):
        return {}
    nested = value.get(key)
    if not isinstance(nested, dict):
        return value
    result = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    result.update(copy.deepcopy(nested))
    return result


def _shot_number(value: Any) -> Optional[int]:
    match = SHOT_NUMBER_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def discover_shots(tracklet_input: Path) -> list[ShotInput]:
    root = tracklet_input.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted(root.rglob("tracklet_signatures.json"))
    shots: list[ShotInput] = []
    seen: set[str] = set()
    for path in paths:
        if path.name != "tracklet_signatures.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        shot_id = str(payload.get("shot_id") or path.parent.name)
        if shot_id in seen:
            raise ValueError(f"duplicate shot_id {shot_id}: {path}")
        seen.add(shot_id)
        overlay = path.parent / "tracklet_overlay.mp4"
        shots.append(ShotInput(shot_id, path, overlay, payload))
    if not shots:
        raise FileNotFoundError(f"no tracklet_signatures.json found under {root}")
    return sorted(shots, key=lambda item: (_shot_number(item.shot_id) or 10**12,
                                           item.shot_id))


def _successful(row: dict) -> bool:
    return row.get("success") is not False


def load_caption_context(global_input: Optional[Path], shot_input: Optional[Path],
                         segment_input: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Index compact caption evidence by numeric shot identity."""
    by_number: dict[int, dict[str, Any]] = {}

    global_manifest = _find_manifest(global_input, "global_captions.jsonl")
    if global_manifest:
        for row in _load_jsonl(global_manifest):
            if not _successful(row):
                continue
            payload = _unwrap(row, "global_caption")
            compact = {
                "global_id": row.get("global_id"),
                "source_time_range_sec": row.get("time_range_sec"),
                "global_caption": payload.get("global_caption", payload.get("global_description", "")),
                "global_scene_description": payload.get("global_scene_description", ""),
                "subject_registry": payload.get("subject_registry", []),
                "semantic_role_summary": payload.get("semantic_role_summary", {}),
                "semantic_role_observations": payload.get("semantic_role_observations", []),
                "subject_event_timeline": payload.get("subject_event_timeline", []),
                "global_event_timeline": payload.get("global_event_timeline", []),
            }
            for raw_shot_id in row.get("shot_ids", []):
                number = _shot_number(raw_shot_id)
                if number is not None:
                    by_number.setdefault(number, {})["global"] = compact

    shot_manifest = _find_manifest(shot_input, "shot_captions.jsonl")
    if shot_manifest:
        for row in _load_jsonl(shot_manifest):
            if not _successful(row):
                continue
            number = _shot_number(row.get("shot_id"))
            if number is None:
                continue
            payload = _unwrap(row, "shot_caption")
            by_number.setdefault(number, {})["shot"] = {
                "shot_id": row.get("shot_id"),
                "time_range_sec": row.get("time_range_sec"),
                "shot_caption": payload.get("shot_caption", payload),
                "visible_count": payload.get("visible_count"),
                "subjects": payload.get("subjects", []),
                "semantic_role_summary": payload.get("semantic_role_summary", {}),
                "semantic_role_observations": payload.get("semantic_role_observations", []),
                "visual_events": payload.get("visual_events", []),
                "dialogue": payload.get("dialogue", []),
            }

    segment_manifest = _find_manifest(segment_input, "segment_captions.jsonl")
    if segment_manifest:
        for row in _load_jsonl(segment_manifest):
            if not _successful(row):
                continue
            payload = _unwrap(row, "segment_caption")
            segments = payload.get("segments", [])
            if not isinstance(segments, list):
                continue
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                compact = {
                    "global_id": row.get("global_id"),
                    "segment_id": segment.get("segment_id"),
                    "time_range_sec": segment.get("time_range_sec"),
                    "shot_ids": segment.get("shot_ids", []),
                    "description": segment.get("description", ""),
                    "key_subject_ids": segment.get("key_subject_ids", []),
                    "semantic_role_summary": segment.get("semantic_role_summary", {}),
                    "semantic_role_observations": segment.get("semantic_role_observations", []),
                    "dialogue": segment.get("dialogue", []),
                }
                for raw_shot_id in segment.get("shot_ids", []):
                    number = _shot_number(raw_shot_id)
                    if number is not None:
                        by_number.setdefault(number, {})["segment"] = compact
    return {str(key): value for key, value in by_number.items()}


def allowed_subject_ids(context: dict[str, Any]) -> list[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"subject_id", "id"} and SUBJECT_PATTERN.fullmatch(str(item or "")):
                    found.add(str(item))
                elif key.endswith("_subject_ids") or key in {"subject_ids", "key_subject_ids", "ids"}:
                    if isinstance(item, list):
                        for raw in item:
                            if SUBJECT_PATTERN.fullmatch(str(raw or "")):
                                found.add(str(raw))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(context)
    return sorted(found, key=lambda value: int(value[1:]))


def subject_importance(context: dict[str, Any]) -> dict[str, str]:
    """Resolve stable role tiers, preferring the global registry and summary."""
    result: dict[str, str] = {}
    global_context = context.get("global", {})
    for subject in global_context.get("subject_registry", []):
        if not isinstance(subject, dict):
            continue
        subject_id = str(subject.get("subject_id") or "")
        raw_tier = subject.get("importance_tier")
        tier = str(raw_tier) if raw_tier is not None else ""
        if SUBJECT_PATTERN.fullmatch(subject_id) and tier in IMPORTANCE_TIERS:
            result[subject_id] = tier
    summary_fields = {
        "primary_subject_ids": "primary",
        "supporting_subject_ids": "supporting",
        "incidental_subject_ids": "incidental",
        "background_subject_ids": "background",
        "unknown_subject_ids": "unknown",
    }
    # Global is authoritative. Segment and shot summaries only fill IDs absent
    # from older global-caption schemas; they never override a global tier.
    for level in ("global", "segment", "shot"):
        summary = context.get(level, {}).get("semantic_role_summary", {})
        if not isinstance(summary, dict):
            continue
        for field, tier in summary_fields.items():
            for raw in summary.get(field, []):
                subject_id = str(raw)
                if SUBJECT_PATTERN.fullmatch(subject_id):
                    result.setdefault(subject_id, tier)
    return result


def semantic_scope_id(context: dict[str, Any]) -> str:
    """Return the global-local namespace in which S IDs are stable."""
    for level in ("global", "segment"):
        value = context.get(level, {})
        if isinstance(value, dict) and value.get("global_id") is not None:
            raw = str(value["global_id"])
            if raw.isdigit():
                return f"global_{int(raw):04d}"
            return raw
    return "unscoped"


def tracklet_inventory(shot: ShotInput) -> list[dict]:
    inventory = []
    for item in shot.tracklets:
        faces = sorted(item.get("face_gallery", []),
                       key=lambda value: float(value.get("quality_score", 0.0)), reverse=True)
        bodies = sorted(item.get("body_gallery", []),
                        key=lambda value: float(value.get("quality_score", 0.0)), reverse=True)
        inventory.append({
            "tracklet_id": str(item.get("tracklet_id")),
            "frame_range": [item.get("start_frame"), item.get("end_frame")],
            "time_range_sec": [item.get("start_time"), item.get("end_time")],
            "observation_count": len(item.get("observation_ids", [])),
            "face_evidence_count": len(faces),
            "best_face": ({key: faces[0].get(key) for key in
                           ("observation_id", "frame_index", "det_score", "quality_score")}
                          if faces else None),
            "body_evidence_count": len(bodies),
            "best_body": ({key: bodies[0].get(key) for key in
                           ("observation_id", "frame_index", "grounding_score", "quality_score")}
                          if bodies else None),
            "cannot_links": list(item.get("cannot_links", item.get("cooccurring_tracklets", []))),
            "track_quality": item.get("quality", {}),
        })
    return inventory


def build_request_prompt(base_prompt: str, shot: ShotInput,
                         context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    inventory = tracklet_inventory(shot)
    package = {
        "shot_id": shot.shot_id,
        "allowed_semantic_subject_ids": allowed_subject_ids(context),
        "authoritative_subject_importance": subject_importance(context),
        "caption_evidence": context,
        "tracklet_inventory": inventory,
    }
    prompt = (base_prompt.rstrip()
              + "\n\nINPUT PACKAGE\n"
              + json.dumps(package, ensure_ascii=False, indent=2))
    return prompt, package


def _finite_confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(result, 0.0), 1.0), 6) if math.isfinite(result) else 0.0


def normalize_alignment(payload: dict, shot: ShotInput,
                        allowed_ids: Sequence[str], reject_confidence: float,
                        authoritative_tiers: Optional[dict[str, str]] = None) -> dict:
    """Validate MiMo output and apply deterministic fail-open safeguards."""
    if str(payload.get("shot_id")) != shot.shot_id:
        raise ValueError(f"response shot_id {payload.get('shot_id')!r} != {shot.shot_id}")
    expected = [str(item.get("tracklet_id")) for item in shot.tracklets]
    expected_set = set(expected)
    rows = payload.get("tracklet_alignments")
    if not isinstance(rows, list):
        raise ValueError("tracklet_alignments must be a list")
    by_id: dict[str, dict] = {}
    allowed = set(allowed_ids) | {"unknown", "new_subject_candidate"}
    authoritative_tiers = authoritative_tiers or {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"tracklet_alignments[{index}] must be an object")
        tracklet_id = str(raw.get("tracklet_id") or "")
        if tracklet_id not in expected_set:
            raise ValueError(f"unknown tracklet_id in response: {tracklet_id}")
        if tracklet_id in by_id:
            raise ValueError(f"duplicate alignment for {tracklet_id}")
        entity_type = str(raw.get("entity_type") or "uncertain")
        importance = str(raw.get("importance_tier") or "unknown")
        decision = str(raw.get("decision") or "review")
        subject_id = str(raw.get("subject_id") or "unknown")
        confidence = _finite_confidence(raw.get("confidence"))
        if entity_type not in ENTITY_TYPES:
            entity_type = "uncertain"
        if importance not in IMPORTANCE_TIERS:
            importance = "unknown"
        if decision not in DECISIONS:
            decision = "review"
        if subject_id not in allowed:
            raise ValueError(f"unregistered semantic subject ID {subject_id} for {tracklet_id}")
        authoritative = authoritative_tiers.get(subject_id)
        if authoritative in IMPORTANCE_TIERS:
            importance = authoritative
        safety_reason = None
        if entity_type == "non_person":
            subject_id, importance = "unknown", "background"
            if confidence < reject_confidence:
                decision, safety_reason = "review", "low_confidence_non_person_rejection"
            else:
                decision = "reject_non_person"
        elif decision in {"reject_background", "reject_non_person"} and confidence < reject_confidence:
            decision, safety_reason = "review", "low_confidence_rejection"
        elif decision == "reject_non_person" and entity_type != "non_person":
            decision, safety_reason = "review", "entity_type_decision_mismatch"
        elif decision == "reject_background" and entity_type != "person":
            decision, safety_reason = "review", "background_rejection_requires_person"
        elif decision.startswith("keep_") and entity_type != "person":
            decision, safety_reason = "review", "keep_decision_requires_person"
        elif decision == "keep_primary" and (importance != "primary"
                                               or not SUBJECT_PATTERN.fullmatch(subject_id)):
            decision, safety_reason = "review", "primary_requires_registered_primary_subject"
        elif decision == "keep_supporting" and (importance != "supporting"
                                                  or not SUBJECT_PATTERN.fullmatch(subject_id)):
            decision, safety_reason = "review", "supporting_requires_registered_supporting_subject"
        elif decision == "keep_context" and importance not in {"incidental", "unknown"}:
            decision, safety_reason = "review", "context_importance_mismatch"
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
        by_id[tracklet_id] = {
            "tracklet_id": tracklet_id,
            "entity_type": entity_type,
            "subject_id": subject_id,
            "importance_tier": importance,
            "decision": decision,
            "confidence": confidence,
            "evidence": evidence,
            "safety_adjustment": safety_reason,
        }

    normalized = []
    for tracklet_id in expected:
        normalized.append(by_id.get(tracklet_id, {
            "tracklet_id": tracklet_id,
            "entity_type": "uncertain",
            "subject_id": "unknown",
            "importance_tier": "unknown",
            "decision": "review",
            "confidence": 0.0,
            "evidence": {"caption_levels": [], "time_ranges_sec": [],
                         "visual_cues": [], "reason": "missing_from_model_response"},
            "safety_adjustment": "missing_alignment_fail_open",
        }))

    tracklet_source = {str(item.get("tracklet_id")): item for item in shot.tracklets}
    subject_owner: dict[str, list[dict]] = {}
    for row in normalized:
        if SUBJECT_PATTERN.fullmatch(row["subject_id"]):
            subject_owner.setdefault(row["subject_id"], []).append(row)
    # A VLM that puts two co-occurring tracklets under one subject is wrong about
    # one of them, not about the whole shot: keep the first and demote the rest to
    # unassigned instead of failing the entire shot open.
    dropped: set[str] = set()
    for assigned in subject_owner.values():
        kept: list[dict] = []
        for row in assigned:
            source = tracklet_source[row["tracklet_id"]]
            row_cannot = set(source.get("cannot_links", source.get("cooccurring_tracklets", [])))
            if any(other["tracklet_id"] in row_cannot for other in kept):
                dropped.add(row["tracklet_id"])
                continue
            kept.append(row)
    for row in normalized:
        if row["tracklet_id"] in dropped:
            LOGGER.warning("%s: dropping cannot-link-conflicting assignment %s (was %s)",
                           shot.shot_id, row["tracklet_id"], row["subject_id"])
            row["subject_id"] = "unknown"
            row["decision"] = "review"
            row["safety_adjustment"] = "cannot_link_conflict_dropped"

    unmatched = payload.get("unmatched_semantic_subject_ids", [])
    unmatched = [str(value) for value in unmatched if str(value) in allowed]
    return {
        "schema_version": SCHEMA_VERSION,
        "shot_id": shot.shot_id,
        "status": "ok",
        "tracklet_alignments": normalized,
        "unmatched_semantic_subject_ids": sorted(set(unmatched)),
        "unmatched_tracklet_ids": [row["tracklet_id"] for row in normalized
                                    if row["decision"] == "review"],
        "quality_issues": payload.get("quality_issues", [])
        if isinstance(payload.get("quality_issues"), list) else [],
    }


def fail_open_alignment(shot: ShotInput, reason: str, *, dry_run: bool = False) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "shot_id": shot.shot_id,
        "status": "planned" if dry_run else "failed_open",
        "error": reason,
        "tracklet_alignments": [{
            "tracklet_id": str(item.get("tracklet_id")),
            "entity_type": "uncertain", "subject_id": "unknown",
            "importance_tier": "unknown", "decision": "review", "confidence": 0.0,
            "evidence": {"caption_levels": [], "time_ranges_sec": [],
                         "visual_cues": [], "reason": reason},
            "safety_adjustment": "fail_open",
        } for item in shot.tracklets],
        "unmatched_semantic_subject_ids": [],
        "unmatched_tracklet_ids": [str(item.get("tracklet_id")) for item in shot.tracklets],
        "quality_issues": ["dry_run" if dry_run else "alignment_failed"],
    }


async def align_one(client: Any, semaphore: asyncio.Semaphore, shot: ShotInput,
                    context: dict[str, Any], base_prompt: str,
                    args: argparse.Namespace,
                    rate_gate: Optional[AsyncRateGate] = None) -> dict:
    scope_id = semantic_scope_id(context)
    request_prompt, package = build_request_prompt(base_prompt, shot, context)
    if args.save_requests or args.dry_run:
        _write_json(args.output / "requests" / f"{shot.shot_id}.json", {
            "schema_version": SCHEMA_VERSION,
            "shot_id": shot.shot_id,
            "overlay_path": str(shot.overlay_path),
            "prompt_file": str(args.prompt_file),
            "input_package": package,
        })
    if args.dry_run:
        result = fail_open_alignment(shot, "dry_run_no_model_call", dry_run=True)
        result["semantic_scope_id"] = scope_id
        return result
    if not context:
        result = fail_open_alignment(shot, "missing_caption_context")
        result["semantic_scope_id"] = scope_id
        return result
    if not shot.overlay_path.is_file():
        result = fail_open_alignment(shot, f"missing_tracklet_overlay:{shot.overlay_path}")
        result["semantic_scope_id"] = scope_id
        return result

    formats = [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
    allowed_ids = allowed_subject_ids(context)
    authoritative_tiers = subject_importance(context)
    last_error = "unknown_error"
    last_response = ""
    started = time.perf_counter()
    async with semaphore:
        if rate_gate is not None:
            await rate_gate.wait()
        encoded = await asyncio.to_thread(encode_video_b64, shot.overlay_path)
        for _attempt in range(args.retries):
            # Greedy-low temperature reproduces degenerate repetition loops and
            # the same invalid assignment on every retry; escalate to escape.
            attempt_temperature = min(1.0, args.temperature + 0.2 * _attempt)
            for request_format in formats:
                try:
                    extra_body = ({"chat_template_kwargs": {"enable_thinking": True}}
                                  if args.thinking else {})
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": build_video_content(
                                request_prompt, encoded, request_format)}],
                            temperature=attempt_temperature,
                            max_tokens=args.max_completion_tokens,
                            extra_body=extra_body,
                        ), timeout=args.timeout)
                    last_response = extract_response_text(response)
                    parsed, parse_error = parse_json_response(last_response)
                    if parse_error is not None or parsed is None:
                        raise ValueError(parse_error or "response is not JSON")
                    result = normalize_alignment(parsed, shot, allowed_ids,
                                                 args.reject_confidence,
                                                 authoritative_tiers)
                    result.update({
                        "semantic_scope_id": scope_id,
                        "model": args.model,
                        "request_format": request_format,
                        "response_seconds": round(time.perf_counter() - started, 3),
                    })
                    return result
                except Exception as exc:  # noqa: BLE001 - retry boundary
                    last_error = f"{request_format}: {type(exc).__name__}: {exc}"
            if _attempt + 1 < args.retries:
                delay = min(2 ** (_attempt + 1), 16)
                LOGGER.warning("%s alignment attempt %d failed; retrying in %ds: %s",
                               shot.shot_id, _attempt + 1, delay, last_error)
                await asyncio.sleep(delay)
    if last_response:
        raw_path = args.output / "raw_responses" / f"{shot.shot_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(last_response, encoding="utf-8")
    result = fail_open_alignment(shot, last_error)
    result["semantic_scope_id"] = scope_id
    result["response_seconds"] = round(time.perf_counter() - started, 3)
    return result


def selected_ids(result: dict, policy: str, reject_confidence: float) -> set[str]:
    selected: set[str] = set()
    for row in result.get("tracklet_alignments", []):
        decision = row.get("decision")
        confidence = _finite_confidence(row.get("confidence"))
        tracklet_id = str(row.get("tracklet_id"))
        if policy in KEEP_BY_POLICY:
            if decision in KEEP_BY_POLICY[policy]:
                selected.add(tracklet_id)
        elif policy == "safe":
            rejected = decision in {"reject_background", "reject_non_person"} \
                and confidence >= reject_confidence
            if not rejected:
                selected.add(tracklet_id)
        else:
            raise ValueError(f"unknown selection policy: {policy}")
    return selected


def _semantic_identity_decisions(results: Sequence[dict], selected: set[str],
                                 min_confidence: float) -> list[dict]:
    """Build conservative cross-shot identity hints for Stage 3.

    These are deliberately hints rather than unconditional merge edges:
    ``visual_character_registry_builder`` consumes them only for pairs whose
    visual evidence already produced a POSSIBLE_LINK.  Unknown IDs, review or
    rejected rows, same-shot pairs, and IDs from different global scopes are
    never emitted.
    """
    eligible_decisions = {"keep_primary", "keep_supporting", "keep_context"}
    by_subject: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        shot_id = str(result.get("shot_id") or "")
        scope_id = str(result.get("semantic_scope_id") or "unscoped")
        for row in result.get("tracklet_alignments", []):
            tracklet_id = str(row.get("tracklet_id") or "")
            subject_id = str(row.get("subject_id") or "")
            confidence = _finite_confidence(row.get("confidence"))
            if (tracklet_id not in selected
                    or row.get("entity_type") != "person"
                    or row.get("decision") not in eligible_decisions
                    or not SUBJECT_PATTERN.fullmatch(subject_id)
                    or confidence < min_confidence):
                continue
            by_subject.setdefault((scope_id, subject_id), []).append({
                "shot_id": shot_id,
                "tracklet_id": tracklet_id,
                "confidence": confidence,
            })

    decisions: list[dict] = []
    for (scope_id, subject_id), rows in sorted(by_subject.items()):
        rows = sorted(rows, key=lambda item: (item["shot_id"], item["tracklet_id"]))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                if left["shot_id"] == right["shot_id"]:
                    continue
                decisions.append({
                    "tracklet_a": left["tracklet_id"],
                    "tracklet_b": right["tracklet_id"],
                    "same_identity": True,
                    "confidence": min(left["confidence"], right["confidence"]),
                    "source": "mimo_semantic_subject_alignment",
                    "semantic_scope_id": scope_id,
                    "subject_id": subject_id,
                    "reason": "same_high_confidence_global_scoped_subject_id",
                })
    return decisions


def export_stage3_views(shots: Sequence[ShotInput], results: Sequence[dict],
                        output: Path, reject_confidence: float,
                        semantic_link_confidence: float,
                        feedback_marks: Optional[dict[str, dict]] = None
                        ) -> tuple[dict[str, int], dict[str, int]]:
    """Export the three Stage-3 views; optionally annotate V6 feedback shots.

    ``feedback_marks`` maps shot_id to an audit record (e.g. {"round": 1,
    "tracklets_source": ...}) for shots re-aligned from a feedback tracklet
    directory.  When omitted (the default), the exported payloads are
    byte-identical to the one-way chain.
    """
    by_shot = {str(item["shot_id"]): item for item in results}
    counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    for policy in ("primary", "narrative", "safe"):
        total = 0
        selected_for_policy: set[str] = set()
        for shot in shots:
            result = by_shot[shot.shot_id]
            keep = selected_ids(result, policy, reject_confidence)
            alignment_by_tracklet = {
                str(row.get("tracklet_id")): row
                for row in result.get("tracklet_alignments", [])
            }
            selected_tracklets = []
            for item in shot.tracklets:
                tracklet_id = str(item.get("tracklet_id"))
                if tracklet_id not in keep:
                    continue
                enriched = copy.deepcopy(item)
                enriched["semantic_alignment"] = {
                    "semantic_scope_id": result.get("semantic_scope_id", "unscoped"),
                    **copy.deepcopy(alignment_by_tracklet.get(tracklet_id, {})),
                }
                selected_tracklets.append(enriched)
            payload = {
                "schema_version": shot.signature_payload.get("schema_version", "shot-tracklets-v1"),
                "shot_id": shot.shot_id,
                "semantic_filter": {
                    "schema_version": SCHEMA_VERSION,
                    "policy": policy,
                    "source_signature": str(shot.signature_path),
                    "source_alignment": str(output / "alignments" / f"{shot.shot_id}.json"),
                    "fail_open": policy == "safe",
                },
                "tracklet_signatures": selected_tracklets,
            }
            if feedback_marks and shot.shot_id in feedback_marks:
                payload["semantic_filter"]["semantic_feedback"] = copy.deepcopy(
                    feedback_marks[shot.shot_id])
            total += len(payload["tracklet_signatures"])
            selected_for_policy.update(keep)
            _write_json(output / "stage3_inputs" / policy / "shots" / shot.shot_id
                        / "tracklet_signatures.json", payload)
        counts[policy] = total
        decisions = _semantic_identity_decisions(
            results, selected_for_policy, semantic_link_confidence)
        decision_payload = {
            "schema_version": "semantic-identity-decisions-v1",
            "policy": policy,
            "min_alignment_confidence": semantic_link_confidence,
            "application_contract": (
                "Stage 3 may promote only existing POSSIBLE_LINK edges; "
                "cannot-link constraints remain authoritative."
            ),
            "decisions": decisions,
        }
        _write_json(output / "stage3_inputs" / policy
                    / "semantic_identity_decisions.json", decision_payload)
        decision_counts[policy] = len(decisions)
    return counts, decision_counts


def build_semantic_index(results: Sequence[dict]) -> dict:
    subjects: dict[str, list[dict]] = {}
    decisions: dict[str, int] = {name: 0 for name in sorted(DECISIONS)}
    for result in results:
        for row in result.get("tracklet_alignments", []):
            decision = str(row.get("decision"))
            decisions[decision] = decisions.get(decision, 0) + 1
            subject_id = str(row.get("subject_id"))
            if SUBJECT_PATTERN.fullmatch(subject_id):
                scope_id = str(result.get("semantic_scope_id") or "unscoped")
                semantic_key = f"{scope_id}:{subject_id}"
                subjects.setdefault(semantic_key, []).append({
                    "semantic_scope_id": scope_id,
                    "subject_id": subject_id,
                    "shot_id": result.get("shot_id"),
                    "tracklet_id": row.get("tracklet_id"),
                    "importance_tier": row.get("importance_tier"),
                    "decision": decision,
                    "confidence": row.get("confidence"),
                })
    return {
        "schema_version": SCHEMA_VERSION,
        "subjects": {key: subjects[key] for key in sorted(subjects)},
        "decision_counts": decisions,
    }


async def run(args: argparse.Namespace) -> dict:
    shots = discover_shots(args.tracklets)
    if args.shot_ids:
        requested = {_shot_number(value) for value in args.shot_ids}
        shots = [shot for shot in shots if _shot_number(shot.shot_id) in requested]
    if args.max_shots is not None:
        shots = shots[:args.max_shots]
    if not shots:
        raise ValueError("no shots selected")
    contexts = load_caption_context(args.global_captions, args.shot_captions,
                                    args.segment_captions)
    base_prompt = args.prompt_file.read_text(encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for a real run") from exc
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                             timeout=args.timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_gate = AsyncRateGate(args.rate_per_sec)
    existing: dict[str, dict] = {}
    if args.skip_existing:
        for shot in shots:
            path = args.output / "alignments" / f"{shot.shot_id}.json"
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("status") == "ok":
                        existing[shot.shot_id] = value
                except (OSError, json.JSONDecodeError):
                    pass

    tasks = []
    for shot in shots:
        if shot.shot_id in existing:
            continue
        context = contexts.get(str(_shot_number(shot.shot_id)), {})
        tasks.append(asyncio.create_task(align_one(
            client, semaphore, shot, context, base_prompt, args, rate_gate)))
    completed = dict(existing)
    for task in asyncio.as_completed(tasks):
        result = await task
        completed[str(result["shot_id"])] = result
        _write_json(args.output / "alignments" / f"{result['shot_id']}.json", result)
        LOGGER.info("%s: %s", result["shot_id"], result["status"])

    results = [completed[shot.shot_id] for shot in shots]
    _write_jsonl(args.output / "alignment_results.jsonl", results)
    index = build_semantic_index(results)
    _write_json(args.output / "semantic_tracklet_index.json", index)
    view_counts, semantic_decision_counts = export_stage3_views(
        shots, results, args.output, args.reject_confidence,
        args.semantic_link_confidence)
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run" if args.dry_run else
                  ("ok" if status_counts.get("failed_open", 0) == 0 else "partial"),
        "source_tracklets": str(args.tracklets.resolve()),
        "caption_inputs": {
            "global": str(args.global_captions.resolve()) if args.global_captions else None,
            "segment": str(args.segment_captions.resolve()) if args.segment_captions else None,
            "shot": str(args.shot_captions.resolve()) if args.shot_captions else None,
        },
        "prompt_file": str(args.prompt_file.resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "rate_per_sec": args.rate_per_sec,
        "shot_count": len(shots),
        "input_tracklet_count": sum(len(shot.tracklets) for shot in shots),
        "status_counts": status_counts,
        "decision_counts": index["decision_counts"],
        "selected_tracklet_counts": view_counts,
        "semantic_identity_decision_counts": semantic_decision_counts,
        "stage3_inputs": {policy: str((args.output / "stage3_inputs" / policy).resolve())
                          for policy in ("primary", "narrative", "safe")},
        "stage3_semantic_decisions": {
            policy: str((args.output / "stage3_inputs" / policy
                         / "semantic_identity_decisions.json").resolve())
            for policy in ("primary", "narrative", "safe")
        },
    }
    _write_json(args.output / "manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage1.5 MiMo semantic-role alignment and Tracklet purification")
    parser.add_argument("--tracklets", required=True, type=Path,
                        help="Stage-1 root or one tracklet_signatures.json")
    parser.add_argument("--global-captions", type=Path,
                        help="global_captions.jsonl or containing directory")
    parser.add_argument("--segment-captions", type=Path,
                        help="segment_captions.jsonl or containing directory")
    parser.add_argument("--shot-captions", type=Path,
                        help="shot_captions.jsonl or containing directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=("video_url", "input_video", "auto"),
                        default="video_url")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate-per-sec", type=float, default=0.5,
                        help="Maximum request start rate; 0 disables throttling")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-completion-tokens", type=int, default=16384,
                        help="Alignment responses for tracklet-dense shots exceed 3k tokens")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reject-confidence", type=float, default=0.85)
    parser.add_argument("--semantic-link-confidence", type=float, default=0.90,
                        help="Minimum per-Tracklet MiMo confidence for Stage-3 identity hints")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-requests", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write request packages and fail-open views without calling MiMo")
    parser.add_argument("--shot-ids", nargs="*",
                        help="Optional numeric or SHOT_x IDs for a small test")
    parser.add_argument("--max-shots", type=int)
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries and timeout must be positive")
    if args.rate_per_sec < 0:
        parser.error("--rate-per-sec must be non-negative")
    if not 0.0 <= args.reject_confidence <= 1.0:
        parser.error("--reject-confidence must be in [0, 1]")
    if not 0.0 <= args.semantic_link_confidence <= 1.0:
        parser.error("--semantic-link-confidence must be in [0, 1]")
    if args.max_shots is not None and args.max_shots < 1:
        parser.error("--max-shots must be positive")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    if not args.dry_run and not any((args.global_captions, args.segment_captions,
                                    args.shot_captions)):
        parser.error("a real run requires at least one caption input")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
