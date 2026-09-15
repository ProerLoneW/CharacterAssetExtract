#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1.5 V2: person Tracklet alignment plus key-object proposal binding.

This version preserves the V1 character outputs and adds a conservative object
proposal stream.  Shot captions nominate objects from clean video; this stage
uses the person-overlay video to bind interactions to semantic S IDs and person
Tracklet IDs.  No result from this module is a verified detection or a global
OBJ identity.  GroundingDINO/SAM remain the verification boundary.

Optional V6 semantic feedback (off by default): ``--enable-semantic-feedback``
writes ``feedback_requests.json`` after the first alignment pass, listing only
shots whose unmatched semantic subjects carry shot-local visible evidence; the
Stage-1 builder can consume that file for one directed re-detection round.
``--feedback-tracklets`` then re-aligns those shots from the feedback products
and marks the results ``semantic_feedback_round=1``.  Without either flag the
pipeline is byte-identical to the one-way chain.

Three recall safeguards are available but off by default:
``--isolate-object-validation`` keeps the person alignment when only the object
proposals fail validation (fallback ``caption_only`` proposals are emitted
instead of failing the shot open), ``--retry-escalate-temperature`` restores
the V1 per-retry temperature escalation, and ``--enable-conflict-review``
re-adjudicates cannot-link-dropped Tracklet identities with one crop-based
review call per shot.  Without these flags the pipeline is byte-identical to
the plain V2 chain.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import semantic_role_tracklet_refiner as v1


LOGGER = logging.getLogger("semantic_role_tracklet_refiner_v2")
SCHEMA_VERSION = "semantic-role-tracklet-object-alignment-v2"
GROUNDING_SCHEMA_VERSION = "object-grounding-candidates-v1"
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "tracklet_semantic_object_alignment_prompt_en_v2.md"

OBJECT_PRIORITIES = {"must", "high", "optional", "ignore"}
OBJECT_STATUSES = {
    "visually_supported", "caption_supported", "caption_only", "uncertain",
}
OBJECT_SOURCES = {"shot_caption", "stage1_5_video", "both"}
OBJECT_REASON_CODES = {
    "character_interaction", "narrative_prop", "recurring", "state_change",
    "closeup", "distinctive", "text_information",
}
SPECIAL_SUBJECT_IDS = {"unknown", "new_subject_candidate"}
CONFIDENCE_SCORES = {"high": 0.9, "medium": 0.65, "low": 0.35, "unknown": 0.0}


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


def _time_range(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        return None
    return [round(start, 6), round(end, 6)]


def _object_confidence(value: Any) -> float:
    if isinstance(value, str) and value.lower() in CONFIDENCE_SCORES:
        return CONFIDENCE_SCORES[value.lower()]
    return v1._finite_confidence(value)


def _load_caption_payload_rows(path: Optional[Path], expected_name: str) -> list[dict]:
    manifest = v1._find_manifest(path, expected_name)
    return v1._load_jsonl(manifest) if manifest else []


def load_caption_context(
    global_input: Optional[Path],
    shot_input: Optional[Path],
    segment_input: Optional[Path],
) -> dict[str, dict[str, Any]]:
    """Load V1 context and retain the object fields V1 intentionally ignored."""
    contexts = v1.load_caption_context(global_input, shot_input, segment_input)

    for row in _load_caption_payload_rows(global_input, "global_captions.jsonl"):
        if not v1._successful(row):
            continue
        payload = v1._unwrap(row, "global_caption")
        extras = {
            "non_person_entities": payload.get("non_person_entities", []),
            "key_object_candidates": payload.get("key_object_candidates", []),
        }
        for raw_shot_id in row.get("shot_ids", []):
            number = v1._shot_number(raw_shot_id)
            if number is not None:
                contexts.setdefault(str(number), {}).setdefault("global", {}).update(extras)

    for row in _load_caption_payload_rows(shot_input, "shot_captions.jsonl"):
        if not v1._successful(row):
            continue
        number = v1._shot_number(row.get("shot_id"))
        if number is None:
            continue
        payload = v1._unwrap(row, "shot_caption")
        contexts.setdefault(str(number), {}).setdefault("shot", {})[
            "key_object_candidates"
        ] = payload.get("key_object_candidates", [])

    for row in _load_caption_payload_rows(segment_input, "segment_captions.jsonl"):
        if not v1._successful(row):
            continue
        payload = v1._unwrap(row, "segment_caption")
        for segment in payload.get("segments", []):
            if not isinstance(segment, dict):
                continue
            extras = segment.get("key_object_candidates", segment.get("key_objects", []))
            for raw_shot_id in segment.get("shot_ids", []):
                number = v1._shot_number(raw_shot_id)
                if number is not None:
                    contexts.setdefault(str(number), {}).setdefault("segment", {})[
                        "key_object_candidates"
                    ] = extras
    return contexts


def _caption_candidates(context: dict[str, Any]) -> list[dict]:
    rows = context.get("shot", {}).get("key_object_candidates", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _caption_candidate_map(context: dict[str, Any]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, row in enumerate(_caption_candidates(context), 1):
        local_id = str(row.get("local_object_id") or f"OC{index}").strip()
        if local_id in result:
            local_id = f"OC{index}"
        result[local_id] = row
    return result


def build_request_prompt(
    base_prompt: str,
    shot: v1.ShotInput,
    context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    inventory = v1.tracklet_inventory(shot)
    package = {
        "shot_id": shot.shot_id,
        "allowed_semantic_subject_ids": v1.allowed_subject_ids(context),
        "authoritative_subject_importance": v1.subject_importance(context),
        "caption_evidence": context,
        "tracklet_inventory": inventory,
        "object_contract": {
            "stage": "proposal_and_person_binding_only",
            "formal_verification_required": ["GroundingDINO", "instance_matcher", "SAM3.1"],
            "forbidden_identity_prefix": "OBJ",
        },
    }
    prompt = base_prompt.rstrip() + "\n\nINPUT PACKAGE\n" + json.dumps(
        package, ensure_ascii=False, indent=2)
    return prompt, package


def _fallback_proposal(
    local_id: str,
    raw: dict,
    shot: v1.ShotInput,
    allowed_subjects: set[str],
) -> dict:
    name = str(raw.get("canonical_name") or "unknown object").strip()
    phrases = _string_list(raw.get("grounding_phrases"), limit=4) or [name]
    priority = str(raw.get("priority") or "optional").lower()
    if priority not in OBJECT_PRIORITIES:
        priority = "optional"
    related_subjects = [
        value for value in _string_list(raw.get("related_subject_ids"))
        if value in allowed_subjects
    ]
    interactions = []
    for item in raw.get("interactions", []):
        if not isinstance(item, dict):
            continue
        subject_id = str(item.get("subject_id") or "unknown")
        action = str(item.get("action") or "").strip()
        if subject_id not in allowed_subjects or not action:
            continue
        interactions.append({
            "subject_id": subject_id,
            "tracklet_id": None,
            "action": action,
            "time_range_sec": _time_range(item.get("time_range_sec")),
        })
    attributes = raw.get("visual_attributes")
    return {
        "candidate_id": "",
        "source": "shot_caption",
        "source_candidate_ids": [local_id],
        "canonical_name": name,
        "grounding_phrases": phrases,
        "priority": priority,
        "reason_codes": [
            value for value in _string_list(raw.get("reason_codes"))
            if value in OBJECT_REASON_CODES
        ],
        "related_semantic_subject_ids": related_subjects,
        "related_tracklet_ids": [],
        "interactions": interactions,
        "visual_attributes": attributes if isinstance(attributes, dict) else {},
        "possible_states": _string_list(raw.get("possible_states"), limit=8),
        "observed_state": str(raw.get("observed_state") or "unknown"),
        "visible_time_range_sec": _time_range(raw.get("visible_time_range_sec")),
        "expected_shots": [shot.shot_id],
        "recurring": bool(raw.get("recurrence_hint", False)),
        "stateful": bool(raw.get("possible_states")),
        "proposal_status": "caption_only",
        "confidence": _object_confidence(raw.get("confidence")),
        "evidence_reason": "preserved_unreviewed_shot_caption_candidate",
    }


def normalize_object_proposals(
    payload: dict,
    shot: v1.ShotInput,
    context: dict[str, Any],
    tracklet_alignments: Sequence[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """Validate model bindings and fail open for unaccounted caption proposals."""
    raw_proposals = payload.get("object_proposals", [])
    if raw_proposals is None:
        raw_proposals = []
    if not isinstance(raw_proposals, list):
        raise ValueError("object_proposals must be a list")

    caption_by_id = _caption_candidate_map(context)
    expected_tracklets = {
        str(item.get("tracklet_id")) for item in shot.tracklets
    }
    allowed_subjects = set(v1.allowed_subject_ids(context)) | SPECIAL_SUBJECT_IDS
    alignment_by_tracklet = {
        str(row.get("tracklet_id")): str(row.get("subject_id") or "unknown")
        for row in tracklet_alignments
    }
    entity_by_tracklet = {
        str(row.get("tracklet_id")): str(row.get("entity_type") or "uncertain")
        for row in tracklet_alignments
    }

    rejected = _string_list(payload.get("rejected_object_candidate_ids"))
    unknown_rejected = [value for value in rejected if value not in caption_by_id]
    if unknown_rejected:
        raise ValueError(f"unknown rejected object candidate IDs: {unknown_rejected}")
    rejected_set = set(rejected)
    used_source_ids: set[str] = set()
    normalized: list[dict] = []
    quality_issues: list[str] = []

    for index, raw in enumerate(raw_proposals, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"object_proposals[{index}] must be an object")
        source_ids = _string_list(raw.get("source_candidate_ids"))
        unknown_source_ids = [value for value in source_ids if value not in caption_by_id]
        if unknown_source_ids:
            raise ValueError(f"object_proposals[{index}] has unknown source IDs: {unknown_source_ids}")
        if rejected_set.intersection(source_ids):
            raise ValueError(f"object candidate cannot be both accepted and rejected: {source_ids}")
        used_source_ids.update(source_ids)
        seed = caption_by_id[source_ids[0]] if source_ids else {}

        canonical_name = str(
            raw.get("canonical_name") or seed.get("canonical_name") or ""
        ).strip()
        if not canonical_name:
            raise ValueError(f"object_proposals[{index}] missing canonical_name")
        phrases = _string_list(raw.get("grounding_phrases"), limit=4)
        if not phrases:
            phrases = _string_list(seed.get("grounding_phrases"), limit=4) or [canonical_name]
        priority = str(raw.get("priority") or seed.get("priority") or "optional").lower()
        if priority not in OBJECT_PRIORITIES:
            raise ValueError(f"invalid object priority {priority!r}")
        status = str(raw.get("proposal_status") or "uncertain").lower()
        if status not in OBJECT_STATUSES - {"caption_only"}:
            status = "uncertain"
        source = str(raw.get("source") or ("shot_caption" if source_ids else "stage1_5_video"))
        if source not in OBJECT_SOURCES:
            source = "shot_caption" if source_ids else "stage1_5_video"

        subject_ids = _string_list(raw.get("related_semantic_subject_ids"))
        invalid_subjects = [value for value in subject_ids if value not in allowed_subjects]
        if invalid_subjects:
            raise ValueError(f"object_proposals[{index}] has invalid subject IDs: {invalid_subjects}")
        tracklet_ids = _string_list(raw.get("related_tracklet_ids"))
        invalid_tracklets = [value for value in tracklet_ids if value not in expected_tracklets]
        if invalid_tracklets:
            raise ValueError(f"object_proposals[{index}] has invalid Tracklet IDs: {invalid_tracklets}")
        person_tracklet_ids: list[str] = []
        for tracklet_id in tracklet_ids:
            if entity_by_tracklet.get(tracklet_id) != "person":
                quality_issues.append(
                    f"object_binding_removed_non_person_tracklet:{tracklet_id}")
                continue
            person_tracklet_ids.append(tracklet_id)
        tracklet_ids = person_tracklet_ids

        interactions: list[dict] = []
        for interaction_index, interaction in enumerate(raw.get("interactions", []), 1):
            if not isinstance(interaction, dict):
                continue
            subject_id = str(interaction.get("subject_id") or "unknown")
            tracklet_id = interaction.get("tracklet_id")
            tracklet_id = str(tracklet_id) if tracklet_id not in (None, "") else None
            action = str(interaction.get("action") or "").strip()
            if subject_id not in allowed_subjects:
                raise ValueError(
                    f"object_proposals[{index}].interactions[{interaction_index}] "
                    f"has invalid subject ID {subject_id}"
                )
            if tracklet_id is not None and tracklet_id not in expected_tracklets:
                raise ValueError(
                    f"object_proposals[{index}].interactions[{interaction_index}] "
                    f"has invalid Tracklet ID {tracklet_id}"
                )
            if tracklet_id is not None and entity_by_tracklet.get(tracklet_id) != "person":
                quality_issues.append(
                    f"object_interaction_binding_removed_non_person_tracklet:{tracklet_id}")
                tracklet_id = None
            aligned_subject = alignment_by_tracklet.get(tracklet_id or "", "unknown")
            if (
                tracklet_id is not None
                and subject_id not in SPECIAL_SUBJECT_IDS
                and aligned_subject not in SPECIAL_SUBJECT_IDS
                and subject_id != aligned_subject
            ):
                raise ValueError(
                    f"object interaction binding mismatch: {tracklet_id} is aligned to "
                    f"{aligned_subject}, not {subject_id}"
                )
            if not action:
                continue
            if subject_id not in subject_ids:
                subject_ids.append(subject_id)
            if tracklet_id is not None and tracklet_id not in tracklet_ids:
                tracklet_ids.append(tracklet_id)
            interactions.append({
                "subject_id": subject_id,
                "tracklet_id": tracklet_id,
                "action": action,
                "time_range_sec": _time_range(interaction.get("time_range_sec")),
            })

        for tracklet_id in tracklet_ids:
            aligned_subject = alignment_by_tracklet.get(tracklet_id, "unknown")
            registered_subjects = [value for value in subject_ids if value not in SPECIAL_SUBJECT_IDS]
            if (
                registered_subjects
                and aligned_subject not in SPECIAL_SUBJECT_IDS
                and aligned_subject not in registered_subjects
            ):
                raise ValueError(
                    f"object proposal binding mismatch: {tracklet_id} is aligned to "
                    f"{aligned_subject}, outside {registered_subjects}"
                )

        attributes = raw.get("visual_attributes", seed.get("visual_attributes", {}))
        possible_states = _string_list(
            raw.get("possible_states", seed.get("possible_states", [])), limit=8)
        reason_codes = [
            value for value in _string_list(raw.get("reason_codes", seed.get("reason_codes", [])))
            if value in OBJECT_REASON_CODES
        ]
        normalized.append({
            "candidate_id": "",
            "source": source,
            "source_candidate_ids": source_ids,
            "canonical_name": canonical_name,
            "grounding_phrases": phrases,
            "priority": priority,
            "reason_codes": reason_codes,
            "related_semantic_subject_ids": subject_ids,
            "related_tracklet_ids": tracklet_ids,
            "interactions": interactions,
            "visual_attributes": attributes if isinstance(attributes, dict) else {},
            "possible_states": possible_states,
            "observed_state": str(
                raw.get("observed_state", seed.get("observed_state", "unknown")) or "unknown"
            ),
            "visible_time_range_sec": _time_range(
                raw.get("visible_time_range_sec", seed.get("visible_time_range_sec"))
            ),
            "expected_shots": [shot.shot_id],
            "recurring": bool(raw.get("recurring", seed.get("recurrence_hint", False))),
            "stateful": bool(raw.get("stateful", possible_states)),
            "proposal_status": status,
            "confidence": _object_confidence(
                raw.get("confidence", seed.get("confidence", "unknown"))),
            "evidence_reason": str(raw.get("evidence_reason") or "").strip(),
        })

    missing = set(caption_by_id) - used_source_ids - rejected_set
    for local_id in caption_by_id:
        if local_id in missing:
            normalized.append(_fallback_proposal(
                local_id, caption_by_id[local_id], shot, allowed_subjects))
            quality_issues.append(f"caption_candidate_preserved_without_model_review:{local_id}")

    for index, row in enumerate(normalized, 1):
        row["candidate_id"] = f"{shot.shot_id}_OC{index:03d}"
    return normalized, rejected, quality_issues


def normalize_alignment(
    payload: dict,
    shot: v1.ShotInput,
    context: dict[str, Any],
    reject_confidence: float,
    *,
    isolate_object_validation: bool = False,
) -> dict:
    allowed_ids = v1.allowed_subject_ids(context)
    result = v1.normalize_alignment(
        payload,
        shot,
        allowed_ids,
        reject_confidence,
        v1.subject_importance(context),
    )
    if isolate_object_validation:
        # P1-2: an object-side validation error must not sacrifice the person
        # alignment; fall back to unreviewed caption candidates instead.
        try:
            proposals, rejected, object_issues = normalize_object_proposals(
                payload, shot, context, result["tracklet_alignments"])
        except ValueError as exc:
            allowed_subjects = set(allowed_ids) | SPECIAL_SUBJECT_IDS
            proposals = [
                _fallback_proposal(local_id, raw, shot, allowed_subjects)
                for local_id, raw in _caption_candidate_map(context).items()
            ]
            for index, row in enumerate(proposals, 1):
                row["candidate_id"] = f"{shot.shot_id}_OC{index:03d}"
            rejected = []
            object_issues = [f"object_validation_isolated:{type(exc).__name__}: {exc}"]
            result["component_status"] = {
                "person": "ok",
                "object": "isolated_failure",
                "object_error": str(exc),
            }
        else:
            result["component_status"] = {"person": "ok", "object": "ok"}
    else:
        proposals, rejected, object_issues = normalize_object_proposals(
            payload, shot, context, result["tracklet_alignments"])
    result["schema_version"] = SCHEMA_VERSION
    result["object_proposals"] = proposals
    result["rejected_object_candidate_ids"] = rejected
    result["quality_issues"] = list(result.get("quality_issues", [])) + object_issues
    return result


def fail_open_alignment(
    shot: v1.ShotInput,
    context: dict[str, Any],
    reason: str,
    *,
    dry_run: bool = False,
) -> dict:
    result = v1.fail_open_alignment(shot, reason, dry_run=dry_run)
    allowed_subjects = set(v1.allowed_subject_ids(context)) | SPECIAL_SUBJECT_IDS
    proposals = [
        _fallback_proposal(local_id, raw, shot, allowed_subjects)
        for local_id, raw in _caption_candidate_map(context).items()
    ]
    for index, row in enumerate(proposals, 1):
        row["candidate_id"] = f"{shot.shot_id}_OC{index:03d}"
    result["schema_version"] = SCHEMA_VERSION
    result["object_proposals"] = proposals
    result["rejected_object_candidate_ids"] = []
    return result


# ---------------------------------------------------------------------------
# P0-2: cannot-link conflict identity review (opt-in)
# ---------------------------------------------------------------------------
def _quality_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gallery_crop_paths(tracklet: dict, max_crops: int) -> list[Path]:
    """Best-quality existing face/body crops of one tracklet, capped in total."""
    paths: list[Path] = []
    for gallery_name in ("face_gallery", "body_gallery"):
        gallery = tracklet.get(gallery_name, [])
        if not isinstance(gallery, list):
            continue
        items = sorted(
            (item for item in gallery if isinstance(item, dict)),
            key=lambda item: _quality_score(item.get("quality_score")),
            reverse=True,
        )
        for item in items:
            raw = item.get("crop_path") or item.get("asset_path")
            if not raw:
                continue
            path = Path(str(raw))
            if path.is_file() and path not in paths:
                paths.append(path)
            if len(paths) >= max_crops:
                return paths
    return paths


def _encode_image_b64(path: Path) -> str:
    """Synchronously read one image crop and return its base64 encoding."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_review_content(
    text: str,
    tracklet_images: Sequence[tuple[str, list[str]]],
) -> list[dict]:
    """OpenAI content array: the review prompt plus labelled crop images."""
    content: list[dict] = [{"type": "text", "text": text}]
    for tracklet_id, encoded_images in tracklet_images:
        if not encoded_images:
            continue
        content.append({"type": "text", "text": f"Reference crops for tracklet {tracklet_id}:"})
        for encoded in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
    return content


def _build_conflict_review_text(
    shot: v1.ShotInput,
    groups: dict[str, dict[str, list[str]]],
    tracklet_source: dict[str, dict],
    allowed: set[str],
) -> str:
    def describe(tracklet_id: str) -> str:
        tracklet = tracklet_source.get(tracklet_id, {})
        return (
            f"{tracklet_id} (frames {tracklet.get('start_frame')}-{tracklet.get('end_frame')}, "
            f"{tracklet.get('start_time')}-{tracklet.get('end_time')}s)"
        )

    def sort_key(value: str) -> tuple[int, Any]:
        if v1.SUBJECT_PATTERN.fullmatch(value):
            return (0, int(value[1:]))
        return (1, value)

    lines = [
        f"Shot {shot.shot_id}: cannot-link identity re-review.",
        "",
        "Detection evidence shows that the tracklets inside each conflict group "
        "below co-occur in the same frames, so they cannot be the same person. "
        "An earlier alignment kept the survivor and dropped the rest of the "
        "group. Re-assign every listed tracklet to one allowed semantic subject "
        "ID, or to \"unknown\" when the evidence is insufficient. Never merge "
        "different people to remove a conflict.",
        "",
        "Allowed subject IDs: " + ", ".join(sorted(allowed, key=sort_key)),
        'Return exactly one JSON object without Markdown: {"assignments": '
        '[{"tracklet_id": "...", "subject_id": "...", "reason": "..."}]}, with '
        "one assignment per listed tracklet.",
        "",
        "Conflict groups:",
    ]
    for subject_id, group in groups.items():
        lines.append(f"- Group originally assigned to {subject_id}:")
        for tracklet_id in group["kept"]:
            lines.append(f"  - {describe(tracklet_id)} -- currently kept as {subject_id}")
        for tracklet_id in group["dropped"]:
            lines.append(
                f"  - {describe(tracklet_id)} -- unassigned after the cannot-link conflict")
    lines.append("")
    lines.append("Reference crops for each tracklet are attached below when available.")
    return "\n".join(lines)


async def _default_conflict_review_fn(
    client: Any,
    content: list[dict],
    args: argparse.Namespace,
) -> str:
    """Single low-temperature review call; at most one per shot, outside retries."""
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=2048,
        ),
        timeout=args.timeout,
    )
    return v1.extract_response_text(response)


async def maybe_review_conflicts(
    client: Any,
    result: dict,
    payload: dict,
    shot: v1.ShotInput,
    context: dict[str, Any],
    args: argparse.Namespace,
    review_fn: Any = None,
) -> dict:
    """Re-adjudicate cannot-link-dropped identities with one extra VLM call.

    Conservative by construction: the review may only re-assign dropped rows,
    the cannot-link arbitration is re-run afterwards, and any failure leaves
    the original arbitration untouched apart from an audit record.
    """
    alignments = result.get("tracklet_alignments", [])
    dropped_ids = [
        str(row.get("tracklet_id")) for row in alignments
        if row.get("safety_adjustment") == "cannot_link_conflict_dropped"
    ]
    if not dropped_ids:
        return result
    payload_by_tracklet = {
        str(row.get("tracklet_id")): row
        for row in payload.get("tracklet_alignments", [])
        if isinstance(row, dict)
    }
    groups: dict[str, dict[str, list[str]]] = {}
    for tracklet_id in dropped_ids:
        original = str(payload_by_tracklet.get(tracklet_id, {}).get("subject_id") or "unknown")
        groups.setdefault(original, {"kept": [], "dropped": []})["dropped"].append(tracklet_id)
    for row in alignments:
        subject_id = str(row.get("subject_id") or "")
        tracklet_id = str(row.get("tracklet_id"))
        if (subject_id in groups and v1.SUBJECT_PATTERN.fullmatch(subject_id)
                and tracklet_id not in groups[subject_id]["dropped"]):
            groups[subject_id]["kept"].append(tracklet_id)
    audits = [{
        "subject_id": subject_id,
        "kept_tracklet_ids": list(group["kept"]),
        "dropped_tracklet_ids": list(group["dropped"]),
        "review_status": "failed",
        "assignments": [],
        "error": None,
    } for subject_id, group in groups.items()]
    try:
        tracklet_source = {str(item.get("tracklet_id")): item for item in shot.tracklets}
        review_ids = [
            tracklet_id
            for group in groups.values()
            for tracklet_id in group["kept"] + group["dropped"]
        ]
        allowed = set(v1.allowed_subject_ids(context)) | {"unknown"}
        tracklet_images = []
        for tracklet_id in review_ids:
            crop_paths = _gallery_crop_paths(
                tracklet_source.get(tracklet_id, {}), args.conflict_review_max_crops)
            encoded_images = [
                await asyncio.to_thread(_encode_image_b64, path) for path in crop_paths
            ]
            tracklet_images.append((tracklet_id, encoded_images))
        content = _build_review_content(
            _build_conflict_review_text(shot, groups, tracklet_source, allowed),
            tracklet_images,
        )
        review_call = review_fn if review_fn is not None else _default_conflict_review_fn
        raw_text = await review_call(client, content, args)
        reviewed, parse_error = v1.parse_json_response(raw_text)
        if parse_error is not None or reviewed is None:
            raise ValueError(parse_error or "conflict review response is not JSON")
        raw_assignments = reviewed.get("assignments", [])
        if not isinstance(raw_assignments, list):
            raise ValueError("conflict review assignments must be a list")
        issues: list[str] = []
        assignments: dict[str, dict] = {}
        for item in raw_assignments:
            if not isinstance(item, dict):
                issues.append("conflict_review_ignored_assignment:not_an_object")
                continue
            tracklet_id = str(item.get("tracklet_id") or "")
            subject_id = str(item.get("subject_id") or "")
            if tracklet_id not in review_ids:
                issues.append(f"conflict_review_ignored_assignment:unknown_tracklet:{tracklet_id}")
                continue
            if subject_id not in allowed:
                issues.append(f"conflict_review_ignored_assignment:invalid_subject:{subject_id}")
                continue
            assignments[tracklet_id] = {
                "tracklet_id": tracklet_id,
                "subject_id": subject_id,
                "reason": str(item.get("reason") or ""),
            }

        # Work on a copy and commit only after every step succeeded, so a
        # review failure can never corrupt the original arbitration.
        work = copy.deepcopy(alignments)
        work_by_tracklet = {str(row.get("tracklet_id")): row for row in work}
        for tracklet_id, assignment in assignments.items():
            row = work_by_tracklet[tracklet_id]
            if row.get("safety_adjustment") != "cannot_link_conflict_dropped":
                continue
            if assignment["subject_id"] == "unknown":
                row["safety_adjustment"] = "cannot_link_conflict_review_unknown"
                continue
            # The disputed part was the identity; the model's per-tracklet
            # observations (decision/confidence/importance/evidence) stand.
            original = payload_by_tracklet.get(tracklet_id, {})
            for field in ("decision", "confidence", "importance_tier", "evidence"):
                if field in original:
                    row[field] = copy.deepcopy(original[field])
            row["subject_id"] = assignment["subject_id"]
            row["safety_adjustment"] = "cannot_link_conflict_reviewed"

        # Re-run the cannot-link arbitration so a reviewed identity can never
        # violate detection co-occurrence (mirrors v1.normalize_alignment).
        subject_owner: dict[str, list[dict]] = {}
        for row in work:
            if v1.SUBJECT_PATTERN.fullmatch(str(row.get("subject_id") or "")):
                subject_owner.setdefault(row["subject_id"], []).append(row)
        still_conflicting: set[str] = set()
        for assigned in subject_owner.values():
            kept_rows: list[dict] = []
            for row in assigned:
                source = tracklet_source.get(str(row.get("tracklet_id")), {})
                row_cannot = set(source.get("cannot_links", source.get("cooccurring_tracklets", [])))
                if any(other["tracklet_id"] in row_cannot for other in kept_rows):
                    still_conflicting.add(str(row["tracklet_id"]))
                    continue
                kept_rows.append(row)
        for row in work:
            if str(row.get("tracklet_id")) in still_conflicting:
                row["subject_id"] = "unknown"
                row["decision"] = "review"
                row["safety_adjustment"] = "cannot_link_conflict_review_still_conflicting"

        # Rebuild the derived unmatched views with the same construction as v1.
        allowed_with_special = allowed | {"new_subject_candidate"}
        unmatched_subjects = sorted({
            str(value) for value in payload.get("unmatched_semantic_subject_ids", [])
            if str(value) in allowed_with_special
        })
        unmatched_tracklets = [
            str(row.get("tracklet_id")) for row in work if row.get("decision") == "review"
        ]
        for audit in audits:
            group_ids = set(audit["kept_tracklet_ids"]) | set(audit["dropped_tracklet_ids"])
            audit["assignments"] = [
                assignment for tracklet_id, assignment in assignments.items()
                if tracklet_id in group_ids
            ]
            adjustments = {
                str(work_by_tracklet[tracklet_id].get("safety_adjustment"))
                for tracklet_id in audit["dropped_tracklet_ids"]
            }
            audit["review_status"] = (
                "still_conflicting"
                if "cannot_link_conflict_review_still_conflicting" in adjustments
                else "applied"
            )

        result["tracklet_alignments"] = work
        result["unmatched_semantic_subject_ids"] = unmatched_subjects
        result["unmatched_tracklet_ids"] = unmatched_tracklets
        if issues:
            result["quality_issues"] = list(result.get("quality_issues", [])) + issues
    except Exception as exc:  # noqa: BLE001 - review must never break the shot
        for audit in audits:
            audit["review_status"] = "failed"
            audit["error"] = f"{type(exc).__name__}: {exc}"
    result["conflict_reviews"] = audits
    return result


async def align_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    shot: v1.ShotInput,
    context: dict[str, Any],
    base_prompt: str,
    args: argparse.Namespace,
    rate_gate: Optional[v1.AsyncRateGate] = None,
) -> dict:
    scope_id = v1.semantic_scope_id(context)
    request_prompt, package = build_request_prompt(base_prompt, shot, context)
    if args.save_requests or args.dry_run:
        v1._write_json(args.output / "requests" / f"{shot.shot_id}.json", {
            "schema_version": SCHEMA_VERSION,
            "shot_id": shot.shot_id,
            "overlay_path": str(shot.overlay_path),
            "prompt_file": str(args.prompt_file),
            "input_package": package,
        })
    if args.dry_run:
        result = fail_open_alignment(shot, context, "dry_run_no_model_call", dry_run=True)
        result["semantic_scope_id"] = scope_id
        return result
    if not context:
        result = fail_open_alignment(shot, context, "missing_caption_context")
        result["semantic_scope_id"] = scope_id
        return result
    if not shot.overlay_path.is_file():
        result = fail_open_alignment(
            shot, context, f"missing_tracklet_overlay:{shot.overlay_path}")
        result["semantic_scope_id"] = scope_id
        return result

    formats = [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
    last_error = "unknown_error"
    last_response = ""
    attempts: list[dict] = []
    backoff_seconds = 0.0
    started = time.perf_counter()
    async with semaphore:
        if rate_gate is not None:
            await rate_gate.wait()
        queue_wait_seconds = time.perf_counter() - started
        encoded = await asyncio.to_thread(v1.encode_video_b64, shot.overlay_path)
        for attempt in range(args.retries):
            attempt_temperature = args.temperature
            if args.retry_escalate_temperature:
                # Greedy-low temperature reproduces the same invalid assignment
                # on every retry; escalate to escape (V1 semantics, opt-in).
                attempt_temperature = min(1.0, args.temperature + 0.2 * attempt)
            for request_format in formats:
                request_started = time.perf_counter()
                attempt_response = ""
                try:
                    if rate_gate is not None:
                        await rate_gate.wait()
                    extra_body = (
                        {"chat_template_kwargs": {"enable_thinking": True}}
                        if args.thinking else {}
                    )
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": v1.build_video_content(
                                request_prompt, encoded, request_format)}],
                            temperature=attempt_temperature,
                            max_tokens=args.max_completion_tokens,
                            extra_body=extra_body,
                        ),
                        timeout=args.timeout,
                    )
                    attempt_response = v1.extract_response_text(response)
                    last_response = attempt_response
                    parsed, parse_error = v1.parse_json_response(attempt_response)
                    if parse_error is not None or parsed is None:
                        raise ValueError(parse_error or "response is not JSON")
                    result = normalize_alignment(
                        parsed, shot, context, args.reject_confidence,
                        isolate_object_validation=args.isolate_object_validation)
                    if args.enable_conflict_review:
                        result = await maybe_review_conflicts(
                            client, result, parsed, shot, context, args)
                    attempts.append({
                        "attempt": attempt + 1,
                        "request_format": request_format,
                        "status": "ok",
                        "error": None,
                        "request_seconds": round(
                            time.perf_counter() - request_started, 3),
                    })
                    result.update({
                        "semantic_scope_id": scope_id,
                        "model": args.model,
                        "request_format": request_format,
                        "response_seconds": round(time.perf_counter() - started, 3),
                        "queue_wait_seconds": round(queue_wait_seconds, 3),
                        "backoff_seconds": round(backoff_seconds, 3),
                        "attempts": attempts,
                    })
                    return result
                except Exception as exc:  # noqa: BLE001 - retry boundary
                    last_error = f"{request_format}: {type(exc).__name__}: {exc}"
                    attempts.append({
                        "attempt": attempt + 1,
                        "request_format": request_format,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "request_seconds": round(
                            time.perf_counter() - request_started, 3),
                    })
                    if attempt_response:
                        raw_attempt_path = (
                            args.output / "raw_responses"
                            / f"{shot.shot_id}.attempt{attempt + 1}.{request_format}.txt"
                        )
                        raw_attempt_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_attempt_path.write_text(attempt_response, encoding="utf-8")
            if attempt + 1 < args.retries:
                delay = min(2 ** (attempt + 1), 16)
                backoff_seconds += delay
                LOGGER.warning(
                    "%s alignment attempt %d failed; retrying in %ds: %s",
                    shot.shot_id, attempt + 1, delay, last_error)
                await asyncio.sleep(delay)
    if last_response:
        raw_path = args.output / "raw_responses" / f"{shot.shot_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(last_response, encoding="utf-8")
    result = fail_open_alignment(shot, context, last_error)
    result["semantic_scope_id"] = scope_id
    result["response_seconds"] = round(time.perf_counter() - started, 3)
    result["queue_wait_seconds"] = round(queue_wait_seconds, 3)
    result["backoff_seconds"] = round(backoff_seconds, 3)
    result["attempts"] = attempts
    return result


def _flatten_object_proposals(results: Sequence[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        for proposal in result.get("object_proposals", []):
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "shot_id": result.get("shot_id"),
                "semantic_scope_id": result.get("semantic_scope_id", "unscoped"),
                **copy.deepcopy(proposal),
            })
    return rows


def build_grounding_queue(results: Sequence[dict], min_confidence: float) -> list[dict]:
    """Emit only V1-scope MUST/HIGH proposals; still explicitly unverified."""
    queue: list[dict] = []
    for row in _flatten_object_proposals(results):
        if row.get("priority") not in {"must", "high"}:
            continue
        if row.get("proposal_status") == "uncertain":
            continue
        if _object_confidence(row.get("confidence")) < min_confidence:
            continue
        queue.append({
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "candidate_id": row["candidate_id"],
            "shot_id": row["shot_id"],
            "semantic_scope_id": row["semantic_scope_id"],
            "canonical_name": row["canonical_name"],
            "grounding_phrases": row["grounding_phrases"],
            "priority": row["priority"],
            "related_semantic_subject_ids": row["related_semantic_subject_ids"],
            "related_tracklet_ids": row["related_tracklet_ids"],
            "interactions": row["interactions"],
            "visual_attributes": row["visual_attributes"],
            "possible_states": row["possible_states"],
            "observed_state": row["observed_state"],
            "expected_shots": row["expected_shots"],
            "recurring": row["recurring"],
            "stateful": row["stateful"],
            "proposal_confidence": row["confidence"],
            "proposal_status": row["proposal_status"],
            "verification_status": "unverified",
        })
    return queue


FEEDBACK_REQUESTS_SCHEMA = "semantic-feedback-requests-v1"
FEEDBACK_ROUND = 1


def _subject_evidence_windows(
    context: dict[str, Any],
    subject_id: str,
    shot_number: Optional[int],
) -> list[list[float]]:
    """Shot-local visible-evidence windows for one unmatched subject.

    Only time ranges expressed in shot-local seconds qualify: the shot-level
    subjects / role observations / visual events, plus segment-level role
    observations whose per-shot ``time_ranges_sec`` pair with this shot.
    Global-level timelines use source-video time and are deliberately not
    used, because they cannot be mapped into a shot without offsets.
    """
    windows: list[list[float]] = []
    seen: set[tuple[float, float]] = set()

    def add(value: Any) -> None:
        window = _time_range(value)
        if window is not None and tuple(window) not in seen:
            seen.add(tuple(window))
            windows.append(window)

    shot_context = context.get("shot", {})
    if isinstance(shot_context, dict):
        for row in shot_context.get("subjects", []) or []:
            if isinstance(row, dict) and str(row.get("id")) == subject_id:
                add(row.get("visible_time_range_sec"))
        for row in shot_context.get("semantic_role_observations", []) or []:
            if isinstance(row, dict) and str(row.get("subject_id")) == subject_id:
                add(row.get("time_range_sec"))
        for row in shot_context.get("visual_events", []) or []:
            if not isinstance(row, dict):
                continue
            if subject_id in [str(item) for item in row.get("subject_ids", []) or []]:
                add(row.get("time_range_sec"))
    segment_context = context.get("segment", {})
    if isinstance(segment_context, dict) and shot_number is not None:
        for row in segment_context.get("semantic_role_observations", []) or []:
            if not isinstance(row, dict) or str(row.get("subject_id")) != subject_id:
                continue
            shot_ids = row.get("shot_ids", [])
            ranges = row.get("time_ranges_sec", [])
            if not isinstance(shot_ids, list) or not isinstance(ranges, list):
                continue
            for raw_shot_id, raw_range in zip(shot_ids, ranges):
                if v1._shot_number(raw_shot_id) == shot_number:
                    add(raw_range)
    return windows


def build_feedback_requests(
    shots: Sequence[v1.ShotInput],
    results: Sequence[dict],
    contexts: dict[str, dict[str, Any]],
    source_tracklets: Path,
    source_output: Path,
) -> dict:
    """Collect one-shot re-detection requests for unmatched semantic subjects.

    A request is emitted only for a shot whose alignment produced
    ``unmatched_semantic_subject_ids`` AND where at least one unmatched
    subject has shot-local visible evidence in the caption context.  Subjects
    without in-shot evidence are reported but never requested, so a bare
    caption claim cannot trigger detection.
    """
    by_shot = {str(result.get("shot_id")): result for result in results}
    requests: list[dict] = []
    for shot in shots:
        result = by_shot.get(shot.shot_id)
        if not result:
            continue
        unmatched = [
            value for value in result.get("unmatched_semantic_subject_ids", [])
            if v1.SUBJECT_PATTERN.fullmatch(str(value))
        ]
        if not unmatched:
            continue
        context = contexts.get(str(v1._shot_number(shot.shot_id)), {})
        shot_number = v1._shot_number(shot.shot_id)
        windows_by_subject: dict[str, list[list[float]]] = {}
        for subject_id in unmatched:
            windows = _subject_evidence_windows(context, subject_id, shot_number)
            if windows:
                windows_by_subject[subject_id] = windows
        if not windows_by_subject:
            continue
        all_windows = sorted({tuple(window) for windows in windows_by_subject.values()
                              for window in windows})
        requests.append({
            "shot_id": shot.shot_id,
            "requested_subjects": sorted(windows_by_subject,
                                         key=lambda value: int(value[1:])),
            "subjects_without_evidence": sorted(
                set(unmatched) - set(windows_by_subject),
                key=lambda value: int(value[1:])),
            "evidence_windows_sec": windows_by_subject,
            "all_evidence_windows_sec": [list(window) for window in all_windows],
            "source_tracklets": str(source_tracklets),
            "source_signature": str(shot.signature_path),
            "source_overlay": str(shot.overlay_path),
            "source_alignment": str(source_output / "alignments" / f"{shot.shot_id}.json"),
        })
    return {
        "schema_version": FEEDBACK_REQUESTS_SCHEMA,
        "round": FEEDBACK_ROUND,
        "source_tracklets": str(source_tracklets),
        "source_alignment_output": str(source_output),
        "request_count": len(requests),
        "requests": requests,
    }


def _apply_feedback_tracklets(
    shots: Sequence[v1.ShotInput],
    feedback_tracklets: Optional[Path],
) -> tuple[list[v1.ShotInput], dict[str, v1.ShotInput]]:
    """Swap shots re-built by a feedback run to their feedback products."""
    if feedback_tracklets is None:
        return list(shots), {}
    feedback_shots = {shot.shot_id: shot for shot in v1.discover_shots(feedback_tracklets)}
    replaced = [feedback_shots.get(shot.shot_id, shot) for shot in shots]
    applied = {shot.shot_id: feedback_shots[shot.shot_id]
               for shot in shots if shot.shot_id in feedback_shots}
    return replaced, applied


async def run(args: argparse.Namespace) -> dict:
    shots = v1.discover_shots(args.tracklets)
    if args.shot_ids:
        requested = {v1._shot_number(value) for value in args.shot_ids}
        shots = [shot for shot in shots if v1._shot_number(shot.shot_id) in requested]
    if args.max_shots is not None:
        shots = shots[:args.max_shots]
    if not shots:
        raise ValueError("no shots selected")

    # V6 second round: shots re-built by the builder's feedback mode are
    # re-aligned from the feedback overlay/tracklets instead of the original
    # products.  The one-way chain is untouched when the flag is absent.
    shots, feedback_shots = _apply_feedback_tracklets(shots, args.feedback_tracklets)
    feedback_shot_ids = set(feedback_shots)

    contexts = load_caption_context(
        args.global_captions, args.shot_captions, args.segment_captions)
    base_prompt = args.prompt_file.read_text(encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for a real run") from exc
        client = AsyncOpenAI(
            base_url=args.base_url, api_key=args.api_key,
            timeout=args.timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_gate = v1.AsyncRateGate(args.rate_per_sec)
    existing: dict[str, dict] = {}
    if args.skip_existing:
        for shot in shots:
            path = args.output / "alignments" / f"{shot.shot_id}.json"
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("status") != "ok" or value.get("schema_version") != SCHEMA_VERSION:
                        continue
                    # Isolated object failures are reusable (the person part is
                    # ok); a failed person part must be recomputed.
                    component_status = value.get("component_status")
                    if (isinstance(component_status, dict)
                            and component_status.get("person") != "ok"):
                        continue
                    # A cached pre-feedback alignment was computed against the
                    # old overlay; it must not satisfy a feedback-round shot.
                    if (shot.shot_id in feedback_shot_ids
                            and value.get("semantic_feedback_round") != FEEDBACK_ROUND):
                        continue
                    existing[shot.shot_id] = value
                except (OSError, json.JSONDecodeError):
                    pass

    tasks = []
    for shot in shots:
        if shot.shot_id in existing:
            continue
        context = contexts.get(str(v1._shot_number(shot.shot_id)), {})
        tasks.append(asyncio.create_task(align_one(
            client, semaphore, shot, context, base_prompt, args, rate_gate)))
    completed = dict(existing)
    for task in asyncio.as_completed(tasks):
        result = await task
        if result["shot_id"] in feedback_shot_ids:
            result["semantic_feedback_round"] = FEEDBACK_ROUND
            result["semantic_feedback_tracklets"] = str(
                feedback_shots[result["shot_id"]].signature_path)
        completed[str(result["shot_id"])] = result
        v1._write_json(args.output / "alignments" / f"{result['shot_id']}.json", result)
        LOGGER.info("%s: %s", result["shot_id"], result["status"])

    results = [completed[shot.shot_id] for shot in shots]
    v1._write_jsonl(args.output / "alignment_results.jsonl", results)
    index = v1.build_semantic_index(results)
    index["schema_version"] = SCHEMA_VERSION
    v1._write_json(args.output / "semantic_tracklet_index.json", index)
    feedback_marks = {
        shot_id: {"round": FEEDBACK_ROUND,
                  "tracklets_source": str(feedback_shots[shot_id].signature_path)}
        for shot_id in feedback_shot_ids
    }
    view_counts, semantic_decision_counts = v1.export_stage3_views(
        shots,
        results,
        args.output,
        args.reject_confidence,
        args.semantic_link_confidence,
        feedback_marks=feedback_marks or None,
    )

    object_rows = _flatten_object_proposals(results)
    grounding_queue = build_grounding_queue(results, args.object_proposal_confidence)
    v1._write_jsonl(args.output / "object_proposals.jsonl", object_rows)
    v1._write_jsonl(args.output / "object_grounding_candidates.jsonl", grounding_queue)

    # V6 first round: after the one-way alignment, collect directed
    # re-detection requests for unmatched subjects that carry shot-local
    # visible evidence.  A run consuming --feedback-tracklets never emits new
    # requests: feedback is limited to a single round.
    feedback_summary: Optional[dict] = None
    if args.feedback_tracklets:
        still_unmatched = {
            str(result.get("shot_id")): result.get("unmatched_semantic_subject_ids", [])
            for result in results
            if result.get("shot_id") in feedback_shot_ids
            and result.get("unmatched_semantic_subject_ids")
        }
        feedback_summary = {
            "round": FEEDBACK_ROUND,
            "mode": "consumed",
            "feedback_tracklets": str(args.feedback_tracklets.expanduser().resolve()),
            "applied_shot_ids": sorted(feedback_shot_ids),
            "further_requests_emitted": False,
            "still_unmatched_semantic_subject_ids": still_unmatched,
        }
    elif args.enable_semantic_feedback:
        requests_payload = build_feedback_requests(
            shots, results, contexts, args.tracklets.expanduser().resolve(),
            args.output.expanduser().resolve())
        requests_path = args.output / "feedback_requests.json"
        v1._write_json(requests_path, requests_payload)
        feedback_summary = {
            "round": FEEDBACK_ROUND,
            "mode": "requests",
            "requests_path": str(requests_path.resolve()),
            "request_count": requests_payload["request_count"],
            "requested_shot_ids": [request["shot_id"]
                                   for request in requests_payload["requests"]],
        }

    status_counts: dict[str, int] = {}
    object_status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    for row in object_rows:
        status = str(row.get("proposal_status"))
        object_status_counts[status] = object_status_counts.get(status, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run" if args.dry_run else (
            "ok" if status_counts.get("failed_open", 0) == 0 else "partial"),
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
        "object_proposal_count": len(object_rows),
        "object_proposal_status_counts": object_status_counts,
        "object_grounding_candidate_count": len(grounding_queue),
        "object_proposals": str((args.output / "object_proposals.jsonl").resolve()),
        "object_grounding_candidates": str(
            (args.output / "object_grounding_candidates.jsonl").resolve()),
        "stage3_inputs": {
            policy: str((args.output / "stage3_inputs" / policy).resolve())
            for policy in ("primary", "narrative", "safe")
        },
    }
    if feedback_summary is not None:
        summary["semantic_feedback"] = feedback_summary
    v1._write_json(args.output / "manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage1.5 V2 semantic Tracklet alignment and object proposal binding")
    parser.add_argument("--tracklets", required=True, type=Path,
                        help="Stage-1 root or one tracklet_signatures.json")
    parser.add_argument("--global-captions", type=Path)
    parser.add_argument("--segment-captions", type=Path)
    parser.add_argument("--shot-captions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=v1.resolve_base_url())
    parser.add_argument("--api-key", default=v1.resolve_api_key())
    parser.add_argument("--model", default=v1.resolve_model())
    parser.add_argument("--request-format", choices=("video_url", "input_video", "auto"),
                        default="video_url")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate-per-sec", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reject-confidence", type=float, default=0.85)
    parser.add_argument("--semantic-link-confidence", type=float, default=0.90)
    parser.add_argument(
        "--object-proposal-confidence", type=float, default=0.50,
        help="Minimum proposal confidence for object_grounding_candidates.jsonl")
    parser.add_argument(
        "--isolate-object-validation", action="store_true",
        help="P1-2: keep the person alignment when only object proposals fail "
             "validation; emit fallback caption_only proposals instead of "
             "failing the whole shot open (default: off)")
    parser.add_argument(
        "--retry-escalate-temperature", action="store_true",
        help="Escalate the sampling temperature by +0.2 per retry attempt "
             "(capped at 1.0) to escape repeated invalid assignments "
             "(default: off, constant --temperature)")
    parser.add_argument(
        "--enable-conflict-review", action="store_true",
        help="P0-2: re-adjudicate cannot-link-dropped Tracklet identities with "
             "one extra crop-based review call per shot (default: off)")
    parser.add_argument(
        "--conflict-review-max-crops", type=int, default=3,
        help="Maximum face/body crop images attached per tracklet in a "
             "conflict-review request")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-requests", action="store_true")
    parser.add_argument(
        "--enable-semantic-feedback", action="store_true",
        help="V6: after alignment, write feedback_requests.json listing shots "
             "whose unmatched semantic subjects carry shot-local visible "
             "evidence; the builder can consume it for one directed "
             "re-detection round (default: off, one-way chain unchanged)")
    parser.add_argument(
        "--feedback-tracklets", type=Path,
        help="V6: builder feedback output directory from a directed "
             "re-detection round; shots present there are re-aligned from the "
             "feedback overlay/tracklets and marked semantic_feedback_round=1. "
             "Never emits further feedback requests (max one round)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shot-ids", nargs="*")
    parser.add_argument("--max-shots", type=int)
    return parser


def main() -> int:
    v1.load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries and timeout must be positive")
    if args.rate_per_sec < 0:
        parser.error("--rate-per-sec must be non-negative")
    for name in (
        "reject_confidence", "semantic_link_confidence", "object_proposal_confidence",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.max_shots is not None and args.max_shots < 1:
        parser.error("--max-shots must be positive")
    if args.conflict_review_max_crops < 1:
        parser.error("--conflict-review-max-crops must be positive")
    if args.feedback_tracklets is not None and not args.feedback_tracklets.exists():
        parser.error(f"--feedback-tracklets does not exist: {args.feedback_tracklets}")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    if not args.dry_run and not any((
        args.global_captions, args.segment_captions, args.shot_captions,
    )):
        parser.error("a real run requires at least one caption input")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
