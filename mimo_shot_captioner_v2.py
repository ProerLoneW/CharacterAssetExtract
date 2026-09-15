#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 shot captioner with deterministic key-object proposal validation.

The network/orchestration code remains in :mod:`mimo_shot_captioner`.  This
version changes the default prompt and validates ``key_object_candidates`` so
that Stage1.5 and the future object grounding pipeline receive a stable schema.

Object rows produced here are proposals only.  ``proposal_status`` is forced to
``unverified``; this module never creates global ``OBJ`` identities.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import mimo_shot_captioner as v1
from mimo_caption_client import (
    build_video_content,
    encode_video_b64,
    extract_text,
    load_dotenv,
    parse_json_response,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "shot_caption_prompt_en_semantic_roles_objects_v4.md"
LOGGER = logging.getLogger("mimo_shot_captioner_v2")
V1_LOAD_GLOBAL_SUBJECT_CONTEXT = v1.load_global_subject_context

OBJECT_PRIORITIES = {"must", "high", "optional", "ignore"}
OBJECT_REASON_CODES = {
    "character_interaction", "narrative_prop", "recurring", "state_change",
    "closeup", "distinctive", "text_information",
}
CONFIDENCE_LABELS = {"high", "medium", "low", "unknown"}
SPECIAL_SUBJECT_IDS = {"unknown", "new_subject_candidate"}


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


def normalize_key_object_candidates(payload: dict, global_context: dict | None) -> dict:
    """Validate and canonicalise shot-local, explicitly unverified proposals."""
    raw_candidates = payload.get("key_object_candidates", [])
    if raw_candidates is None:
        raw_candidates = []
    if not isinstance(raw_candidates, list):
        raise ValueError("key_object_candidates must be a list")

    allowed_subject_ids = SPECIAL_SUBJECT_IDS | {
        str(item.get("subject_id"))
        for item in (global_context or {}).get("subject_registry", [])
        if isinstance(item, dict) and item.get("subject_id")
    }
    normalized: list[dict] = []
    for index, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"key_object_candidates[{index}] must be an object")
        canonical_name = str(raw.get("canonical_name") or "").strip()
        if not canonical_name:
            raise ValueError(f"key_object_candidates[{index}] missing canonical_name")
        priority = str(raw.get("priority") or "optional").lower()
        if priority not in OBJECT_PRIORITIES:
            raise ValueError(f"invalid object priority {priority!r}")
        confidence = str(raw.get("confidence") or "unknown").lower()
        if confidence not in CONFIDENCE_LABELS:
            confidence = "unknown"

        subject_ids = _string_list(raw.get("related_subject_ids"))
        invalid_subjects = [value for value in subject_ids if value not in allowed_subject_ids]
        if invalid_subjects:
            raise ValueError(
                f"key_object_candidates[{index}] uses unregistered subject IDs: "
                f"{invalid_subjects}"
            )

        interactions: list[dict] = []
        for interaction_index, interaction in enumerate(raw.get("interactions", []), 1):
            if not isinstance(interaction, dict):
                continue
            subject_id = str(interaction.get("subject_id") or "unknown")
            if subject_id not in allowed_subject_ids:
                raise ValueError(
                    f"key_object_candidates[{index}].interactions[{interaction_index}] "
                    f"uses unregistered subject ID {subject_id}"
                )
            action = str(interaction.get("action") or "").strip()
            if not action:
                continue
            if subject_id not in subject_ids:
                subject_ids.append(subject_id)
            interactions.append({
                "subject_id": subject_id,
                "action": action,
                "time_range_sec": _time_range(interaction.get("time_range_sec")),
            })

        phrases = _string_list(raw.get("grounding_phrases"), limit=4)
        if not phrases:
            phrases = [canonical_name]
        reason_codes = [
            value for value in _string_list(raw.get("reason_codes"))
            if value in OBJECT_REASON_CODES
        ]
        visual_attributes = raw.get("visual_attributes")
        if not isinstance(visual_attributes, dict):
            visual_attributes = {}
        possible_states = _string_list(raw.get("possible_states"), limit=8)
        observed_state = str(raw.get("observed_state") or "unknown").strip() or "unknown"

        normalized.append({
            "local_object_id": f"OC{index}",
            "canonical_name": canonical_name,
            "grounding_phrases": phrases,
            "priority": priority,
            "reason_codes": reason_codes,
            "visual_attributes": visual_attributes,
            "related_subject_ids": subject_ids,
            "interactions": interactions,
            "visible_time_range_sec": _time_range(raw.get("visible_time_range_sec")),
            "possible_states": possible_states,
            "observed_state": observed_state,
            "recurrence_hint": bool(raw.get("recurrence_hint", False)),
            "confidence": confidence,
            "proposal_status": "unverified",
        })
    payload["key_object_candidates"] = normalized
    return payload


def normalize_shot_payload(payload: dict, global_context: dict | None) -> dict:
    v1.validate_shot_subject_ids(payload, global_context)
    payload = v1.normalize_shot_description(payload)
    return normalize_key_object_candidates(payload, global_context)


def load_global_subject_context(input_path: Path | None) -> dict[int, dict]:
    """Preserve V1 person context and expose global object recurrence hints."""
    result = V1_LOAD_GLOBAL_SUBJECT_CONTEXT(input_path)
    if input_path is None:
        return result
    manifest = v1.find_global_caption_manifest(input_path)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("success") is not True:
            continue
        payload = record.get("global_caption")
        if not isinstance(payload, dict):
            continue
        nested = payload.get("global_caption")
        if isinstance(nested, dict):
            merged = {key: value for key, value in payload.items() if key != "global_caption"}
            merged.update(nested)
            payload = merged
        extras = {
            "non_person_entities": payload.get("non_person_entities", []),
            "key_object_candidates": payload.get("key_object_candidates", []),
        }
        for raw_shot_id in record.get("shot_ids", []):
            try:
                shot_id = int(raw_shot_id)
            except (TypeError, ValueError):
                continue
            result.setdefault(shot_id, {}).update(extras)
    return result


async def annotate_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    record: dict,
    prompt: str,
    args: argparse.Namespace,
    global_context: dict | None = None,
) -> dict:
    """V1-compatible request path with V2 object normalisation."""
    clip_path = Path(record["clip_path"])
    shot_id = int(record["shot_id"])
    context = (
        f"\nThis is shot {shot_id}. Its absolute source-video time range is "
        f"{record.get('time_range_sec')}. Annotate only this clip.\n"
    )
    if global_context:
        context += (
            "\nGLOBAL PERSON CONTEXT (use these stable S IDs; do not create a new ID for a matching person):\n"
            + json.dumps(global_context, ensure_ascii=False, indent=2)
            + "\n"
        )
    request_prompt = prompt.rstrip() + context
    formats = [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
    last_error = ""
    last_response = ""
    started = time.perf_counter()
    async with semaphore:
        encoded = await asyncio.to_thread(encode_video_b64, clip_path)
        for current_format in formats:
            try:
                extra_body = (
                    {"chat_template_kwargs": {"enable_thinking": args.thinking}}
                    if args.thinking else {}
                )
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": build_video_content(
                            request_prompt, encoded, current_format)}],
                        temperature=args.temperature,
                        max_tokens=args.max_completion_tokens,
                        extra_body=extra_body,
                    ),
                    timeout=args.timeout,
                )
                last_response = extract_text(response)
                parsed, parse_error = parse_json_response(last_response)
                if parse_error is None and parsed is not None:
                    parsed = normalize_shot_payload(parsed, global_context)
                    return {
                        **record,
                        "success": True,
                        "schema_version": "shot-caption-key-objects-v2",
                        "model": args.model,
                        "request_format": current_format,
                        "global_subject_ids": sorted(
                            str(item.get("subject_id"))
                            for item in (global_context or {}).get("subject_registry", [])
                            if isinstance(item, dict) and item.get("subject_id")
                        ),
                        "shot_caption": parsed,
                        "response_seconds": round(time.perf_counter() - started, 3),
                    }
                last_error = parse_error or "invalid response"
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = f"{current_format}: {type(exc).__name__}: {exc}"
    return {
        **record,
        "success": False,
        "schema_version": "shot-caption-key-objects-v2",
        "model": args.model,
        "error": last_error,
        "response_text": last_response or None,
        "response_seconds": round(time.perf_counter() - started, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate shots with semantic roles and key-object proposals (V2).")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--global-input", type=Path, default=None,
                        help="global_captions.jsonl or its output directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=["auto", "video_url", "input_video"],
                        default="video_url")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-each", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1:
        parser.error("--concurrency and --retries must be positive")
    if args.max_api_seconds is not None:
        if args.max_api_seconds <= 0:
            parser.error("--max-api-seconds must be positive")
        args.timeout = min(args.timeout, args.max_api_seconds)
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    for logger_name in ("httpx", "httpcore", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).disabled = True

    # V1's manifest/retry orchestration resolves this module-global callback at
    # runtime. Replacing it keeps one maintained orchestration implementation
    # while this executable owns the V2 request/normalisation contract.
    v1.annotate_one = annotate_one
    v1.load_global_subject_context = load_global_subject_context
    return asyncio.run(v1.async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
