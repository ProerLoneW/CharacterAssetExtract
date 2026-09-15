#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global-scope tracklet → character alignment (bypasses CHR clustering).

Motivation
----------
The legacy chain aligns tracklets to semantic S-IDs shot by shot, then merges
tracklets into visual CHR clusters (Stage 3) and finally votes CHR → S-ID
(Stage 5).  On movie_full_pipeline_20260912_v4 this loses most of the recall:
190 tracklets → 42 survive the primary policy view → 7 CHR + 29 dropped
pending candidates → 10 materialised (scope, S-ID) bindings.

This module removes both lossy hops.  For every global clip it:

1.  Stitches the per-shot ``tracklet_overlay.mp4`` files (boxes already carry
    globally unique ``<SHOT_ID>_T###`` labels) back into one
    ``global_####_tracklets.mp4`` with ffmpeg, in ``shot_ids`` order.
2.  Sends that single video to MiMo together with one tracklet inventory
    (global-relative time ranges) and the global/segment/shot caption
    evidence, asking for a direct tracklet → S-ID assignment.
3.  Writes validated per-global alignment JSON plus a combined JSONL.

Character materialisation (copying image crops and voice previews into
``characters/<scope>/<S-ID>/``) lives in ``global_character_materializer.py``.

This script never mutates the v4 workspace; everything lands under --output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_EFP_ROOT = Path(__file__).resolve().parents[1]
if str(_EFP_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_EFP_ROOT))
_VP_DIR = _EFP_ROOT / "assets" / "visual_person"
if str(_VP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_VP_DIR))

from mimo_caption_client import (  # noqa: E402
    build_video_content,
    configure_no_proxy_for_url,
    encode_video_b64,
    extract_response_text,
    parse_json_response,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from semantic_role_tracklet_refiner import (  # noqa: E402
    AsyncRateGate,
    allowed_subject_ids,
    load_caption_context,
    subject_importance,
)

LOGGER = logging.getLogger("global_tracklet_aligner")
SCHEMA_VERSION = "global-tracklet-character-alignment-v1"
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "global_tracklet_alignment_prompt_en_v1.md"
SHOT_NUMBER_PATTERN = re.compile(r"(\d+)$")
SUBJECT_PATTERN = re.compile(r"S[1-9][0-9]*$")


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #

def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _shot_number(value: Any) -> Optional[int]:
    match = SHOT_NUMBER_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def _scope_id(global_id: Any) -> str:
    raw = str(global_id)
    return f"global_{int(raw):04d}" if raw.isdigit() else raw


@dataclass
class GlobalInput:
    scope_id: str
    global_id: Any
    shot_numbers: List[int]                                    # ordered
    time_range_sec: List[float]
    overlays: List[Path] = field(default_factory=list)          # ordered, per shot
    signatures: Dict[str, dict] = field(default_factory=dict)   # SHOT dir name -> payload
    shot_dir_names: List[str] = field(default_factory=list)


def build_global_inputs(tracklets_root: Path, global_manifest: Path,
                        shots_manifest: Path) -> List[GlobalInput]:
    """Join global_clips.jsonl with the local tracklets/shots/<SHOT>/ layout."""
    tracklets_root = tracklets_root.resolve()
    shots_dir = tracklets_root / "shots"
    if not shots_dir.is_dir():
        raise FileNotFoundError(f"missing {shots_dir}")

    available: Dict[int, Path] = {}
    for entry in sorted(shots_dir.iterdir()):
        if not entry.is_dir():
            continue
        number = _shot_number(entry.name)
        if number is None:
            continue
        if (entry / "tracklet_signatures.json").is_file():
            available[number] = entry
    if not available:
        raise FileNotFoundError(f"no tracklet_signatures.json under {shots_dir}")

    shot_rows: Dict[int, dict] = {}
    for row in _load_jsonl(shots_manifest):
        number = _shot_number(row.get("shot_id"))
        if number is not None:
            shot_rows[number] = row

    inputs: List[GlobalInput] = []
    for row in _load_jsonl(global_manifest):
        ordered: List[int] = []
        for raw in row.get("shot_ids", []):
            number = _shot_number(raw)
            if number is None:
                continue
            ordered.append(number)
        if not ordered:
            continue
        item = GlobalInput(
            scope_id=_scope_id(row.get("global_id")),
            global_id=row.get("global_id"),
            shot_numbers=ordered,
            time_range_sec=list(row.get("time_range_sec", [0.0, 0.0])),
        )
        for number in ordered:
            entry = available.get(number)
            if entry is None:
                LOGGER.warning("%s: shot %d has no tracklets, skipped in stitch",
                               item.scope_id, number)
                continue
            overlay = entry / "tracklet_overlay.mp4"
            if not overlay.is_file():
                LOGGER.warning("%s: shot %d missing overlay %s", item.scope_id, number, overlay)
                continue
            payload = json.loads((entry / "tracklet_signatures.json").read_text(encoding="utf-8"))
            item.overlays.append(overlay)
            item.signatures[entry.name] = payload
            item.shot_dir_names.append(entry.name)
        if item.overlays:
            inputs.append(item)
    if not inputs:
        raise ValueError("no global clip has any tracklet overlay")
    return inputs


# --------------------------------------------------------------------------- #
# stitching
# --------------------------------------------------------------------------- #

def stitch_global_overlay(item: GlobalInput, output_dir: Path, ffmpeg: str,
                          reencode: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{item.scope_id}_tracklets.mp4"
    if out_path.is_file():
        return out_path
    if len(item.overlays) == 1:
        shutil.copyfile(item.overlays[0], out_path)
        return out_path

    list_path = output_dir / f"{item.scope_id}_concat.txt"
    list_path.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in item.overlays),
        encoding="utf-8")
    if reencode:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", str(out_path)]
    else:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
               "-an", "-c", "copy", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed for {item.scope_id}: "
                           f"{result.stderr[-2000:]}")
    list_path.unlink(missing_ok=True)
    return out_path


# --------------------------------------------------------------------------- #
# inventory / prompt
# --------------------------------------------------------------------------- #

def global_tracklet_inventory(item: GlobalInput, shot_rows: Dict[int, dict]) -> List[dict]:
    """Tracklet inventory with global-clip-relative time ranges."""
    inventory: List[dict] = []
    global_start = float(item.time_range_sec[0])
    for dir_name in item.shot_dir_names:
        payload = item.signatures[dir_name]
        number = _shot_number(dir_name)
        shot_row = shot_rows.get(number, {})
        shot_start = float(shot_row.get("time_range_sec", [0.0, 0.0])[0])
        offset = shot_start - global_start  # shot start within the global clip
        for tracklet in payload.get("tracklet_signatures", []):
            faces = sorted(tracklet.get("face_gallery", []),
                           key=lambda v: float(v.get("quality_score", 0.0)), reverse=True)
            bodies = sorted(tracklet.get("body_gallery", []),
                            key=lambda v: float(v.get("quality_score", 0.0)), reverse=True)
            inventory.append({
                "tracklet_id": str(tracklet.get("tracklet_id")),
                "shot_id": str(tracklet.get("shot_id", dir_name)),
                "time_range_sec_in_global_clip": [
                    round(offset + float(tracklet.get("start_time", 0.0)), 3),
                    round(offset + float(tracklet.get("end_time", 0.0)), 3),
                ],
                "observation_count": len(tracklet.get("observation_ids", [])),
                "face_evidence_count": len(faces),
                "best_face": ({key: faces[0].get(key) for key in
                               ("observation_id", "frame_index", "det_score", "quality_score")}
                              if faces else None),
                "body_evidence_count": len(bodies),
                "best_body": ({key: bodies[0].get(key) for key in
                               ("observation_id", "frame_index", "grounding_score", "quality_score")}
                              if bodies else None),
                "cannot_links": list(tracklet.get("cannot_links", [])),
                "track_quality": tracklet.get("quality", {}),
            })
    return inventory


def build_caption_evidence(item: GlobalInput,
                           caption_context: Dict[str, dict]) -> List[dict]:
    """Per-shot caption context, ordered, plus the global-level block first."""
    evidence: List[dict] = []
    global_block: Optional[dict] = None
    for dir_name in item.shot_dir_names:
        number = _shot_number(dir_name)
        context = caption_context.get(str(number), {})
        if global_block is None and isinstance(context.get("global"), dict):
            global_block = context["global"]
        evidence.append({
            "shot_number": number,
            "shot": context.get("shot", {}),
            "segment": context.get("segment", {}),
        })
    return [{"global": global_block or {}}, *evidence]


def build_request_prompt(base_prompt: str, item: GlobalInput, video_path: Path,
                         inventory: List[dict], caption_evidence: List[dict],
                         allowed_ids: Sequence[str], importance: dict) -> tuple:
    package = {
        "global_scope_id": item.scope_id,
        "global_clip_duration_sec": round(
            float(item.time_range_sec[1]) - float(item.time_range_sec[0]), 3),
        "stitched_video": video_path.name,
        "allowed_semantic_subject_ids": list(allowed_ids),
        "authoritative_subject_importance": importance,
        "caption_evidence": caption_evidence,
        "tracklet_inventory": inventory,
    }
    prompt = (base_prompt.rstrip()
              + "\n\nINPUT PACKAGE\n"
              + json.dumps(package, ensure_ascii=False, indent=2))
    return prompt, package


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(result, 0.0), 1.0), 6) if math.isfinite(result) else 0.0


def normalize_alignment(payload: dict, item: GlobalInput,
                        inventory: List[dict], allowed_ids: Sequence[str]) -> dict:
    if str(payload.get("global_scope_id") or payload.get("global_id")) not in {
            item.scope_id, str(item.global_id)}:
        raise ValueError(f"response scope {payload.get('global_scope_id')!r} "
                         f"!= {item.scope_id}")
    expected = {row["tracklet_id"] for row in inventory}
    rows = payload.get("tracklet_alignments")
    if not isinstance(rows, list):
        raise ValueError("tracklet_alignments must be a list")

    allowed = set(allowed_ids) | {"unknown", "new_subject_candidate"}
    by_id: Dict[str, dict] = {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"tracklet_alignments[{index}] must be an object")
        tracklet_id = str(raw.get("tracklet_id") or "")
        if tracklet_id not in expected:
            raise ValueError(f"unknown tracklet_id {tracklet_id!r}")
        if tracklet_id in by_id:
            raise ValueError(f"duplicate tracklet_id {tracklet_id!r}")
        subject_id = str(raw.get("subject_id") or "unknown")
        if subject_id not in allowed:
            raise ValueError(f"subject_id {subject_id!r} not in allowed set")
        by_id[tracklet_id] = {
            "tracklet_id": tracklet_id,
            "subject_id": subject_id,
            "importance_tier": str(raw.get("importance_tier") or "unknown"),
            "confidence": _confidence(raw.get("confidence")),
            "visual_cues": [str(v) for v in raw.get("visual_cues", [])
                            if isinstance(v, (str, int, float))],
            "reason": str(raw.get("reason") or ""),
        }

    # fail-open: every inventory tracklet must appear exactly once
    missing = []
    for row in inventory:
        tracklet_id = row["tracklet_id"]
        if tracklet_id not in by_id:
            missing.append(tracklet_id)
            by_id[tracklet_id] = {
                "tracklet_id": tracklet_id,
                "subject_id": "unknown",
                "importance_tier": "unknown",
                "confidence": 0.0,
                "visual_cues": [],
                "reason": "missing_alignment_fail_open",
            }

    assigned = sum(1 for row in by_id.values()
                   if row["subject_id"] not in {"unknown", "new_subject_candidate"})
    return {
        "schema_version": SCHEMA_VERSION,
        "global_scope_id": item.scope_id,
        "global_id": item.global_id,
        "status": "ok",
        "stitched_video": f"{item.scope_id}_tracklets.mp4",
        "tracklet_count": len(inventory),
        "assigned_tracklet_count": assigned,
        "tracklet_alignments": [by_id[row["tracklet_id"]] for row in inventory],
        "missing_alignment_tracklet_ids": missing,
        "unmatched_semantic_subject_ids": [str(v) for v in
                                           payload.get("unmatched_semantic_subject_ids", [])],
        "quality_issues": [str(v) for v in payload.get("quality_issues", [])],
    }


# --------------------------------------------------------------------------- #
# MiMo call
# --------------------------------------------------------------------------- #

async def align_one(client, semaphore, item: GlobalInput, shot_rows: Dict[int, dict],
                    caption_context: Dict[str, dict], base_prompt: str,
                    args: argparse.Namespace, rate_gate: Optional[AsyncRateGate]) -> dict:
    inventory = global_tracklet_inventory(item, shot_rows)
    if not inventory:
        return {"schema_version": SCHEMA_VERSION, "global_scope_id": item.scope_id,
                "status": "skipped_no_tracklets"}

    context_for_ids: Dict[str, Any] = {}
    for dir_name in item.shot_dir_names:
        number = _shot_number(dir_name)
        context_for_ids.update(caption_context.get(str(number), {}))
    allowed_ids = allowed_subject_ids(context_for_ids)
    importance = subject_importance(context_for_ids)

    video_path = stitch_global_overlay(item, args.output / "stitched",
                                       args.ffmpeg, reencode=not args.stream_copy)
    prompt, package = build_request_prompt(base_prompt, item, video_path, inventory,
                                           build_caption_evidence(item, caption_context),
                                           allowed_ids, importance)
    (args.output / "requests").mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "requests" / f"{item.scope_id}_package.json", {
        "schema_version": SCHEMA_VERSION,
        "global_scope_id": item.scope_id,
        "stitched_video": str(video_path),
        "prompt_file": str(args.prompt_file),
        "input_package": package,
    })
    if args.dry_run:
        return {"schema_version": SCHEMA_VERSION, "global_scope_id": item.scope_id,
                "status": "dry_run_no_model_call", "stitched_video": str(video_path),
                "tracklet_count": len(inventory)}

    formats = ["video_url", "input_video"]
    attempts: List[dict] = []
    backoff = 0.0
    started = time.perf_counter()
    async with semaphore:
        if rate_gate is not None:
            await rate_gate.wait()
        encoded = await asyncio.to_thread(encode_video_b64, video_path)
        last_error = "unknown_error"
        for attempt in range(args.retries):
            temperature = min(1.0, args.temperature + 0.2 * attempt)
            for request_format in formats:
                request_started = time.perf_counter()
                try:
                    if rate_gate is not None:
                        await rate_gate.wait()
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": build_video_content(
                            prompt, encoded, request_format)}],
                        temperature=temperature,
                        max_tokens=args.max_completion_tokens,
                        timeout=args.timeout,
                    )
                    text = extract_response_text(response)
                    parsed, parse_error = parse_json_response(text)
                    if parse_error is not None or parsed is None:
                        raise ValueError(parse_error or "response is not JSON")
                    result = normalize_alignment(parsed, item, inventory, allowed_ids)
                    result.update({
                        "model": args.model,
                        "request_format": request_format,
                        "response_seconds": round(time.perf_counter() - started, 3),
                        "attempts": attempts + [{
                            "attempt": attempt + 1,
                            "request_format": request_format,
                            "status": "ok",
                            "request_seconds": round(time.perf_counter() - request_started, 3),
                        }],
                    })
                    return result
                except Exception as exc:  # noqa: BLE001 - retry boundary
                    last_error = f"{request_format}: {type(exc).__name__}: {exc}"
                    LOGGER.warning("%s attempt %d %s failed: %s",
                                   item.scope_id, attempt + 1, request_format, last_error)
                    attempts.append({
                        "attempt": attempt + 1,
                        "request_format": request_format,
                        "status": "error",
                        "error": last_error[-800:],
                    })
            backoff = min(60.0, 2.0 ** attempt)
            await asyncio.sleep(backoff)
        return {
            "schema_version": SCHEMA_VERSION,
            "global_scope_id": item.scope_id,
            "global_id": item.global_id,
            "status": "failed",
            "error": last_error,
            "attempts": attempts,
            "response_seconds": round(time.perf_counter() - started, 3),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    global_inputs = build_global_inputs(args.tracklets, args.global_manifest, args.shots)
    shot_rows = {_shot_number(row.get("shot_id")): row
                 for row in _load_jsonl(args.shots) if _shot_number(row.get("shot_id"))}
    caption_context = load_caption_context(args.global_captions, args.shot_captions,
                                           args.segment_captions)
    base_prompt = args.prompt_file.read_text(encoding="utf-8")

    from openai import AsyncOpenAI  # deferred: keeps --dry-run usable without it
    base_url = resolve_base_url(args.base_url)
    configure_no_proxy_for_url(base_url)
    client = AsyncOpenAI(base_url=base_url,
                         api_key=resolve_api_key(args.api_key))
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_gate = AsyncRateGate(args.rate_limit)

    results = await asyncio.gather(*[
        align_one(client, semaphore, item, shot_rows, caption_context,
                  base_prompt, args, rate_gate)
        for item in global_inputs])

    results = sorted(results, key=lambda row: str(row.get("global_scope_id")))
    (args.output / "global_tracklet_alignment.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8")
    for row in results:
        _write_json(args.output / "alignments" / f"{row.get('global_scope_id')}.json", row)

    ok = [row for row in results if row.get("status") == "ok"]
    total = sum(row.get("tracklet_count", 0) for row in ok)
    assigned = sum(row.get("assigned_tracklet_count", 0) for row in ok)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "global_count": len(results),
        "ok_count": len(ok),
        "failed_count": sum(1 for row in results if row.get("status") == "failed"),
        "tracklet_count": total,
        "assigned_tracklet_count": assigned,
        "assignment_rate": round(assigned / total, 4) if total else 0.0,
        "per_global": [{
            "global_scope_id": row.get("global_scope_id"),
            "status": row.get("status"),
            "tracklet_count": row.get("tracklet_count"),
            "assigned_tracklet_count": row.get("assigned_tracklet_count"),
        } for row in results],
    }
    _write_json(args.output / "summary.json", summary)
    LOGGER.info("done: %s", json.dumps(summary["per_global"], ensure_ascii=False))
    return 0 if ok or args.dry_run else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", type=Path, required=True,
                        help="v4 tracklets root (contains shots/<SHOT_ID>/)")
    parser.add_argument("--global-manifest", type=Path, required=True,
                        help="global_clips.jsonl")
    parser.add_argument("--shots", type=Path, required=True, help="shots.jsonl")
    parser.add_argument("--global-captions", type=Path, default=None)
    parser.add_argument("--shot-captions", type=Path, default=None)
    parser.add_argument("--segment-captions", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--stream-copy", action="store_true",
                        help="concat without re-encode (fast, requires identical codecs)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true",
                        help="stitch and write request packages without calling MiMo")
    args = parser.parse_args()
    args.model = resolve_model(args.model)
    args.prompt_file = args.prompt_file.resolve()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
