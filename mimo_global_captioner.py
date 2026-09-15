#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annotate 1-2 minute global clips with MiMo through an OpenAI-compatible API.

Atomic migration of ``lsj/video_process/scripts/annotate_global_mimo.py``. Reads
``global_clips.jsonl`` produced by :mod:`mimo_global_clip_aggregator`, sends each
clip together with :data:`SOURCE_GLOBAL_PROMPT` to a MiMo endpoint, validates
the ``subject_registry`` / ``subject_event_timeline`` / ``global_event_timeline``
contract and writes:

    <output>/<video>/global_captions/global_<id>.json
    <output>/<video>/raw_responses/global_<id>.txt
    <output>/<video>/global_captions.jsonl
    <output>/<video>/summary.json

Usage::

    python mimo_global_captioner.py \
        --input ./outputs/global_outputs/<video> \
        --output-dir ./outputs/global_caption_outputs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from mimo_caption_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_video_content,
    default_prompt_path,
    encode_video_b64,
    extract_text,
    find_global_clip_manifest,
    load_dotenv,
    parse_json_response,
    persist_item,
    resolve_api_key,
    resolve_base_url,
    resolve_clip_path,
    resolve_model,
    write_jsonl,
)

LOGGER = logging.getLogger("mimo_global_captioner")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_global_payload(payload: dict) -> None:
    """Require the global person registry used by downstream shot/segment stages."""
    registry = payload.get("subject_registry")
    timeline = payload.get("subject_event_timeline")
    event_timeline = payload.get("global_event_timeline")
    if not isinstance(registry, list):
        raise ValueError("global response must contain subject_registry (use [] when no people appear)")
    if not isinstance(timeline, list):
        raise ValueError("global response must contain subject_event_timeline (use [] when no people appear)")
    if not isinstance(event_timeline, list):
        raise ValueError("global response must contain global_event_timeline (use [] when no events are visible)")
    ids: set[str] = set()
    for index, subject in enumerate(registry, start=1):
        if not isinstance(subject, dict):
            raise ValueError(f"subject_registry[{index}] must be an object")
        subject_id = str(subject.get("subject_id", ""))
        if not re.fullmatch(r"S[1-9][0-9]*", subject_id):
            raise ValueError(f"subject_registry[{index}] must use a person ID such as S1: {subject_id!r}")
        if subject.get("subject_type") != "person":
            raise ValueError(f"{subject_id} must have subject_type=person")
        if subject_id in ids:
            raise ValueError(f"duplicate subject ID: {subject_id}")
        ids.add(subject_id)
    for index, event in enumerate(timeline, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"subject_event_timeline[{index}] must be an object")
        if not isinstance(event.get("subject_ids", []), list):
            raise ValueError(f"subject_event_timeline[{index}].subject_ids must be a list")
        for subject_id in event.get("subject_ids", []):
            if subject_id != "unknown" and subject_id not in ids:
                raise ValueError(f"timeline references unregistered subject: {subject_id}")
        speaker_id = event.get("speaker_id", "unknown")
        if speaker_id != "unknown" and speaker_id not in ids:
            raise ValueError(f"timeline references unregistered speaker: {speaker_id}")
    allowed_phases = {"introduction", "continuation", "escalation", "turning_point", "resolution", "transition"}
    for index, event in enumerate(event_timeline, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"global_event_timeline[{index}] must be an object")
        for field in ("time_range_sec", "event_phase", "subject_ids", "event", "state_before", "state_after", "speaker_ids"):
            if field not in event:
                raise ValueError(f"global_event_timeline[{index}] missing {field}")
        time_range = event["time_range_sec"]
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            raise ValueError(f"global_event_timeline[{index}].time_range_sec must have two numbers")
        try:
            if float(time_range[1]) < float(time_range[0]):
                raise ValueError(f"global_event_timeline[{index}] ends before it starts")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"global_event_timeline[{index}].time_range_sec must be numeric") from exc
        if event["event_phase"] not in allowed_phases:
            raise ValueError(f"global_event_timeline[{index}] has invalid event_phase: {event['event_phase']!r}")
        if not isinstance(event["subject_ids"], list) or not isinstance(event["speaker_ids"], list):
            raise ValueError(f"global_event_timeline[{index}] subject_ids and speaker_ids must be lists")
        for field in ("subject_ids", "speaker_ids"):
            for subject_id in event[field]:
                if subject_id != "unknown" and subject_id not in ids:
                    raise ValueError(f"global_event_timeline[{index}] references unregistered subject: {subject_id}")
        for field in ("event", "state_before", "state_after"):
            if not isinstance(event[field], str):
                raise ValueError(f"global_event_timeline[{index}].{field} must be a string")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def find_manifests(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "global_clips.jsonl":
            raise ValueError(f"Input file must be global_clips.jsonl: {input_path}")
        return [input_path]
    manifests = sorted(input_path.rglob("global_clips.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No global_clips.jsonl found under {input_path}")
    return manifests


def load_records(manifest: Path) -> list[dict]:
    records = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {manifest}:{line_no}: {exc}") from exc
        if not isinstance(record, dict) or "global_id" not in record:
            raise ValueError(f"Missing global_id at {manifest}:{line_no}")
        clip_path = record.get("clip_path")
        if not clip_path:
            raise ValueError(f"Missing clip_path at {manifest}:{line_no}")
        path = resolve_clip_path(clip_path, manifest)
        if not path.is_file():
            raise FileNotFoundError(f"Global clip does not exist: {path}")
        record = dict(record)
        record["clip_path"] = str(path)
        records.append(record)
    records.sort(key=lambda item: int(item["global_id"]))
    if not records:
        raise ValueError(f"Manifest is empty: {manifest}")
    return records


# ---------------------------------------------------------------------------
# Async annotation
# ---------------------------------------------------------------------------
async def annotate_one(client, semaphore, record: dict, prompt: str, args: argparse.Namespace) -> dict:
    clip_path = Path(record["clip_path"])
    global_id = int(record["global_id"])
    context = (
        f"\nThis is global clip {global_id}. Its source-video time range is "
        f"{record.get('time_range_sec')}, and it contains shot IDs "
        f"{record.get('shot_ids', [])}. Watch this complete clip and annotate only it.\n"
    )
    request_prompt = prompt.rstrip() + context
    formats = [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
    started = time.perf_counter()
    last_error = ""
    last_response = ""
    async with semaphore:
        encoded = await asyncio.to_thread(encode_video_b64, clip_path)
        for current_format in formats:
            try:
                extra_body = {"chat_template_kwargs": {"enable_thinking": args.thinking}} if args.thinking else {}
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "user", "content": build_video_content(request_prompt, encoded, current_format)}
                        ],
                        temperature=args.temperature,
                        max_tokens=args.max_completion_tokens,
                        extra_body=extra_body,
                    ),
                    timeout=args.timeout,
                )
                last_response = extract_text(response)
                parsed, parse_error = parse_json_response(last_response)
                if parse_error is None and parsed is not None:
                    validate_global_payload(parsed)
                    return {
                        **record,
                        "success": True,
                        "model": args.model,
                        "request_format": current_format,
                        "global_caption": parsed,
                        "response_seconds": round(time.perf_counter() - started, 3),
                    }
                last_error = parse_error or "invalid response"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{current_format}: {type(exc).__name__}: {exc}"
    return {
        **record,
        "success": False,
        "model": args.model,
        "error": last_error,
        "response_text": last_response or None,
        "response_seconds": round(time.perf_counter() - started, 3),
    }


async def annotate_manifest(manifest: Path, args: argparse.Namespace, client) -> dict:
    records = load_records(manifest)
    output_dir = args.output_dir if getattr(args, "flat", False) else args.output_dir / manifest.parent.name
    individual_dir = output_dir / "global_captions"
    raw_dir = output_dir / "raw_responses"
    individual_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)
    jsonl_path = output_dir / "global_captions.jsonl"

    all_results: dict[int, dict] = {}
    if args.skip_existing:
        for record in records:
            output_path = individual_dir / f"global_{int(record['global_id']):05d}.json"
            if output_path.is_file():
                try:
                    existing = json.loads(output_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and existing.get("success") is True:
                        all_results[int(record["global_id"])] = existing
                except json.JSONDecodeError:
                    pass

    rounds = max(1, args.retries)
    base_temperature = args.temperature
    for round_no in range(1, rounds + 1):
        args.temperature = min(1.0, base_temperature + 0.2 * (round_no - 1))
        pending = [r for r in records if all_results.get(int(r["global_id"]), {}).get("success") is not True]
        if pending:
            tasks = [asyncio.create_task(annotate_one(client, semaphore, record, prompt, args)) for record in pending]
            progress_bar = None
            if args.progress:
                try:
                    from tqdm import tqdm
                    progress_bar = tqdm(
                        total=len(tasks),
                        desc=f"Global caption {manifest.parent.name} (round {round_no}/{rounds})",
                        unit="clip",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass
            success_count = 0
            failed_count = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                global_id = int(result["global_id"])
                all_results[global_id] = result
                persist_item(
                    individual_dir, raw_dir, "global", global_id, result, result.get("response_text")
                )
                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(ok=success_count, fail=failed_count, refresh=False)
                    if args.log_each:
                        progress_bar.write(
                            f"[{success_count + failed_count}/{len(tasks)}] "
                            f"{'ok' if result.get('success') else 'fail'} global_{global_id:05d}"
                        )
            if progress_bar is not None:
                progress_bar.close()

        ordered = [all_results[int(record["global_id"])] for record in records]
        write_jsonl(jsonl_path, ordered)
        total_success = sum(1 for result in all_results.values() if result.get("success"))
        total_failed = len(all_results) - total_success
        summary = {
            "global_manifest": str(manifest),
            "global_clip_count": len(records),
            "success_count": total_success,
            "failed_count": total_failed,
            "rounds_used": round_no,
            "model": args.model,
            "base_url": args.base_url,
            "output": str(jsonl_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        LOGGER.info(
            "%s: round %d/%d: %d/%d global captions succeeded",
            manifest.parent.name, round_no, rounds, total_success, len(records),
        )
        if not pending or total_failed == 0:
            break
        LOGGER.warning("%s: round %d left %d failures, running next round", manifest.parent.name, round_no, total_failed)
        backoff = min(300.0, 60.0 * (2 ** (round_no - 1)))
        LOGGER.info("%s: sleeping %.0fs before retry round %d", manifest.parent.name, backoff, round_no + 1)
        await asyncio.sleep(backoff)
    return summary


async def async_main(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Missing openai package.") from exc
    manifests = find_manifests(args.input)
    if getattr(args, "flat", False) and len(manifests) > 1:
        raise SystemExit("--flat 只支持单个 manifest 输入")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    failed = 0
    for manifest in manifests:
        try:
            summary = await annotate_manifest(manifest, args, client)
            failed += int(summary["failed_count"] > 0)
        except Exception:
            failed += 1
            LOGGER.exception("Failed to annotate %s", manifest)
    return 1 if failed else 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Annotate global clips with MiMo via an OpenAI-compatible API.")
    parser.add_argument("--input", required=True, type=Path, help="global_clips.jsonl or a directory containing manifests.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录")
    parser.add_argument("--prompt-file", type=Path, default=default_prompt_path("global"))
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=["auto", "video_url", "input_video"], default="video_url")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    # V3 includes the role registry, per-shot visibility and two timelines;
    # 4096 truncated a real 76-second global before subject_event_timeline.
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-each", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1:
        parser.error("--concurrency and --retries must be positive")
    if args.max_api_seconds is not None:
        if args.max_api_seconds <= 0:
            parser.error("--max-api-seconds must be positive")
        args.timeout = min(args.timeout, args.max_api_seconds)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    for logger_name in ("httpx", "httpcore", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).disabled = True
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
