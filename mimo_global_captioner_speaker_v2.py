#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-A global captioner: one MiMo call for caption + speaker attribution.

This is intentionally a new executable.  ``mimo_global_captioner.py`` and its
default prompt retain their previous behavior.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import mimo_global_captioner as base
from mimo_caption_client import (
    build_video_content,
    encode_video_b64,
    extract_text,
    load_dotenv,
    parse_json_response,
    persist_item,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    write_jsonl,
)
from mimo_segment_captioner import (
    align_authoritative_dialogue,
    load_asr_sidecars,
    prepare_authoritative_asr,
)

LOGGER = logging.getLogger("mimo_global_captioner_speaker_v2")
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "global_caption_prompt_en_semantic_roles_speakers_v4.md"
SCHEMA_VERSION = "global-caption-speaker-attribution-v2"


def _duration(record: dict[str, Any]) -> float:
    if record.get("duration_sec") is not None:
        return float(record["duration_sec"])
    start, end = record["time_range_sec"]
    return float(end) - float(start)


def build_input_text(
    prompt: str,
    record: dict[str, Any],
    asr_timeline: Sequence[dict[str, Any]],
) -> str:
    context = {
        "global_id": record.get("global_id"),
        "source_video_time_range_sec": record.get("time_range_sec"),
        "duration_sec": _duration(record),
        "shot_ids": record.get("shot_ids", []),
    }
    return (
        prompt.rstrip()
        + "\n\nINPUT DATA\n[Global Clip Metadata]\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n[Authoritative ASR Timeline]\n"
        + json.dumps(list(asr_timeline), ensure_ascii=False, indent=2)
        + "\n\nReturn one attribution for every supplied asr_id. "
          "Do not reproduce ASR text or timestamps in the response.\n"
    )


def validate_and_enrich(
    payload: dict[str, Any],
    record: dict[str, Any],
    asr_timeline: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate MiMo judgments, then perform an immutable local ASR join."""
    base.validate_global_payload(payload)
    raw_attributions = payload.get("asr_speaker_attributions")
    if not isinstance(raw_attributions, list):
        raise ValueError("global response must contain asr_speaker_attributions")

    expected_ids = [str(row["asr_id"]) for row in asr_timeline]
    returned_ids: list[str] = []
    for index, row in enumerate(raw_attributions, 1):
        if not isinstance(row, dict):
            raise ValueError(f"asr_speaker_attributions[{index}] must be an object")
        asr_id = str(row.get("asr_id") or "").strip()
        if not asr_id:
            raise ValueError(f"asr_speaker_attributions[{index}] missing asr_id")
        returned_ids.append(asr_id)
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("MiMo returned duplicate ASR IDs")
    if set(returned_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(returned_ids))
        invented = sorted(set(returned_ids) - set(expected_ids))
        raise ValueError(f"ASR attribution ID mismatch: missing={missing}, invented={invented}")

    duration = _duration(record)
    joined_segment = {
        "segment_id": 1,
        "time_range_sec": [0.0, duration],
        "dialogue": copy.deepcopy(raw_attributions),
    }
    registered_ids = {
        str(item.get("subject_id"))
        for item in payload.get("subject_registry", [])
        if isinstance(item, dict) and item.get("subject_id")
    }
    align_authoritative_dialogue([joined_segment], asr_timeline, registered_ids)

    result = copy.deepcopy(payload)
    result["asr_speaker_attributions"] = joined_segment["dialogue"]
    result["authoritative_asr_timeline"] = copy.deepcopy(list(asr_timeline))
    result["schema_version"] = SCHEMA_VERSION
    result["speaker_attribution_summary"] = {
        "asr_count": len(asr_timeline),
        "resolved_count": sum(
            row["speaker_id"] != "unknown" for row in joined_segment["dialogue"]
        ),
        "accepted_for_audio_asset_count": sum(
            bool(row["audio_asset_candidate"]) for row in joined_segment["dialogue"]
        ),
    }
    return result


async def annotate_one(client, semaphore, record, prompt, args, asr_timeline):
    started = time.perf_counter()
    request_prompt = build_input_text(prompt, record, asr_timeline)
    formats = [args.request_format] if args.request_format != "auto" else ["video_url", "input_video"]
    last_error = ""
    last_response = ""
    async with semaphore:
        encoded = await asyncio.to_thread(encode_video_b64, Path(record["clip_path"]))
        for current_format in formats:
            try:
                extra_body = {"chat_template_kwargs": {"enable_thinking": args.thinking}} if args.thinking else {}
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        messages=[{
                            "role": "user",
                            "content": build_video_content(request_prompt, encoded, current_format),
                        }],
                        temperature=args.temperature,
                        max_tokens=args.max_completion_tokens,
                        extra_body=extra_body,
                    ),
                    timeout=args.max_api_seconds or args.timeout,
                )
                last_response = extract_text(response)
                parsed, parse_error = parse_json_response(last_response)
                if parse_error is None and parsed is not None:
                    enriched = validate_and_enrich(parsed, record, asr_timeline)
                    return {
                        **record,
                        "success": True,
                        "model": args.model,
                        "request_format": current_format,
                        "global_caption": enriched,
                        "response_seconds": round(time.perf_counter() - started, 3),
                        "_raw_response_text": last_response,
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


def flatten_attributions(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("success") is not True:
            continue
        caption = result.get("global_caption") or {}
        for attribution in caption.get("asr_speaker_attributions") or []:
            rows.append({
                "video_id": result.get("video_id"),
                "global_id": result.get("global_id"),
                "global_clip_path": result.get("clip_path"),
                "source_video": result.get("source_video"),
                **copy.deepcopy(attribution),
            })
    return sorted(rows, key=lambda row: (
        str(row.get("video_id")), int(row.get("global_id") or 0),
        float((row.get("time_range_sec") or [0.0])[0]), str(row.get("asr_id")),
    ))


async def annotate_manifest(manifest: Path, args, client) -> dict[str, Any]:
    records = [
        {**record, "duration_sec": _duration(record)}
        for record in base.load_records(manifest)
    ]
    asr_index = load_asr_sidecars(args.asr_input)
    output_dir = args.output_dir if getattr(args, "flat", False) else args.output_dir / manifest.parent.name
    item_dir = output_dir / "global_captions"
    raw_dir = output_dir / "raw_responses"
    item_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)
    all_results: dict[int, dict[str, Any]] = {}

    if args.skip_existing:
        for record in records:
            path = item_dir / f"global_{int(record['global_id']):05d}.json"
            if not path.is_file():
                continue
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                expected = prepare_authoritative_asr(record, asr_index, args.asr_timebase)
                existing_asr = (existing.get("global_caption") or {}).get("authoritative_asr_timeline")
                if existing.get("success") is True and existing_asr == expected:
                    all_results[int(record["global_id"])] = existing
                elif existing.get("success") is True:
                    LOGGER.info("invalidate cached global %s: authoritative ASR changed", record["global_id"])
            except (OSError, json.JSONDecodeError, ValueError):
                pass

    rounds = max(1, args.retries)
    base_temperature = args.temperature
    summary: dict[str, Any] = {}
    for round_no in range(1, rounds + 1):
        args.temperature = min(1.0, base_temperature + 0.2 * (round_no - 1))
        pending = [r for r in records if int(r["global_id"]) not in all_results]
        if pending:
            tasks = []
            for record in pending:
                timeline = prepare_authoritative_asr(record, asr_index, args.asr_timebase)
                if not timeline:
                    raise ValueError(f"No ASR rows overlap global {record['global_id']}")
                tasks.append(asyncio.create_task(
                    annotate_one(client, semaphore, record, prompt, args, timeline)
                ))
            progress = None
            if args.progress:
                try:
                    from tqdm import tqdm
                    progress = tqdm(total=len(tasks), desc=f"Global+speaker {manifest.parent.name}", unit="clip")
                except ImportError:
                    pass
            for task in asyncio.as_completed(tasks):
                result = await task
                global_id = int(result["global_id"])
                raw_response = result.pop("_raw_response_text", None)
                all_results[global_id] = result
                persist_item(item_dir, raw_dir, "global", global_id, result,
                             raw_response or result.get("response_text"))
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()

        ordered = [all_results[int(record["global_id"])] for record in records]
        caption_path = output_dir / "global_captions.jsonl"
        attribution_path = output_dir / "speaker_attributed_asr.jsonl"
        write_jsonl(caption_path, ordered)
        attribution_rows = flatten_attributions(ordered)
        write_jsonl(attribution_path, attribution_rows)
        success_count = sum(row.get("success") is True for row in ordered)
        summary = {
            "scheme": "A",
            "global_manifest": str(manifest),
            "asr_input": str(args.asr_input),
            "global_clip_count": len(records),
            "success_count": success_count,
            "failed_count": len(records) - success_count,
            "rounds_used": round_no,
            "speaker_attribution_count": len(attribution_rows),
            "speaker_attribution_output": str(attribution_path),
            "output": str(caption_path),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if summary["failed_count"] == 0 or not pending:
            break
        if round_no < rounds:
            await asyncio.sleep(min(300.0, 60.0 * (2 ** (round_no - 1))))
    return summary


async def async_main(args) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Missing openai package.") from exc
    manifests = base.find_manifests(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    failed = 0
    for manifest in manifests:
        try:
            summary = await annotate_manifest(manifest, args, client)
            failed += int(summary["failed_count"] > 0)
        except Exception:
            LOGGER.exception("Failed to annotate %s", manifest)
            failed += 1
    return int(failed > 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scheme A: global caption and ASR-to-person attribution in one MiMo call."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asr-input", required=True, type=Path)
    parser.add_argument("--asr-timebase", choices=("auto", "source", "global-local"), default="auto")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=resolve_base_url())
    parser.add_argument("--api-key", default=resolve_api_key())
    parser.add_argument("--model", default=resolve_model())
    parser.add_argument("--request-format", choices=("auto", "video_url", "input_video"), default="video_url")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-api-seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <manifest父目录名>/ 子目录")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency, retries, and timeout must be positive")
    if args.max_api_seconds is not None and args.max_api_seconds <= 0:
        parser.error("--max-api-seconds must be positive")
    if not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).disabled = True
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
