#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cut global clips into captioned semantic segments using ffmpeg.

Atomic migration of ``lsj/video_process/scripts/split_segments.py``. Reads a
``segment_captions.jsonl`` produced by :mod:`mimo_segment_captioner`, locates
the corresponding ``global_clips.jsonl`` for clip-path fallback, and cuts
each segment from its parent global clip via ffmpeg.

Outputs::

    <output>/<video>/clips/<video>_global_NNNN_segment_NNN.mp4
    <output>/<video>/segments.jsonl
    <output>/<video>/summary.json

Usage::

    python mimo_segment_cutter.py \
        --input ./outputs/segment_caption_outputs/<video> \
        --output-dir ./outputs/segment_outputs
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import subprocess
from pathlib import Path

from mimo_caption_client import find_segment_caption_manifest, load_jsonl

LOGGER = logging.getLogger("mimo_segment_cutter")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def find_manifest(path: Path) -> Path:
    return find_segment_caption_manifest(path)


def resolve_path(value: str | Path | None, bases: list[Path]) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases] + [raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_global_clip_map(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    manifest = path
    if path.is_dir():
        candidates = sorted(path.rglob("global_clips.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No global_clips.jsonl found under {path}")
        manifest = candidates[0]
    result: dict[int, str] = {}
    for record in load_jsonl(manifest):
        if "global_id" in record and record.get("clip_path"):
            result[int(record["global_id"])] = str(record["clip_path"])
    return result


def segment_list(record: dict) -> list[dict]:
    payload = record.get("segment_caption")
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return payload["segments"]
    if isinstance(record.get("segments"), list):
        return record["segments"]
    return []


def time_range(value, duration: float, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-item range")
    start, end = float(value[0]), float(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"{label} is invalid: {value}")
    if duration > 0 and end > duration + 0.25:
        raise ValueError(f"{label} ends at {end:.3f}s but global clip is {duration:.3f}s")
    return max(0.0, start), min(end, duration) if duration > 0 else end


def cut_clip(input_path: Path, output_path: Path, start: float, end: float, args) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    command = [
        args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}", "-i", str(input_path), "-t", f"{duration:.6f}",
        "-map", "0:v:0?", "-map", "0:a:0?",
    ]
    if args.copy:
        command += ["-c", "copy"]
    else:
        command += [
            "-c:v", args.video_codec, "-preset", args.preset, "-crf", str(args.crf),
            "-c:a", args.audio_codec, "-b:a", args.audio_bitrate,
            "-movflags", "+faststart",
        ]
    command.append(str(output_path))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut global clips into captioned semantic segments using ffmpeg.")
    parser.add_argument("--input", required=True, type=Path,
                        help="segment output directory or segment_captions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flat", action="store_true",
                        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录")
    parser.add_argument("--global-input", type=Path, default=None,
                        help="Optional global_outputs directory/manifest for clip-path fallback.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--copy", action="store_true",
                        help="Stream-copy instead of re-encoding (faster but keyframe-inaccurate).")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        parser.error(f"ffmpeg not found: {args.ffmpeg}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = find_manifest(args.input)
    records = load_jsonl(manifest)
    global_map = load_global_clip_map(args.global_input)
    video_name = manifest.parent.name
    if args.flat:
        output_root = args.output_dir
        for _rec in records:
            _src = Path(str(_rec.get("source_video") or "")).stem
            if _src:
                video_name = _src
                break
    else:
        output_root = args.output_dir / video_name
    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    bases = [manifest.parent, args.input if args.input.is_dir() else manifest.parent,
             Path(__file__).resolve().parent, Path.cwd()]

    jobs = []
    skipped_records = []
    for record in records:
        if record.get("success") is not True:
            skipped_records.append({"global_id": record.get("global_id"), "success": False,
                                    "error": "segment annotation failed"})
            continue
        gid = int(record["global_id"])
        global_path = resolve_path(record.get("clip_path") or global_map.get(gid), bases)
        if global_path is None:
            LOGGER.warning("Global clip not found for global_id=%s", gid)
            skipped_records.append({"global_id": gid, "success": False, "error": "global clip not found"})
            continue
        caption_payload = record.get("segment_caption")
        caption_duration = caption_payload.get("video_duration_sec") if isinstance(caption_payload, dict) else None
        duration = float(record.get("duration_sec") or caption_duration or 0)
        for index, segment in enumerate(segment_list(record), 1):
            jobs.append((record, segment, gid, index, global_path, duration))

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(jobs), desc=f"Split segments {video_name}", unit="segment",
                        dynamic_ncols=True) if args.progress else None
    except ImportError:
        progress = None
    output_records = list(skipped_records)
    manifest_path = output_root / "segments.jsonl"
    with manifest_path.open("w", encoding="utf-8", buffering=1) as handle:
        for record in skipped_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for source_record, segment, gid, index, global_path, duration in jobs:
            output_name = f"{video_name}_global_{gid:04d}_segment_{index:03d}.mp4"
            output_path = clips_dir / output_name
            result = copy.deepcopy(segment)
            result.update({
                "global_id": gid,
                "segment_id": int(segment.get("segment_id", index)),
                "global_clip_path": str(global_path),
                "segment_clip_path": str(output_path.resolve()),
                "source_video": source_record.get("source_video"),
                "success": False,
            })
            try:
                start, end = time_range(segment.get("time_range_sec"), duration,
                                        f"global {gid} segment {index} time_range_sec")
                if not args.skip_existing or not output_path.is_file():
                    cut_clip(global_path, output_path, start, end, args)
                result["time_range_sec"] = [round(start, 6), round(end, 6)]
                result["duration_sec"] = round(end - start, 6)
                result["success"] = True
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"
                LOGGER.error("Failed global %s segment %s: %s", gid, index, exc)
            output_records.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if progress is not None:
                progress.update(1)
                progress.set_postfix(ok=sum(item.get("success") is True for item in output_records), refresh=False)
    if progress is not None:
        progress.close()
    output_records.sort(key=lambda item: (int(item.get("global_id", 0)), int(item.get("segment_id", 0))))
    summary = {
        "input_manifest": str(manifest.resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "segment_count": len(output_records),
        "success_count": sum(item.get("success") is True for item in output_records),
        "failed_count": sum(item.get("success") is not True for item in output_records),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d/%d successful)", manifest_path, summary["success_count"], summary["segment_count"])
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())