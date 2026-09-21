#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified video downsampling preprocessing (480p / 15 fps by default).

All downstream stages of the MiMo annotation chain (shot splitting, global
aggregation, captioning, segment export) consume the SAME normalised video,
so frame payloads stay bounded and separate chain runs remain comparable:

    raw_video.mp4
      -> ffprobe                (width / height / fps / duration)
      -> ffmpeg re-encode       (scale=-2:<height>,fps=<fps>; libx264 + aac)
      -> normalized.mp4
      -> normalized.mp4.info.json

The output is ALWAYS re-encoded, even when the input already satisfies the
target height / fps: a stream copy would preserve heterogeneous source
codecs, while a fixed encoder (libx264, yuv420p, +faststart) guarantees one
uniform decoding surface across the whole corpus.

A sidecar ``<output>.info.json`` records the probed input/output geometry,
frame rates, durations, the complete ffmpeg command line and a timestamp,
so any workspace can always tell how its normalised video was produced.

Usage::

    python preprocessing/normalize_video.py \
        --input /path/to/source.mp4 \
        --output /path/to/workspace/normalized/normalized.mp4 \
        [--height 480] [--fps 15] [--crf 20] [--preset fast] [--skip-existing]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from fractions import Fraction
from pathlib import Path

LOGGER = logging.getLogger("normalize_video")

# How many stderr lines to surface when ffmpeg/ffprobe fails.
STDERR_TAIL_LINES = 20


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _stderr_tail(stderr: str, lines: int = STDERR_TAIL_LINES) -> str:
    """Last `lines` lines of a captured stderr stream (may be empty)."""
    return "\n".join((stderr or "").splitlines()[-lines:])


def parse_rate(value: str | None) -> float | None:
    """Parse an ffprobe rational rate (e.g. "30000/1001") into float fps."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def probe_video(video_path: Path, ffprobe_bin: str) -> dict:
    """ffprobe the first video stream; return path/width/height/fps/duration."""
    cmd = [
        ffprobe_bin, "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {video_path} (exit {proc.returncode}); stderr tail:\n"
            f"{_stderr_tail(proc.stderr)}")
    info = json.loads(proc.stdout or "{}")
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise RuntimeError(f"no video stream found: {video_path}")
    duration = stream.get("duration") or info.get("format", {}).get("duration")
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    if duration is None or fps is None:
        raise RuntimeError(f"duration/fps unavailable for: {video_path}")
    return {
        "path": str(video_path),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": round(fps, 3),
        "duration_sec": round(float(duration), 3),
    }


def build_command(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> list[str]:
    """Assemble the normalisation ffmpeg argv (unified encoder, no stream copy)."""
    return [
        ffmpeg_bin, "-y",
        "-i", str(input_path),
        "-vf", f"scale=-2:{height},fps={fps:g}",
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]


def normalize(
    input_path: str,
    output_path: str,
    height: int = 480,
    fps: float = 15.0,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    skip_existing: bool = False,
) -> dict:
    """Probe -> (re-)encode -> probe output -> write ``<output>.info.json``.

    Returns the info dict that was written next to the output. Raises
    RuntimeError (with an ffmpeg/ffprobe stderr tail) on any failure.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.is_file():
        raise RuntimeError(f"input video not found: {input_path}")

    src = probe_video(input_path, ffprobe_bin)
    already = src["height"] <= height and abs(src["fps"] - fps) < 0.01
    if already:
        LOGGER.info("input already within target (height %d<=%d, fps %.3f ~ %.3f);"
                    " re-encoding anyway to guarantee a uniform encoder",
                    src["height"], height, src["fps"], fps)

    cmd = build_command(ffmpeg_bin, input_path, output_path, height, fps, crf, preset)
    skipped = False
    if skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
        LOGGER.info("skip-existing: %s already present; ffmpeg not re-run", output_path)
        skipped = True
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("transcode -> %s (height<=%d, fps=%g, crf=%d, preset=%s)",
                    output_path, height, fps, crf, preset)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}); stderr tail:\n"
                f"{_stderr_tail(proc.stderr)}")
        if not output_path.is_file():
            raise RuntimeError(f"ffmpeg reported success but output missing: {output_path}")

    dst = probe_video(output_path, ffprobe_bin)
    if skipped and (dst["height"] > height or abs(dst["fps"] - fps) >= 0.01):
        LOGGER.warning("existing output does not match target (height %d, fps %.3f);"
                       " delete it or drop --skip-existing to re-encode",
                       dst["height"], dst["fps"])

    info = {
        "input": src,
        "output": dst,
        "target": {"height": height, "fps": fps, "crf": crf, "preset": preset},
        "already_within_target": already,
        "skipped_existing": skipped,
        "command": cmd,
        "command_line": " ".join(cmd),
        "created_at": _now(),
    }
    info_path = Path(str(output_path) + ".info.json")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", info_path)
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="input video (anything ffmpeg can read)")
    p.add_argument("--output", required=True, help="output normalised mp4")
    p.add_argument("--height", type=int, default=480,
                   help="target frame height (width auto-derived, aspect preserved)")
    p.add_argument("--fps", type=float, default=15, help="target frame rate")
    p.add_argument("--crf", type=int, default=20, help="libx264 CRF quality")
    p.add_argument("--preset", default="fast", help="libx264 speed preset")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    p.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse an existing non-empty output instead of re-encoding")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        info = normalize(
            args.input, args.output,
            height=args.height, fps=args.fps, crf=args.crf, preset=args.preset,
            ffmpeg_bin=args.ffmpeg, ffprobe_bin=args.ffprobe,
            skip_existing=args.skip_existing,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps({
        "input": {"width": info["input"]["width"], "height": info["input"]["height"],
                  "fps": info["input"]["fps"], "duration_sec": info["input"]["duration_sec"]},
        "output": {"width": info["output"]["width"], "height": info["output"]["height"],
                   "fps": info["output"]["fps"], "duration_sec": info["output"]["duration_sec"]},
        "already_within_target": info["already_within_target"],
        "skipped_existing": info["skipped_existing"],
        "output_path": info["output"]["path"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
