#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-file orchestration of the MiMo three-level caption pipeline.

Atomic migration of ``lsj/video_process/bin/run_video_pipeline.sh``. The
script does not re-implement the heavy lifting; it simply invokes the five
atomic modules in order:

    1. :mod:`shot_splitter`              split_shots
    2. :mod:`global_clip_aggregator`     aggregate_global_clips
    3. :mod:`mimo_global_captioner`       global caption
    4. :mod:`mimo_shot_captioner`         shot caption
    5. :mod:`mimo_segment_captioner`      segment caption
    6. :mod:`mimo_segment_cutter`         cut final segment videos

Existing outputs are reused unless ``--force`` is passed; any individual
stage can be selected with ``--stage``.  Equivalent to the upstream bash
wrapper, but portable and inspectable from Python.

工作区采用扁平布局（各原子模块以 --flat 调用，不追加 <视频名>/ 子层）：

    <work_dir>/
    ├── shots/      shots.jsonl + clips/*.mp4 + summary.json
    ├── global/     global_clips.jsonl + global_clips/*.mp4 + summary.json
    ├── captions/   global/ shot/ segment/ 三级 caption 产物
    └── segments/   segments.jsonl + clips/*.mp4 + summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from mimo_caption_client import SOURCE_VIDEO_PROCESS_ROOT

LOGGER = logging.getLogger("mimo_video_pipeline")
HERE = Path(__file__).resolve().parent

STAGE_CHOICES = [
    "split-shots",
    "aggregate-global",
    "global-caption",
    "shot-caption",
    "segment-caption",
    "cut-segments",
    "all",
]


def run(cmd: list[str]) -> int:
    LOGGER.info("→ %s", " ".join(cmd))
    return subprocess.call(cmd)


def _captions_all_success(manifest: Path) -> bool:
    """True only if the caption jsonl exists and every record succeeded.

    A present-but-failed jsonl (truncated / invalid JSON) must NOT suppress a
    re-run, otherwise the captioner's success-gated --skip-existing retry is
    never reached and the bad caption is pinned forever.
    """
    if not manifest.is_file():
        return False
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
        count += 1
        if not (isinstance(record, dict) and record.get("success") is True):
            return False
    return count > 0


def stage_split_shots(args, work_dir: Path) -> int:
    if not args.input:
        raise SystemExit("--input is required for split-shots")
    shot_root = work_dir / "shots"
    manifest = shot_root / "shots.jsonl"
    if manifest.is_file() and not args.force:
        LOGGER.info("skip split-shots: %s exists", manifest)
        return 0
    cmd = [
        sys.executable, str(HERE / "shot_splitter.py"),
        "--input", args.input,
        "--output-dir", str(shot_root),
        "--detector", args.detector,
        "--transnet-device", args.transnet_device,
        "--flat",
    ]
    return run(cmd)


def stage_aggregate_global(args, work_dir: Path) -> int:
    shot_input = args.shot_input or str(work_dir / "shots")
    global_root = work_dir / "global"
    manifest = global_root / "global_clips.jsonl"
    if manifest.is_file() and not args.force:
        LOGGER.info("skip aggregate-global: %s exists", manifest)
        return 0
    cmd = [
        sys.executable, str(HERE / "global_clip_aggregator.py"),
        "--input", shot_input,
        "--output-dir", str(global_root),
        "--min-sec", str(args.min_global_sec),
        "--target-sec", str(args.target_global_sec),
        "--max-sec", str(args.max_global_sec),
        "--scene-detector", args.scene_detector,
        "--scrl-device", args.scrl_device,
        "--flat",
    ]
    return run(cmd)


def stage_global_caption(args, work_dir: Path) -> int:
    target = args.global_input or str(work_dir / "global")
    out_dir = work_dir / "captions" / "global"
    manifest = out_dir / "global_captions.jsonl"
    if _captions_all_success(manifest) and not args.force:
        LOGGER.info("skip global-caption: %s exists and all records succeeded", manifest)
        return 0
    cmd = [
        sys.executable, str(HERE / "mimo_global_captioner.py"),
        "--input", target,
        "--output-dir", str(out_dir),
        "--concurrency", str(args.global_workers),
        "--retries", str(args.retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        # MiMo's global caption JSON for long videos exceeds 8k tokens and was
        # being truncated mid-object, which failed every retry as invalid JSON.
        "--max-completion-tokens", "32768",
        "--flat",
    ]
    if args.thinking:
        cmd.append("--thinking")
    else:
        cmd.append("--no-thinking")
    if not args.force:
        cmd.append("--skip-existing")
    return run(cmd)


def stage_shot_caption(args, work_dir: Path) -> int:
    shot_input = args.shot_input or str(work_dir / "shots")
    global_input = args.global_input or str(work_dir / "captions" / "global")
    out_dir = work_dir / "captions" / "shot"
    cmd = [
        sys.executable, str(HERE / "mimo_shot_captioner.py"),
        "--input", shot_input,
        "--global-input", global_input,
        "--output-dir", str(out_dir),
        "--concurrency", str(args.shot_workers),
        "--retries", str(args.retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        "--max-completion-tokens", "8192",
        "--flat",
    ]
    if args.thinking:
        cmd.append("--thinking")
    else:
        cmd.append("--no-thinking")
    if not args.force:
        cmd.append("--skip-existing")
    return run(cmd)


def stage_segment_caption(args, work_dir: Path) -> int:
    global_input = args.global_input or str(work_dir / "captions" / "global")
    shot_input = args.shot_input or str(work_dir / "captions" / "shot")
    out_dir = work_dir / "captions" / "segment"
    cmd = [
        sys.executable, str(HERE / "mimo_segment_captioner_v2.py"),
        "--global-input", global_input,
        "--shot-input", shot_input,
        "--output-dir", str(out_dir),
        "--concurrency", str(args.segment_workers),
        "--retries", str(args.segment_retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--flat",
    ]
    if getattr(args, "asr_input", None):
        cmd.extend(["--asr-input", args.asr_input, "--asr-timebase", args.asr_timebase])
    if args.segment_text_only:
        cmd.append("--text-only")
    if args.thinking:
        cmd.append("--thinking")
    else:
        cmd.append("--no-thinking")
    if not args.force:
        cmd.append("--skip-existing")
    return run(cmd)


def stage_cut_segments(args, work_dir: Path) -> int:
    segment_input = args.global_input or str(work_dir / "captions" / "segment")
    out_dir = work_dir / "segments"
    cmd = [
        sys.executable, str(HERE / "mimo_segment_cutter.py"),
        "--input", segment_input,
        "--output-dir", str(out_dir),
        "--flat",
    ]
    return run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the MiMo three-level caption pipeline.")
    parser.add_argument("--stage", default="all", choices=STAGE_CHOICES)
    parser.add_argument("--input", help="Source video path; also used to derive video stem.")
    parser.add_argument("--global-input", help="Override global input path.")
    parser.add_argument("--shot-input", help="Override shot input path.")
    parser.add_argument("--work-dir", help="Pipeline work root (default: ./pipeline_outputs/<stem>).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--detector", choices=["auto", "transnetv2", "adaptive", "content"], default="auto")
    parser.add_argument("--transnet-device", default="auto")
    parser.add_argument("--scene-detector", choices=["auto", "scrl", "metadata", "none"], default="scrl")
    parser.add_argument("--scrl-device", default="auto")
    parser.add_argument("--min-global-sec", type=float, default=60.0)
    parser.add_argument("--target-global-sec", type=float, default=90.0)
    parser.add_argument("--max-global-sec", type=float, default=120.0)
    parser.add_argument("--global-workers", type=int, default=2)
    parser.add_argument("--shot-workers", type=int, default=4)
    parser.add_argument("--segment-workers", type=int, default=1)
    parser.add_argument("--asr-input",
                        help="ASR JSON/JSONL file or directory passed to the segment-caption stage")
    parser.add_argument("--asr-timebase", choices=["auto", "source", "global-local"], default="auto")
    parser.add_argument("--segment-text-only", action="store_true",
                        help="Aggregate complete global/shot captions without re-uploading video.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--segment-retries", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=16384,
                        help="Max completion tokens for the segment-caption stage")
    parser.add_argument("--request-format", choices=["video_url", "input_video", "auto"], default="video_url")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    if shutil.which("ffmpeg") is None:
        LOGGER.warning("ffmpeg not found in PATH; stage commands will fail.")

    if args.input is None and args.global_input is None:
        raise SystemExit("--input is required (or pass --global-input for annotation-only runs)")

    stem = Path(args.input).stem if args.input else Path(args.global_input.rstrip("/")).name
    work_dir = Path(args.work_dir) if args.work_dir else HERE / "pipeline_outputs" / stem
    work_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("work dir: %s", work_dir)

    stages = [args.stage] if args.stage != "all" else [
        "split-shots", "aggregate-global",
        "global-caption", "shot-caption", "segment-caption", "cut-segments",
    ]
    handlers = {
        "split-shots": stage_split_shots,
        "aggregate-global": stage_aggregate_global,
        "global-caption": stage_global_caption,
        "shot-caption": stage_shot_caption,
        "segment-caption": stage_segment_caption,
        "cut-segments": stage_cut_segments,
    }
    failed = 0
    for stage in stages:
        if stage not in handlers:
            raise SystemExit(f"Unknown stage: {stage}")
        try:
            rc = handlers[stage](args, work_dir)
        except Exception:
            LOGGER.exception("stage %s crashed", stage)
            rc = 2
        failed += int(rc != 0)
        if rc != 0:
            LOGGER.error("Stopping after failed stage %s; dependent stages were not run", stage)
            break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
