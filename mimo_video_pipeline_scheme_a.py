#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme-A caption-only pipeline entry point.

Differences from ``mimo_video_pipeline.py``:
  * global-caption also attributes authoritative ASR to registered S-IDs;
  * segment-caption consumes those global judgments read-only;
  * shot and segment semantic caption calls remain, preserving caption quality;
  * upstream shot/segment/global structure is consumed, never planned by MiMo.

Use a distinct ``--work-dir`` to keep Scheme-A outputs separate from legacy
pipeline outputs. ``--stage all`` means only global, shot, and fixed-segment
captioning; structural preprocessing belongs to the upstream pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mimo_video_pipeline as base

HERE = Path(__file__).resolve().parent
SHOT_PROMPT_V5 = HERE / "prompts" / "shot_caption_prompt_en_semantic_roles_objects_v5.md"


def stage_global_caption(args, work_dir: Path) -> int:
    if not args.asr_input:
        raise SystemExit("Scheme A requires --asr-input for global-caption")
    target = args.global_input or str(work_dir / "global" / Path(args.input).stem)
    out_dir = work_dir / "annotations" / "global"
    manifest = out_dir / Path(target.rstrip("/")).name / "global_captions.jsonl"
    if base._captions_all_success(manifest) and not args.force:
        base.LOGGER.info("skip global-caption: %s exists and all records succeeded", manifest)
        return 0
    cmd = [
        sys.executable, str(HERE / "mimo_global_captioner_speaker_v2.py"),
        "--input", target,
        "--output-dir", str(out_dir),
        "--asr-input", args.asr_input,
        "--asr-timebase", args.asr_timebase,
        "--concurrency", str(args.global_workers),
        "--retries", str(args.retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        "--max-completion-tokens", "32768",
        "--thinking" if args.thinking else "--no-thinking",
    ]
    if not args.force:
        cmd.append("--skip-existing")
    return base.run(cmd)


def stage_shot_caption(args, work_dir: Path) -> int:
    shot_input = args.shot_input or str(work_dir / "shots" / Path(args.input).stem)
    global_input = args.global_input or str(
        work_dir / "annotations" / "global" / Path(args.input).stem
    )
    out_dir = work_dir / "annotations" / "shot"
    cmd = [
        sys.executable, str(HERE / "mimo_shot_captioner.py"),
        "--input", shot_input,
        "--global-input", global_input,
        "--output-dir", str(out_dir),
        "--prompt-file", str(SHOT_PROMPT_V5),
        "--concurrency", str(args.shot_workers),
        "--retries", str(args.retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        "--max-completion-tokens", "8192",
        "--thinking" if args.thinking else "--no-thinking",
    ]
    if not args.force:
        cmd.append("--skip-existing")
    return base.run(cmd)


def stage_segment_caption(args, work_dir: Path) -> int:
    if not args.segment_input:
        raise SystemExit(
            "Scheme A fixed-segment captioning requires --segment-input "
            "(the upstream segments.jsonl)"
        )
    global_input = args.global_input or str(
        work_dir / "annotations" / "global" / Path(args.input).stem
    )
    shot_input = args.shot_input or str(
        work_dir / "annotations" / "shot" / Path(args.input).stem
    )
    out_dir = work_dir / "annotations" / "segment"
    cmd = [
        sys.executable, str(HERE / "mimo_segment_captioner_fixed_segments_v4.py"),
        "--global-input", global_input,
        "--shot-input", shot_input,
        "--segment-input", args.segment_input,
        "--output-dir", str(out_dir),
        "--concurrency", str(args.segment_workers),
        "--retries", str(args.segment_retries),
        "--request-format", args.request_format,
        "--timeout", str(args.request_timeout),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--thinking" if args.thinking else "--no-thinking",
    ]
    if args.segment_text_only:
        cmd.append("--text-only")
    if not args.force:
        cmd.append("--skip-existing")
    return base.run(cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scheme-A fixed-structure three-level caption pipeline."
    )
    parser.add_argument("--stage", default="all", choices=base.STAGE_CHOICES)
    parser.add_argument("--input", help="Source video path; also derives the video stem.")
    parser.add_argument("--global-input", help="Override global clip/caption input path.")
    parser.add_argument("--shot-input", help="Override shot clip/caption input path.")
    parser.add_argument("--segment-input", help="Authoritative upstream segments.jsonl.")
    parser.add_argument("--work-dir")
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
    parser.add_argument("--asr-input", help="Authoritative ASR JSON/JSONL path.")
    parser.add_argument("--asr-timebase", choices=["auto", "source", "global-local"], default="auto")
    parser.add_argument("--segment-text-only", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--segment-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--request-format", choices=["video_url", "input_video", "auto"], default="video_url")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    if args.input is None and args.global_input is None:
        raise SystemExit("--input is required (or pass --global-input for caption-only runs)")
    stem = Path(args.input).stem if args.input else Path(args.global_input.rstrip("/")).name
    work_dir = Path(args.work_dir) if args.work_dir else HERE / "pipeline_outputs" / stem
    work_dir.mkdir(parents=True, exist_ok=True)
    base.LOGGER.info("work dir: %s", work_dir)

    # In Scheme A V2, `all` deliberately excludes split/aggregation. Those
    # structures must already have been produced without an LLM upstream.
    stages = [args.stage] if args.stage != "all" else [
        "global-caption", "shot-caption", "segment-caption",
    ]
    handlers = {
        "split-shots": base.stage_split_shots,
        "aggregate-global": base.stage_aggregate_global,
        "global-caption": stage_global_caption,
        "shot-caption": stage_shot_caption,
        "segment-caption": stage_segment_caption,
        "cut-segments": base.stage_cut_segments,
    }
    for stage in stages:
        try:
            return_code = handlers[stage](args, work_dir)
        except Exception:
            base.LOGGER.exception("stage %s crashed", stage)
            return_code = 2
        if return_code:
            base.LOGGER.error("Stopping after failed stage %s", stage)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
