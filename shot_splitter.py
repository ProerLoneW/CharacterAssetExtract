#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shot boundary detection and clip export.

Atomic migration of ``lsj/video_process/scripts/split_shots.py``. The script
reads one or more videos, detects candidate cuts (TransNetV2 with adaptive
PySceneDetect fallback), normalises coverage and duration, and exports each
shot as an mp4 clip plus a JSONL manifest.

Pipeline:

    raw_video.mp4
      -> detector (transnetv2 | adaptive | content)
      -> coverage-aware interval post-processing
      -> optional merge / forced split by min/max duration
      -> ffmpeg clip export
      -> shots.jsonl  (one record per shot)

The TransNetV2 detector code defaults to the vendored, self-contained copy
under ``m02_shots/teleai_src/`` (no teleAI-T2V repo code is imported); the
model weights still default to the upstream teleAI-T2V model directory.
Both can be overridden via CLI flags.

Usage::

    python mimo_shot_splitter.py \
        --input  /path/to/video.mp4 \
        --output-dir /path/to/shot_outputs \
        --detector transnetv2 --transnet-device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import types
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from mimo_caption_client import SOURCE_VIDEO_PROCESS_ROOT

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
# 默认使用本目录下 m02_shots/teleai_src/ 的自包含副本——不 import teleAI-T2V
# 仓库的任何代码（用户要求：切镜链路必须干净）。模型权重仍在 teleAI-T2V
# local_models/（保留上游模型所有权）。两者都可用 CLI flag 覆盖。
DEFAULT_TRANSNET_SOURCE_DIR = Path(__file__).resolve().parent / "m02_shots" / "teleai_src"
DEFAULT_TRANSNET_MODEL = Path(
    "/gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V/local_models/transnetv2/"
    "transnetv2-pytorch-weights.pth"
)
LOGGER = logging.getLogger("mimo_shot_splitter")


@dataclass
class Interval:
    start: float
    end: float
    boundary_type: str = "detected_cut"
    raw_scene_index: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def run_json_command(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe_video(video_path: Path, ffprobe_bin: str) -> dict:
    command = [
        ffprobe_bin, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path),
    ]
    info = run_json_command(command)
    video_stream = next(
        (stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"No video stream found: {video_path}")
    duration = video_stream.get("duration") or info.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Video duration is unavailable: {video_path}")
    fps = parse_rate(video_stream.get("avg_frame_rate")) or parse_rate(video_stream.get("r_frame_rate"))
    if fps is None:
        raise RuntimeError(f"Video frame rate is unavailable: {video_path}")
    return {
        "duration_sec": float(duration),
        "fps": fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in info.get("streams", [])),
    }


# ---------------------------------------------------------------------------
# PySceneDetect detector
# ---------------------------------------------------------------------------
def detect_pyscenedetect_scenes(
    video_path: Path,
    detector_name: str,
    threshold: float,
    detector_min_scene_sec: float,
) -> list[Interval]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is required. Install it with: pip install scenedetect[opencv]"
        ) from exc

    video = open_video(str(video_path))
    fps = float(video.frame_rate)
    manager = SceneManager()
    min_scene_len = max(1, int(round(detector_min_scene_sec * fps)))
    if detector_name == "adaptive":
        manager.add_detector(
            AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)
        )
    else:
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    try:
        manager.detect_scenes(video=video, show_progress=False)
        scene_list = manager.get_scene_list()
    finally:
        close_method = getattr(video, "close", None)
        if callable(close_method):
            close_method()
        else:
            release_method = getattr(video, "release", None)
            if callable(release_method):
                release_method()

    intervals = []
    for index, (start, end) in enumerate(scene_list):
        intervals.append(
            Interval(
                start=float(start.get_seconds()),
                end=float(end.get_seconds()),
                raw_scene_index=index,
            )
        )
    return intervals


# ---------------------------------------------------------------------------
# TransNetV2 backend (loads from teleAI-T2V/new_data_ppl_v1.0/video_split.py)
# ---------------------------------------------------------------------------
def load_teleai_transnet_module(source_dir: Path):
    module_name = "teleai_t2v_video_split"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source_file = source_dir / "video_split.py"
    if not source_file.is_file():
        raise RuntimeError(f"TransNetV2 source file not found: {source_file}")

    try:
        import moviepy.editor  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        editor_module = types.ModuleType("moviepy.editor")
        editor_module.VideoClip = object
        editor_module.VideoFileClip = object
        sys.modules.setdefault("moviepy.editor", editor_module)

    source_dir_text = str(source_dir)
    others_dir_text = str(source_dir / "others")
    for path in (source_dir_text, others_dir_text):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TransNetV2 source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_transnet_device(requested_device: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the TransNetV2 detector.") from exc
    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested TransNetV2 device {requested_device}, but CUDA is unavailable.")
    return requested_device


def predict_transnet_with_progress(pipe, original_predict, video, total_frames: int, video_name: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return original_predict(video)
    predictions = []
    buffers = []
    is_end = False
    previous = 0
    import numpy as np
    with tqdm(total=total_frames, desc=f"TransNet {video_name}", unit="frame", leave=False) as progress:
        while not is_end:
            frames, buffers, is_end = pipe.get_frames(video, buffers)
            if frames is None:
                return []
            predictions.append(pipe.forward_model(frames))
            current = max(0, video.cur_frame_idx + 1)
            progress.update(max(0, current - previous))
            previous = current
        if len(buffers) > 25:
            predictions.append(pipe.forward_model(buffers))
        progress.update(max(0, total_frames - previous))
    predictions = np.concatenate(predictions)
    if len(predictions) != video.cur_frame_idx + 1:
        raise RuntimeError("TransNetV2 returned an inconsistent frame count")
    return predictions


def call_transnet(pipe, video_path: Path, total_frames: int, video_mode: str, show_progress: bool):
    if not show_progress:
        return pipe(str(video_path), video_mode=video_mode)
    original_predict = pipe.predict_video
    pipe.predict_video = lambda video: predict_transnet_with_progress(
        pipe, original_predict, video, total_frames, video_path.name
    )
    try:
        return pipe(str(video_path), video_mode=video_mode)
    finally:
        pipe.predict_video = original_predict


def detect_transnetv2_scenes(
    video_path: Path, fps: float, duration_sec: float, args: argparse.Namespace
) -> list[Interval]:
    model_path = args.transnet_model
    if model_path is None or not model_path.is_file():
        raise RuntimeError(
            "TransNetV2 requires --transnet-model PATH to the checkpoint file."
        )
    module = load_teleai_transnet_module(args.transnet_source_dir)
    if not hasattr(args, "_transnet_pipe"):
        device = resolve_transnet_device(args.transnet_device)
        if device.startswith("cuda"):
            args._transnet_pipe = module.TransNetV2Pipeline(str(model_path)).to(device)
        else:
            original_torch_load = module.torch.load

            def load_on_cpu(*load_args, **load_kwargs):
                load_kwargs.setdefault("map_location", "cpu")
                return original_torch_load(*load_args, **load_kwargs)

            module.torch.load = load_on_cpu
            try:
                args._transnet_pipe = module.TransNetV2Pipeline(str(model_path)).to(device)
            finally:
                module.torch.load = original_torch_load
        args._transnet_device = device
        LOGGER.info("Loaded TransNetV2 on %s from %s", device, model_path)
    pipe = args._transnet_pipe

    last_error: Exception | None = None
    total_frames = max(1, int(round(duration_sec * fps)))
    for video_mode in args.transnet_video_modes:
        try:
            frame_scenes, _ = call_transnet(
                pipe, video_path, total_frames, video_mode, args.progress
            )
            return [
                Interval(
                    start=max(0.0, float(start_frame) / fps),
                    end=max(0.0, float(end_frame + 1) / fps),
                    raw_scene_index=index,
                )
                for index, (start_frame, end_frame) in enumerate(frame_scenes)
            ]
        except Exception as exc:
            last_error = exc
            LOGGER.warning("TransNetV2 reader mode %s failed for %s: %s", video_mode, video_path.name, exc)
    raise RuntimeError(f"TransNetV2 could not read {video_path}: {last_error}")


def detect_raw_scenes(
    video_path: Path, metadata: dict, args: argparse.Namespace
) -> tuple[list[Interval], str]:
    requested = args.detector
    if requested in {"auto", "transnetv2"}:
        can_use_transnet = args.transnet_model is not None and args.transnet_model.is_file()
        if requested == "transnetv2" or can_use_transnet:
            try:
                return detect_transnetv2_scenes(
                    video_path, metadata["fps"], metadata["duration_sec"], args
                ), "transnetv2"
            except Exception:
                if requested == "transnetv2":
                    raise
                LOGGER.warning("TransNetV2 unavailable; falling back to adaptive PySceneDetect.", exc_info=True)
    return (
        detect_pyscenedetect_scenes(
            video_path,
            "adaptive" if requested == "auto" else requested,
            args.threshold,
            args.detector_min_scene_sec,
        ),
        "adaptive" if requested == "auto" else requested,
    )


# ---------------------------------------------------------------------------
# Interval post-processing
# ---------------------------------------------------------------------------
def ensure_coverage(raw_scenes: list[Interval], duration_sec: float) -> list[Interval]:
    """Normalise detector output to a contiguous half-open timeline."""
    if not raw_scenes:
        return [Interval(0.0, duration_sec, "no_detected_cut", None)]
    normalized = []
    cursor = 0.0
    for scene in raw_scenes:
        start = min(max(scene.start, cursor), duration_sec)
        end = min(max(scene.end, start), duration_sec)
        if end - start > 1e-4:
            normalized.append(replace(scene, start=start, end=end))
            cursor = end
    if not normalized:
        return [Interval(0.0, duration_sec, "no_detected_cut", None)]
    if normalized[0].start > 1e-4:
        normalized.insert(0, Interval(0.0, normalized[0].start, "coverage_padding", None))
    if normalized[-1].end < duration_sec - 1e-4:
        normalized.append(Interval(normalized[-1].end, duration_sec, "coverage_padding", None))
    return normalized


def merge_short_scenes(scenes: list[Interval], min_sec: float, max_sec: float | None) -> list[Interval]:
    if min_sec <= 0:
        return scenes
    result: list[Interval] = []
    index = 0
    while index < len(scenes):
        current = scenes[index]
        if current.duration >= min_sec or len(scenes) == 1:
            result.append(current)
            index += 1
            continue

        next_scene = scenes[index + 1] if index + 1 < len(scenes) else None
        previous = result[-1] if result else None
        if next_scene is not None and (max_sec is None or current.duration + next_scene.duration <= max_sec):
            result.append(
                Interval(current.start, next_scene.end, "merged_short_scene", current.raw_scene_index)
            )
            index += 2
        elif previous is not None and (max_sec is None or previous.duration + current.duration <= max_sec):
            result[-1] = replace(previous, end=current.end, boundary_type="merged_short_scene")
            index += 1
        else:
            result.append(replace(current, boundary_type="short_scene"))
            index += 1
    return result


def split_long_scenes(scenes: Iterable[Interval], max_sec: float) -> list[Interval]:
    result: list[Interval] = []
    for scene in scenes:
        if scene.duration <= max_sec + 1e-6:
            result.append(scene)
            continue
        start = scene.start
        while start < scene.end - 1e-6:
            end = min(start + max_sec, scene.end)
            result.append(Interval(start, end, "forced_max_duration", scene.raw_scene_index))
            start = end
    return result


def build_intervals(
    raw_scenes: list[Interval], duration_sec: float, min_sec: float, max_sec: float | None, merge_short: bool
) -> list[Interval]:
    scenes = ensure_coverage(raw_scenes, duration_sec)
    if merge_short:
        scenes = merge_short_scenes(scenes, min_sec, max_sec)
    if max_sec is None:
        return scenes
    return split_long_scenes(scenes, max_sec)


# ---------------------------------------------------------------------------
# ffmpeg clip export
# ---------------------------------------------------------------------------
def export_clip(
    video_path: Path, output_path: Path, start_sec: float, end_sec: float, ffmpeg_bin: str, crf: int
) -> None:
    duration = max(0.001, end_sec - start_sec)
    # Input seeking (-ss before -i): seek to the nearest preceding keyframe and
    # decode forward.  With re-encoding ffmpeg still lands on the exact frame
    # (accurate_seek is the default), but skips decoding the whole movie prefix
    # that output seeking (-ss after -i) would require.
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}",
        "-i", str(video_path),
        "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", "-y", str(output_path),
    ]
    subprocess.run(command, check=True)


def process_video(video_path: Path, args: argparse.Namespace) -> dict:
    metadata = probe_video(video_path, args.ffprobe)
    raw_scenes, actual_detector = detect_raw_scenes(video_path, metadata, args)
    intervals = build_intervals(
        raw_scenes,
        metadata["duration_sec"],
        args.min_shot_sec,
        args.max_shot_sec,
        args.merge_short,
    )

    video_output_dir = args.output_dir.resolve() if args.flat else (args.output_dir / video_path.stem).resolve()
    clip_dir = video_output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    records = []
    export_progress = None
    if not args.no_export and args.progress:
        try:
            from tqdm import tqdm
            export_progress = tqdm(total=len(intervals), desc=f"Export {video_path.name}", unit="shot", leave=False)
        except ImportError:
            pass
    for shot_id, interval in enumerate(intervals, start=1):
        clip_name = f"{video_path.stem}_shot_{shot_id:05d}.mp4"
        clip_path = clip_dir / clip_name
        if not args.no_export:
            export_clip(video_path, clip_path, interval.start, interval.end, args.ffmpeg, args.crf)
        start_frame = max(0, int(round(interval.start * metadata["fps"])))
        end_frame_exclusive = min(
            int(round(metadata["duration_sec"] * metadata["fps"])),
            int(round(interval.end * metadata["fps"])),
        )
        record = {
            "shot_id": shot_id,
            "source_video": str(video_path.resolve()),
            "clip_path": str(clip_path) if not args.no_export else None,
            "time_range_sec": [round(interval.start, 6), round(interval.end, 6)],
            "duration_sec": round(interval.duration, 6),
            "frame_range": [start_frame, max(start_frame, end_frame_exclusive - 1)],
            "fps": metadata["fps"],
            "boundary_type": interval.boundary_type,
            "raw_scene_index": interval.raw_scene_index,
        }
        records.append(record)
        if export_progress is not None:
            export_progress.update(1)
    if export_progress is not None:
        export_progress.close()

    manifest_path = video_output_dir / "shots.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_video": str(video_path),
        "duration_sec": metadata["duration_sec"],
        "fps": metadata["fps"],
        "resolution": [metadata["width"], metadata["height"]],
        "detector": actual_detector,
        "threshold": args.threshold,
        "min_shot_sec": args.min_shot_sec,
        "max_shot_sec": args.max_shot_sec,
        "merge_short": args.merge_short,
        "raw_scene_count": len(raw_scenes),
        "shot_count": len(records),
        "manifest": str(manifest_path),
    }
    (video_output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def find_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and export shot clips with JSONL timestamps.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--flat", action="store_true",
        help="单视频模式：产物直接写入 --output-dir，不再追加 <视频名>/ 子目录",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "transnetv2", "adaptive", "content"],
        default="auto",
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--detector-min-scene-sec", type=float, default=0.5)
    parser.add_argument("--min-shot-sec", type=float, default=2.0)
    parser.add_argument("--max-shot-sec", type=float, default=None)
    parser.add_argument("--merge-short", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--transnet-model", type=Path, default=DEFAULT_TRANSNET_MODEL)
    parser.add_argument("--transnet-source-dir", type=Path, default=DEFAULT_TRANSNET_SOURCE_DIR)
    parser.add_argument("--transnet-device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--transnet-video-modes",
        nargs="+",
        default=["cv2", "decord", "av"],
        choices=["cv2", "decord", "av"],
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_shot_sec <= 0:
        raise SystemExit("Require --min-shot-sec > 0")
    if args.max_shot_sec is not None and args.max_shot_sec < args.min_shot_sec:
        raise SystemExit("Require --max-shot-sec >= --min-shot-sec when --max-shot-sec is set")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    videos = find_videos(args.input, args.recursive)
    if not videos:
        LOGGER.error("No supported videos found under %s", args.input)
        return 2
    if args.flat and len(videos) > 1:
        LOGGER.error("--flat 只支持单视频输入（找到 %d 个）", len(videos))
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for video_path in videos:
        try:
            summary = process_video(video_path, args)
            LOGGER.info("%s: %d shots -> %s", video_path.name, summary["shot_count"], summary["manifest"])
        except Exception:
            failed += 1
            LOGGER.exception("Failed to process %s", video_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())