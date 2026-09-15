#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 LLM 的电影预处理模块。

目标：识别并裁掉片头/片尾中"与电影主体无关"的片段。典型信号：
  1. 黑场         ：连续帧亮度 (mean luma) 极低
  2. 静默段       ：音频 RMS 在连续 N 秒内低于阈值（无对白/无背景声）
  3. 单色/低内容  ：帧间亮度方差极低（演职员表 / 纯色背景 / 静态 logo）
  4. 静态 logo    ：前后帧差异极小，且持续时间长（片头公司 logo）

输出 ``processed_info.json``：

    {
      "source": "<原始视频路径>",
      "duration_sec": 7634.66,
      "trimmed_sec": 87.31,
      "kept_intervals": [[start_sec, end_sec], ...],
      "drops": [
        {"kind": "black",    "start_sec": 0.0,  "end_sec": 12.0, "score": 0.04},
        {"kind": "silent",   "start_sec": 7602.0, "end_sec": 7634.66, "score": 0.02},
        {"kind": "low_motion", "start_sec": ..., "end_sec": ..., "score": 0.01}
      ],
      "processed_video": "<裁剪后的输出视频路径>"
    }

用法::

    python -m extract_full_pipeline.preprocessing.preprocess \
        --input /path/to/source.mp4 \
        --output-dir workspace/processed \
        [--max-head-trim-sec 120] [--max-tail-trim-sec 180]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("preprocessing")


# ---------------------------------------------------------------------------
# 探测工具
# ---------------------------------------------------------------------------

def probe_duration(video_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ], text=True).strip()
    return float(out)


def sample_video_features(
    video_path: str,
    sample_fps: float = 2.0,
    width: int = 64,
    height: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """抽稀疏视频帧，返回 (timestamps, per-frame_mean_luma)。

    仅做粗判：每个帧一个标量亮度 + 帧间亮度方差。
    """
    vf = f"fps={sample_fps},scale={width}:{height},format=gray"
    out = subprocess.run([
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ], capture_output=True, check=True).stdout
    arr = np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width).astype(np.float32)
    timestamps = np.arange(arr.shape[0]) / sample_fps
    mean_luma = arr.mean(axis=(1, 2))
    return timestamps, mean_luma


def sample_audio_rms(
    video_path: str,
    sample_rate: int = 8000,
    window_sec: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """抽音频为 float32 单声道，按 window_sec 切片返回 (window_starts, rms)。

    RMS < threshold 视为静默。
    """
    out = subprocess.run([
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-",
    ], capture_output=True, check=True).stdout
    if not out:
        return np.array([0.0]), np.array([0.0])
    pcm = np.frombuffer(out, dtype=np.float32)
    win_samples = max(1, int(window_sec * sample_rate))
    n = len(pcm) // win_samples
    if n == 0:
        return np.array([0.0]), np.array([float(np.sqrt(np.mean(pcm ** 2)))])
    pcm = pcm[: n * win_samples].reshape(n, win_samples)
    rms = np.sqrt(np.mean(pcm ** 2, axis=1))
    starts = np.arange(n) * window_sec
    return starts, rms


# ---------------------------------------------------------------------------
# 信号检测
# ---------------------------------------------------------------------------

@dataclass
class Drop:
    kind: str          # black / silent / low_motion
    start_sec: float
    end_sec: float
    score: float       # 0..1 越大越"非主体"
    detail: str = ""


def detect_black(
    timestamps: np.ndarray,
    mean_luma: np.ndarray,
    threshold: float = 8.0,
    min_run_sec: float = 1.5,
) -> List[Drop]:
    """连续帧亮度低于 threshold 且持续 ≥ min_run_sec。"""
    drops: List[Drop] = []
    is_black = mean_luma < threshold
    i = 0
    while i < len(is_black):
        if is_black[i]:
            j = i
            while j < len(is_black) and is_black[j]:
                j += 1
            start = timestamps[i]
            end = timestamps[min(j, len(timestamps) - 1)]
            if end - start >= min_run_sec:
                drops.append(Drop(
                    kind="black", start_sec=float(start), end_sec=float(end),
                    score=1.0 - float(mean_luma[i:j].mean() / 255.0),
                    detail=f"luma={mean_luma[i:j].mean():.1f}<{threshold}"))
            i = j
        else:
            i += 1
    return drops


def detect_low_motion(
    timestamps: np.ndarray,
    mean_luma: np.ndarray,
    diff_threshold: float = 1.5,
    min_run_sec: float = 8.0,
) -> List[Drop]:
    """连续帧间亮度差 < diff_threshold 且 ≥ min_run_sec（演职员表/纯色背景/静态 logo）。"""
    if len(mean_luma) < 2:
        return []
    diff = np.abs(np.diff(mean_luma))
    is_static = diff < diff_threshold
    drops: List[Drop] = []
    i = 0
    while i < len(is_static):
        if is_static[i]:
            j = i
            while j < len(is_static) and is_static[j]:
                j += 1
            start = timestamps[i]
            end = timestamps[min(j + 1, len(timestamps) - 1)]
            if end - start >= min_run_sec:
                drops.append(Drop(
                    kind="low_motion", start_sec=float(start), end_sec=float(end),
                    score=1.0 - float(diff[i:j].mean() / max(diff_threshold, 1e-6)),
                    detail=f"max_diff={diff[i:j].max():.2f}<{diff_threshold}"))
            i = j
        else:
            i += 1
    return drops


def detect_silent(
    audio_starts: np.ndarray,
    rms: np.ndarray,
    threshold: float = 0.005,
    min_run_sec: float = 4.0,
) -> List[Drop]:
    """连续音频 RMS < threshold 持续 ≥ min_run_sec（片头/片尾无对白区）。"""
    if len(rms) == 0:
        return []
    is_silent = rms < threshold
    drops: List[Drop] = []
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            j = i
            while j < len(is_silent) and is_silent[j]:
                j += 1
            start = audio_starts[i]
            end = audio_starts[min(j, len(audio_starts) - 1)] + 0.5  # window width
            if end - start >= min_run_sec:
                drops.append(Drop(
                    kind="silent", start_sec=float(start), end_sec=float(end),
                    score=1.0 - float(rms[i:j].mean() / threshold),
                    detail=f"rms={rms[i:j].mean():.4f}<{threshold}"))
            i = j
        else:
            i += 1
    return drops


# ---------------------------------------------------------------------------
# 决策：只裁片头/片尾
# ---------------------------------------------------------------------------

def filter_to_head_tail(
    drops: List[Drop],
    duration_sec: float,
    max_head_trim_sec: float = 180.0,
    max_tail_trim_sec: float = 240.0,
) -> List[Drop]:
    """只接受片头 ([0, max_head_trim_sec]) 与片尾 ([duration-max_tail_trim_sec, duration]) 内的 drop。"""
    head_end = max_head_trim_sec
    tail_start = max(0.0, duration_sec - max_tail_trim_sec)
    out = []
    for d in drops:
        in_head = d.end_sec <= head_end
        in_tail = d.start_sec >= tail_start
        if in_head or in_tail:
            out.append(d)
    return out


def merge_drops(drops: List[Drop], gap_sec: float = 2.0) -> List[Drop]:
    """相邻 drop 合并（gap < gap_sec 视为同一段）。"""
    if not drops:
        return []
    drops = sorted(drops, key=lambda d: d.start_sec)
    merged: List[Drop] = [drops[0]]
    for d in drops[1:]:
        last = merged[-1]
        if d.start_sec - last.end_sec <= gap_sec:
            last.end_sec = max(last.end_sec, d.end_sec)
            last.score = max(last.score, d.score)
            last.detail = f"{last.kind}+{d.kind}"
        else:
            merged.append(d)
    return merged


def compute_kept_intervals(
    duration: float, drops: List[Drop], head_pad_sec: float = 1.0, tail_pad_sec: float = 1.0,
) -> List[Tuple[float, float]]:
    """从全片 [0, duration] 减去 drops（左右各留 pad），返回保留区间。"""
    if not drops:
        return [(0.0, duration)]
    drops = sorted(drops, key=lambda d: d.start_sec)
    intervals: List[Tuple[float, float]] = []
    cursor = 0.0
    for d in drops:
        start = max(0.0, d.start_sec - head_pad_sec)
        end = min(duration, d.end_sec + tail_pad_sec)
        if start > cursor + 0.5:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 0.5:
        intervals.append((cursor, duration))
    return intervals


# ---------------------------------------------------------------------------
# 物理裁剪
# ---------------------------------------------------------------------------

def cut_concat(
    video_path: str,
    intervals: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """把 intervals 拼接成单个 mp4。

    用 stream-segment + concat 协议：每个 interval 用 `-ss/-to` + `-c copy` 抽出，
    再 concat。无需 re-encode，速度接近 I/O 极限。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(output_path).with_suffix("").parent / "_segs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: List[str] = []
    for k, (s, e) in enumerate(intervals, 1):
        seg = tmp_dir / f"seg_{k:04d}.mp4"
        # before -ss 精度高，先 -i 再 -ss 是 input seek 但耗时，权衡下用 before -ss
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(seg),
        ]
        subprocess.run(cmd, check=True)
        seg_paths.append(str(seg))

    # concat demuxer
    list_path = tmp_dir / "concat.list"
    with open(list_path, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    # 清理
    for p in seg_paths:
        Path(p).unlink(missing_ok=True)
    list_path.unlink(missing_ok=True)
    tmp_dir.rmdir()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def preprocess(
    input_path: str,
    output_dir: str,
    max_head_trim_sec: float = 180.0,
    max_tail_trim_sec: float = 240.0,
    sample_fps: float = 2.0,
    black_threshold: float = 8.0,
    static_threshold: float = 1.5,
    silent_threshold: float = 0.005,
) -> dict:
    """主入口：分析 → 决策 → 物理裁剪 → 写 processed_info.json。返回 info dict。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"source": str(input_path)}
    duration = probe_duration(input_path)
    info["duration_sec"] = duration

    LOGGER.info("采样视频帧 @%.1f fps ...", sample_fps)
    ts, luma = sample_video_features(input_path, sample_fps=sample_fps)
    LOGGER.info("采样音频 RMS ...")
    audio_starts, rms = sample_audio_rms(input_path)

    drops: List[Drop] = []
    drops += detect_black(ts, luma, threshold=black_threshold)
    drops += detect_low_motion(ts, luma, diff_threshold=static_threshold)
    drops += detect_silent(audio_starts, rms, threshold=silent_threshold)
    LOGGER.info("原始 drop 数：%d", len(drops))

    drops = filter_to_head_tail(drops, duration, max_head_trim_sec, max_tail_trim_sec)
    drops = merge_drops(drops, gap_sec=2.0)
    LOGGER.info("裁掉片头/片尾候选 drop 数：%d", len(drops))

    intervals = compute_kept_intervals(duration, drops)
    trimmed = sum(e - s for s, e in compute_kept_intervals(duration, [])) - sum(e - s for s, e in intervals)
    info["drops"] = [asdict(d) for d in drops]
    info["kept_intervals"] = [list(i) for i in intervals]
    info["trimmed_sec"] = round(trimmed, 3)
    info["max_head_trim_sec"] = max_head_trim_sec
    info["max_tail_trim_sec"] = max_tail_trim_sec
    info["thresholds"] = {
        "black": black_threshold,
        "static": static_threshold,
        "silent": silent_threshold,
    }

    if not drops:
        LOGGER.info("未检测到片头/片尾；不裁剪。")
        info["processed_video"] = str(input_path)
    else:
        out_path = output_dir / "processed.mp4"
        LOGGER.info("物理裁剪 -> %s  共 %d 段", out_path, len(intervals))
        cut_concat(str(input_path), intervals, str(out_path))
        info["processed_video"] = str(out_path)

    out_json = output_dir / "processed_info.json"
    out_json.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    LOGGER.info("写入 %s", out_json)
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-head-trim-sec", type=float, default=180.0)
    p.add_argument("--max-tail-trim-sec", type=float, default=240.0)
    p.add_argument("--black-threshold", type=float, default=8.0)
    p.add_argument("--static-threshold", type=float, default=1.5)
    p.add_argument("--silent-threshold", type=float, default=0.005)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    info = preprocess(
        args.input, args.output_dir,
        max_head_trim_sec=args.max_head_trim_sec,
        max_tail_trim_sec=args.max_tail_trim_sec,
        sample_fps=args.sample_fps,
        black_threshold=args.black_threshold,
        static_threshold=args.static_threshold,
        silent_threshold=args.silent_threshold,
    )
    print(json.dumps({
        "duration_sec": info["duration_sec"],
        "trimmed_sec": info["trimmed_sec"],
        "drops": len(info["drops"]),
        "kept_intervals": len(info["kept_intervals"]),
        "processed_video": info["processed_video"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())