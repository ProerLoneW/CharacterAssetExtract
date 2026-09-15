# -*- coding: utf-8 -*-
"""ffmpeg / ffprobe 封装（各模块共用）。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class VideoProbe:
    duration: float = 0.0     # 秒
    nb_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class AudioProbe:
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0


def _run(cmd: list) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _parse_fps(rate: str) -> float:
    try:
        num, den = rate.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except Exception:
        return 0.0


def probe_video(path: str) -> VideoProbe:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json", path,
    ])
    data = json.loads(out)
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return VideoProbe(
        duration=float(fmt.get("duration", 0.0)),
        nb_frames=int(st.get("nb_frames", 0) or 0),
        fps=_parse_fps(st.get("r_frame_rate", "0/1")),
        width=int(st.get("width", 0) or 0),
        height=int(st.get("height", 0) or 0),
    )


def probe_audio(path: str) -> AudioProbe:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels:format=duration",
        "-of", "json", path,
    ])
    data = json.loads(out)
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return AudioProbe(
        duration=float(fmt.get("duration", 0.0)),
        sample_rate=int(st.get("sample_rate", 0) or 0),
        channels=int(st.get("channels", 0) or 0),
    )


def cut_video(src: str, dst: str, start: float, end: float,
              fps: int = 25, overlap: float = 0.0) -> None:
    """按秒区间切视频并重采样到统一 fps。overlap 为切点两侧余量（秒）。"""
    s = max(0.0, start - overlap)
    e = end + overlap
    _run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
        "-i", src, "-r", str(fps), "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", dst,
    ])


def extract_audio_48k(src: str, dst: str, start: float = 0.0, end: float = 0.0,
                      sr: int = 48000) -> None:
    """抽音频并统一 48kHz 单声道 wav。"""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if end > start:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    cmd += ["-i", src, "-vn", "-ar", str(sr), "-ac", "1", dst]
    _run(cmd)


def extract_frame(src: str, dst: str, ts: float) -> None:
    """抽指定时间点的单帧。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", src,
        "-frames:v", "1", "-q:v", "2", dst,
    ])


def read_gray_frames(src: str, fps: float = 1.0, width: int = 64, height: int = 64,
                     max_frames: int = 0):
    """抽灰度帧序列为 numpy 数组 (N, H, W) uint8。供模糊/光流等轻量分析用。"""
    import numpy as np
    vf = f"fps={fps},scale={width}:{height},format=gray"
    out = _run_bytes([
        "ffmpeg", "-v", "error", "-i", src, "-vf", vf, "-f", "rawvideo", "-",
    ])
    frame_size = width * height
    n = len(out) // frame_size
    if max_frames:
        n = min(n, max_frames)
    arr = np.frombuffer(out[: n * frame_size], dtype=np.uint8)
    return arr.reshape(n, height, width)


def read_rgb_frames(src: str, fps: float = 1.0, width: int = 518, height: int = 518,
                    max_frames: int = 0):
    """抽 RGB 帧序列为 numpy 数组 (N, H, W, 3) uint8。供 DINOv2 等视觉模型用。"""
    import numpy as np
    vf = f"fps={fps},scale={width}:{height},format=rgb24"
    out = _run_bytes([
        "ffmpeg", "-v", "error", "-i", src, "-vf", vf, "-f", "rawvideo", "-",
    ])
    frame_size = width * height * 3
    n = len(out) // frame_size
    if max_frames:
        n = min(n, max_frames)
    arr = np.frombuffer(out[: n * frame_size], dtype=np.uint8)
    return arr.reshape(n, height, width, 3)


def read_pcm(src: str, sr: int = 16000, start: float = 0.0, end: float = 0.0):
    """读音频为 float32 单声道 numpy 数组（范围 [-1,1]）。"""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error"]
    if end > start:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    cmd += ["-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = _run_bytes(cmd)
    return np.frombuffer(out, dtype=np.float32), sr


def _run_bytes(cmd: list) -> bytes:
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def detect_silence(src: str, noise_db: float = -35.0, min_duration: float = 0.3):
    """ffmpeg silencedetect → 静音区间列表 [(start, end), ...]（秒）。"""
    proc = subprocess.run([
        "ffmpeg", "-v", "info", "-i", src,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ], capture_output=True, text=True)
    import re
    intervals, cur = [], None
    for line in proc.stderr.splitlines():
        m1 = re.search(r"silence_start: ([\d.]+)", line)
        m2 = re.search(r"silence_end: ([\d.]+)", line)
        if m1:
            cur = float(m1.group(1))
        elif m2 and cur is not None:
            intervals.append((cur, float(m2.group(1))))
            cur = None
    return intervals
