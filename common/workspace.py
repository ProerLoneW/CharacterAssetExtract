# -*- coding: utf-8 -*-
"""工作区目录约定。

每部源片一个工作目录，全部中间产物落盘其下，支持单步重跑：

  <workspace>/<video_id>/
    source/            源片与 25fps 重采样版
    shots/             ② 镜头切片产物（teleAI-T2V 输出的镜像/清单）
    segments/          ① 一级切分大段视频（~segment_minutes，镜头边界对齐）+ manifest.json
    clips/             最终 clip 宇宙（视频/音频文件）
    faces/             ① 抽帧、人脸、embedding、代表脸图集
    annotations/
      clips.jsonl      clip 级记录（Schema 一级）
      segments.jsonl   segment 级记录（Schema 二级）
      global.jsonl     global 级记录（Schema 三级，单行）
      qc_report.json   质检汇总
    logs/
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Workspace:
    root: str          # <workspace>/<video_id>
    video_id: str = ""

    def __post_init__(self):
        if not self.video_id:
            self.video_id = os.path.basename(os.path.abspath(self.root))

    def path(self, *parts: str) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    @property
    def clips_jsonl(self) -> str:
        return self.path("annotations", "clips.jsonl")

    @property
    def segments_jsonl(self) -> str:
        return self.path("annotations", "segments.jsonl")

    @property
    def global_jsonl(self) -> str:
        return self.path("annotations", "global.jsonl")

    @property
    def source_dir(self) -> str:
        return self.path("source", ".keep")

    @property
    def clips_dir(self) -> str:
        return self.path("clips", ".keep")

    @property
    def faces_dir(self) -> str:
        return self.path("faces", ".keep")

    @property
    def shots_dir(self) -> str:
        return self.path("shots", ".keep")

    @property
    def logs_dir(self) -> str:
        return os.path.dirname(self.path("logs", ".keep"))

    def clip_video_path(self, clip_id: str) -> str:
        return self.path("clips", "video", f"{clip_id}.mp4")

    def clip_audio_path(self, clip_id: str) -> str:
        return self.path("clips", "audio", f"{clip_id}.wav")

    def clip_thumb_path(self, clip_id: str) -> str:
        return self.path("clips", "thumb", f"{clip_id}.jpg")
