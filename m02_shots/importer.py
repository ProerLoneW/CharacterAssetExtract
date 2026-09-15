# -*- coding: utf-8 -*-
"""模块②：镜头切片结果接入，建 clip 邻接表骨架。

镜头切分（TransNetV2 切镜 + 25FPS + 音频抽取 + 基础过滤）复用 teleAI-T2V 的
new_data_ppl_v1.0 流水线（cut_v2 / audio / base_filter_v2 模式），本模块只做两件事：

1. 提供两种适配器把切镜产出导入 Schema：
   - manifest：读镜头清单 JSON（含每镜头 start/end，精确帧区间）
   - scandir ：扫描切好镜头的目录，ffprobe 逐个探测，按文件名自然序推回时间轴
2. 建邻接表骨架：首镜头 prev=None/否，末镜头 next=None/否，
   相邻原始镜头 continuous=False（与框架图一致；④ 再切割后才出现"是"）。

teleAI-T2V 输出目录约定（依据 video_full_pipeline.sh）：
  <pkg>/v2.0.0/video/               切镜产物
  <pkg>/v2.0.0/video_25fps/data/    25fps 视频
  <pkg>/v2.0.0/audio/lipsync/       抽出的音频
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List

from ..common.clip_store import rebuild_links
from ..common.ffmpeg import probe_video
from ..common.jsonl import upsert_records
from ..common.workspace import Workspace
from ..schema import ClipRecord

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _fill_probe(rec: ClipRecord, video_path: str, fps_fallback: float = 25.0) -> None:
    p = probe_video(video_path)
    rec.video_path = video_path
    rec.fps = p.fps or fps_fallback
    rec.duration = p.duration
    rec.end_frame = rec.start_frame + (p.nb_frames or int(round(p.duration * rec.fps)))


def from_manifest(video_id: str, manifest_path: str,
                  video_dir: str = "", fps: float = 25.0) -> List[ClipRecord]:
    """从镜头清单导入。清单格式（JSON 列表）：
      [{"shot_id": "001", "start": 0.0, "end": 4.2,
        "start_frame": 0, "end_frame": 105, "video_path": "..."}, ...]
    start/end 为源片秒；缺省帧号时按 fps 换算。
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        shots = json.load(f)
    clips: List[ClipRecord] = []
    for i, s in enumerate(shots):
        rec = ClipRecord(
            clip_id=str(s.get("shot_id") or f"{i + 1:03d}"),
            video_id=video_id,
            fps=fps,
            start_time=float(s.get("start", 0.0)),
            end_time=float(s.get("end", 0.0)),
            start_frame=int(s.get("start_frame", round(float(s.get("start", 0.0)) * fps))),
            end_frame=int(s.get("end_frame", round(float(s.get("end", 0.0)) * fps))),
        )
        rec.duration = rec.end_time - rec.start_time
        vp = s.get("video_path") or ""
        if vp and not os.path.isabs(vp):
            vp = os.path.join(video_dir, vp)
        if vp and os.path.exists(vp):
            _fill_probe(rec, vp, fps)
        clips.append(rec)
    return rebuild_links(clips)


def from_scandir(video_id: str, shot_dir: str, fps: float = 25.0) -> List[ClipRecord]:
    """扫描镜头目录：按文件名自然序排列，ffprobe 探测时长，
    累计偏移推回源片时间轴（无法拿到精确切点时的兜底方案）。"""
    files = sorted(
        (f for f in os.listdir(shot_dir) if f.lower().endswith(VIDEO_EXTS)),
        key=_natural_key,
    )
    clips: List[ClipRecord] = []
    cursor = 0.0
    for i, fname in enumerate(files):
        path = os.path.join(shot_dir, fname)
        rec = ClipRecord(clip_id=f"{i + 1:03d}", video_id=video_id, fps=fps)
        _fill_probe(rec, path, fps)
        rec.start_time = cursor
        rec.end_time = cursor + rec.duration
        rec.start_frame = int(round(rec.start_time * rec.fps))
        rec.end_frame = int(round(rec.end_time * rec.fps))
        cursor = rec.end_time
        clips.append(rec)
    return rebuild_links(clips)


def import_shots(workspace: Workspace, video_id: str,
                 manifest: str = "", shot_dir: str = "",
                 fps: float = 25.0) -> List[ClipRecord]:
    """导入并 upsert 进 workspace 的 clips.jsonl。"""
    if manifest:
        clips = from_manifest(video_id, manifest, fps=fps)
    elif shot_dir:
        clips = from_scandir(video_id, shot_dir, fps=fps)
    else:
        raise ValueError("必须提供 manifest 或 shot_dir 之一")
    upsert_records(workspace.clips_jsonl, ClipRecord, clips, key="clip_id")
    return clips


def main():
    ap = argparse.ArgumentParser(description="模块②：镜头切片导入 + 邻接表骨架")
    ap.add_argument("--workspace", required=True, help="工作区根目录（<root>/<video_id> 的父级）")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--manifest", default="", help="镜头清单 JSON（优先）")
    ap.add_argument("--shot-dir", default="", help="切好镜头的目录（兜底，ffprobe 探测）")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    ws = Workspace(root=os.path.join(args.workspace, args.video_id), video_id=args.video_id)
    clips = import_shots(ws, args.video_id, args.manifest, args.shot_dir, args.fps)
    print(f"导入 {len(clips)} 个镜头，邻接表骨架已写入 {ws.clips_jsonl}")
    for c in clips[:3] + clips[-1:]:
        print(f"  {c.clip_id}: [{c.start_time:.2f},{c.end_time:.2f}]s "
              f"prev={c.prev.clip_id}/{c.prev.continuous} "
              f"next={c.next.clip_id}/{c.next.continuous}")


if __name__ == "__main__":
    main()
