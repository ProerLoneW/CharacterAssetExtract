#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""teleAI-T2V cut_only 入口（本地副本，**不**依赖 teleAI-T2V 仓库）。

直接调 main_split_fast（**本地**版本，比仓库的 main_split_only 更快——
不做每个 clip 的二次 TransNetV2 精修，只 cut + 保留全部 valid_range）。

输入：
  --config    config JSON（与 teleAI 的 config_test.json 兼容）
  --input-dir 源视频目录（递归找 *.mp4）
  --output-dir 数据包目录（与 teleAI 内部约定一致：<output>/v2.0.0/video/data/）

输出：
  - <output>/v2.0.0/video/data/<video>_<shot_id>.mp4
  - <output>/v2.0.0/result/<video>_labels.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "others"))

import numpy as np  # noqa: E402

from video_split import TransNetV2Pipeline, save_cut_with_timeout  # noqa: E402
from video_utils import get_video_length, get_video_size  # noqa: E402
from moviepy.editor import VideoFileClip  # noqa: E402

LOGGER = logging.getLogger("teleai_cut_only")

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv")


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def main_split_fast(
    pipe: TransNetV2Pipeline,
    video_path: str,
    output_folder: str,
    min_length_frames: int = 10,
) -> dict:
    """仅一次 TransNetV2 切镜 + 直接保留所有切镜区间（不做二次精修）。"""
    min_length = int(os.environ.get("SplitMinLength", min_length_frames))
    video_fps = VideoFileClip(video_path, audio=False).fps
    video_height, video_width = get_video_size(video_path)

    # Try different video modes for the source video
    video_modes = ["cv2", "decord", "av"]
    video_intervals = None
    video_length = None
    for video_mode in video_modes:
        try:
            video_intervals, video_length = pipe(video_path, video_length=None, video_mode=video_mode)
            break
        except Exception:
            LOGGER.warning("mode %s failed for %s", video_mode, video_path, exc_info=True)
    if video_intervals is None:
        LOGGER.error("all modes failed for %s", video_path)
        return {"file": [], "intervals": [], "valid_range": [], "fps": video_fps,
                "video_length": 0, "video_height": video_height, "video_width": video_width}

    # 时长过长的 interval 按 clip_max_length（10s）切
    clip_max_length = int(video_fps * 10)
    video_intervals_temp = []
    for vi in video_intervals:
        clip_l = vi[1] - vi[0] + 1
        if clip_l <= clip_max_length:
            video_intervals_temp.append([vi[0], vi[1]])
        else:
            n = clip_l // clip_max_length + 1
            new_l = clip_l // n + (1 if clip_l % n > 0 else 0)
            for _n in range(n):
                video_intervals_temp.append(
                    [vi[0] + _n * new_l, min(vi[1], vi[0] + (_n + 1) * new_l)])

    if not video_intervals_temp:
        video_intervals_temp = [[0, 0]]

    files, intervals, valid_range, clip_frames = [], [], [], []
    count = 0
    for clip in video_intervals_temp:
        start_frame, end_frame = clip
        if end_frame > 0:
            start_frame += 1
            end_frame -= 1
        if start_frame > end_frame or (start_frame == end_frame and end_frame > 0):
            continue
        if end_frame > 0 and end_frame - start_frame < min_length:
            continue
        save_path = os.path.join(output_folder,
                                  f"{os.path.basename(video_path)[:-4]}_{count}.mp4")
        try:
            save_cut_with_timeout(video_path, save_path, start_frame, end_frame)
        except Exception:
            LOGGER.warning("save_cut failed: %s", save_path, exc_info=True)
            continue
        if not os.path.isfile(save_path):
            LOGGER.warning("save_cut did not produce file: %s", save_path)
            continue
        clip_length = get_video_length(save_path)
        # 不做二次精修，valid_range 直接取 [0, clip_length]
        valid_range.append(json.dumps([0, clip_length]))
        files.append(save_path)
        intervals.append([start_frame, end_frame])
        clip_frames.append(clip_length)
        count += 1

    return {
        "file": files,
        "intervals": intervals,
        "fps": video_fps,
        "frames": video_length,
        "clip_frames": clip_frames,
        "video_height": video_height,
        "video_width": video_width,
        "video_length": video_length,
        "video_valid_range": valid_range,
        "raw_video_path": video_path,
    }


def cut_video(
    pipe: TransNetV2Pipeline,
    video_path: Path,
    output_data_dir: Path,
    result_dir: Path,
) -> dict:
    video_name = video_path.stem
    item: dict = {}
    cache_path = result_dir / f"{video_name}_labels.pkl"
    if cache_path.is_file():
        LOGGER.info("use cache: %s", cache_path)
        return pickle.load(open(cache_path, "rb"))

    result = main_split_fast(
        pipe, str(video_path), str(output_data_dir),
    )
    item["results"] = [
        {
            "file": [result["file"][i]],
            "interval": result["intervals"][i],
            "valid_range": json.loads(result["video_valid_range"][i]),
        }
        for i in range(len(result["file"]))
    ]
    # 同时保留完整 result 供后续 importer 使用
    pickle.dump({"results": item["results"], "meta": result}, open(cache_path, "wb"))
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-modes", nargs="+", default=["cv2", "decord", "av"],
                        choices=["cv2", "decord", "av"])
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")

    config = json.load(open(args.config))
    sub = config["datalist"][0]
    target_path = Path(sub["target_path"])
    version = sub.get("version", "v2.0.0")
    video_split_model = config["modelinfo"]["video_split_model"]

    v2 = target_path / version
    (v2 / "video" / "data").mkdir(parents=True, exist_ok=True)
    (v2 / "result").mkdir(parents=True, exist_ok=True)
    process_status = v2 / "process_status.json"
    if not process_status.exists():
        process_status.write_text(
            json.dumps({"cut_v2": False, "cut_only": False,
                        "audio": False, "base_filter_v2": False}, indent=2)
        )

    pipe = TransNetV2Pipeline(video_split_model).to(args.device)
    LOGGER.info("Loaded TransNetV2 on %s from %s", args.device, video_split_model)

    videos = find_videos(Path(args.input_dir))
    LOGGER.info("Found %d videos under %s", len(videos), args.input_dir)

    for v in videos:
        LOGGER.info("Cutting %s ...", v.name)
        t0 = time.time()
        try:
            item = cut_video(
                pipe, v,
                output_data_dir=v2 / "video" / "data",
                result_dir=v2 / "result",
            )
        except Exception as e:
            LOGGER.exception("Failed cutting %s: %s", v.name, e)
            continue
        n = len(item.get("results", []))
        LOGGER.info("  %s -> %d shots (%.1fs)", v.name, n, time.time() - t0)

    status = json.load(open(process_status))
    status["cut_only"] = True
    process_status.write_text(json.dumps(status, indent=2))
    LOGGER.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())