#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全流程统一编排入口(最新主线逻辑,不含旧 CHR 链/声音增强支线)。

给一个输入目录(或显式视频列表)+ 每个视频的 ASR 文件,按主线顺序产出全部产物:

    切镜 → 全局聚合 → 全局描述(说话归属) → 镜头描述 → 固定段规划
    → 片段描述 → 片段视频 → 人物轨迹 → 轨迹全局归属 → 角色声音候选 → 角色素材物化
    → 关键物体候选 → 物体定位分割 → 物体参考集 → 场景参考图

对应模块与入口(均为各阶段最新版):

    shot_split        shot_splitter.py                              (--flat)
    global_group      global_clip_aggregator.py                     (--flat)
    global_describe   mimo_global_captioner_speaker_v2.py           (含 ASR 说话归属)
    shot_describe     mimo_shot_captioner_v3.py                     (V5 prompt)
    segment_plan      segmentation/three_way.py --impls window
    segment_describe  mimo_segment_captioner_fixed_segments_v4.py
    segment_export    mimo_segment_cutter.py
    person_tracks     assets/visual_person/shot_tracklet_builder.py (visual_person_v2 轨迹)
    person_align      global_align_pipeline/global_tracklet_aligner.py (直接归属 S-ID)
    voice_preview     assets/voice/export_all_asr_audio_review.py
    character_pack    global_align_pipeline/global_character_materializer.py
    object_queue      assets/key_object/build_object_grounding_queue_v2.py (时间窗候选)
    object_ground     assets/key_object/object_grounding_segmenter_v3.py (V4 逻辑,含参考集)
    scene_select      assets/scene/important_scene_frame_extractor.py

每个视频一个独立工作区(--output-root/<视频名>),统一布局(与文档二 integrated/basic-flat 对齐):

    <ws>/shots/  global/  captions/{global,shot,segment}/  segment_plan/segments_window/
    <ws>/segments/  tracklets/  global_alignment/  characters/  voice_by_role_src/
    <ws>/key_object/{grounding_queue.jsonl, grounding/}  scene/
    <ws>/logs/<stage>.log  pipeline_state.json  pipeline_state_history.jsonl  viewer_binding.json

断点续跑:阶段成功且产物标记存在则跳过;--force 重跑;--only 选择阶段;失败默认继续下一个视频。
跑完自动生成 viewer_binding.json;--update-viewer-config 可直接并入 result_viewer/viewer_config.json
(重启 serve.py 后网站即可浏览)。

外部前置:ffmpeg/ffprobe、VLM 端点(OPENAI_BASE_URL/OPENAI_API_BASE/API_BASE、OPENAI_API_KEY、MODEL_NAME)、
ASR 文件(每视频必需)、物体定位(SAM3.1/ grounding 模型)、场景(YOLO/DINOv3)、物化可选 InsightFace
等本地模型权重(按各脚本文档)。本编排器不替这些脚本做模型推理配置。

用法::

    python run_full_pipeline.py --input-dir <目录> --asr-dir <ASR 目录> [--dry-run]
    python run_full_pipeline.py --inputs a.mp4 b.mp4 --asr-map asr_map.json
    python run_full_pipeline.py --input-dir <目录> --only person_align character_pack --force
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent
HERE = REPO_ROOT  # 便于阅读;所有阶段脚本都相对仓库根解析

VLM_BASE_ENVS = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "API_BASE")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 阶段注册表:每个阶段 = 命令构造器 + 产物标记(存在视为可跳过) + 上游标记
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, sid: str, title: str, module_id: str,
                 cmd: Callable[[dict], list[str]],
                 produces: Callable[[dict], Path],
                 requires: Callable[[dict], list[Path]]):
        self.id, self.title, self.module_id = sid, title, module_id
        self.cmd, self.produces, self.requires = cmd, produces, requires


def _py(script: Path) -> str:
    return str(script)


def _stages(ns: argparse.Namespace) -> list[Stage]:
    """ns 提供 concurrency/retries/timeout 等编排级参数。ctx 见 VideoJob。"""
    return [
        Stage("split_shots", "切镜", "shot_split",
              lambda c: [sys.executable, _py(HERE / "shot_splitter.py"),
                         "--input", str(c.video), "--output-dir", str(c.ws / "shots"),
                         "--detector", ns.detector, "--flat"],
              lambda c: c.ws / "shots" / "shots.jsonl",
              lambda c: [c.video]),
        Stage("aggregate_global", "全局片段聚合", "global_group",
              lambda c: [sys.executable, _py(HERE / "global_clip_aggregator.py"),
                         "--input", str(c.ws / "shots"), "--output-dir", str(c.ws / "global"),
                         "--min-sec", str(ns.min_global_sec), "--target-sec", str(ns.target_global_sec),
                         "--max-sec", str(ns.max_global_sec), "--flat"],
              lambda c: c.ws / "global" / "global_clips.jsonl",
              lambda c: [c.ws / "shots" / "shots.jsonl"]),
        Stage("global_caption", "全局描述与说话归属", "global_describe",
              lambda c: [sys.executable, _py(HERE / "mimo_global_captioner_speaker_v2.py"),
                         "--input", str(c.ws / "global"), "--output-dir", str(c.ws / "captions" / "global"),
                         "--asr-input", str(c.asr), "--concurrency", str(ns.caption_workers),
                         "--retries", str(ns.retries), "--timeout", str(ns.request_timeout)]
                        + ([] if ns.force else ["--skip-existing"]) + ["--flat"],
              lambda c: c.ws / "captions" / "global" / "global_captions.jsonl",
              lambda c: [c.ws / "global" / "global_clips.jsonl", c.asr]),
        Stage("shot_caption", "镜头描述", "shot_describe",
              lambda c: [sys.executable, _py(HERE / "mimo_shot_captioner_v3.py"),
                         "--input", str(c.ws / "shots"), "--global-input", str(c.ws / "captions" / "global"),
                         "--output-dir", str(c.ws / "captions" / "shot"),
                         "--concurrency", str(ns.caption_workers), "--retries", str(ns.retries),
                         "--timeout", str(ns.request_timeout)]
                        + ([] if ns.force else ["--skip-existing"]) + ["--flat"],
              lambda c: c.ws / "captions" / "shot" / "shot_captions.jsonl",
              lambda c: [c.ws / "captions" / "global" / "global_captions.jsonl"]),
        Stage("segment_plan", "固定段规划(window)", "segment_plan",
              lambda c: [sys.executable, _py(HERE / "segmentation" / "three_way.py"),
                         "--global-clips", str(c.ws / "global" / "global_clips.jsonl"),
                         "--shots-jsonl", str(c.ws / "shots" / "shots.jsonl"),
                         "--out-dir", str(c.ws / "segment_plan"), "--impls", "window"],
              lambda c: c.ws / "segment_plan" / "segments_window" / "segments.jsonl",
              lambda c: [c.ws / "global" / "global_clips.jsonl", c.ws / "shots" / "shots.jsonl"]),
        Stage("segment_caption", "片段描述(固定段 v4)", "segment_describe",
              lambda c: [sys.executable, _py(HERE / "mimo_segment_captioner_fixed_segments_v4.py"),
                         "--global-input", str(c.ws / "captions" / "global"),
                         "--shot-input", str(c.ws / "captions" / "shot"),
                         "--segment-input", str(c.ws / "segment_plan" / "segments_window" / "segments.jsonl"),
                         "--output-dir", str(c.ws / "captions" / "segment"),
                         "--concurrency", str(ns.segment_workers), "--retries", str(ns.segment_retries),
                         "--timeout", str(ns.request_timeout)]
                        + ([] if ns.force else ["--skip-existing"]) + ["--flat"],
              lambda c: c.ws / "captions" / "segment" / "segment_captions.jsonl",
              lambda c: [c.ws / "segment_plan" / "segments_window" / "segments.jsonl",
                         c.ws / "captions" / "shot" / "shot_captions.jsonl"]),
        Stage("cut_segments", "片段视频导出", "segment_export",
              lambda c: [sys.executable, _py(HERE / "mimo_segment_cutter.py"),
                         "--input", str(c.ws / "captions" / "segment"),
                         "--output-dir", str(c.ws / "segments"), "--flat"],
              lambda c: c.ws / "segments" / "segments.jsonl",
              lambda c: [c.ws / "captions" / "segment" / "segment_captions.jsonl"]),
        Stage("person_tracklets", "人物轨迹提取", "person_tracks",
              lambda c: [sys.executable, _py(HERE / "assets" / "visual_person" / "shot_tracklet_builder.py"),
                         "--shots-dir", str(c.ws / "shots" / "clips"),
                         "--output", str(c.ws / "tracklets")],
              lambda c: c.ws / "tracklets" / "manifest.json",
              lambda c: [c.ws / "shots" / "shots.jsonl"]),
        Stage("person_align", "轨迹全局归属", "person_align",
              lambda c: [sys.executable, _py(HERE / "global_align_pipeline" / "global_tracklet_aligner.py"),
                         "--tracklets", str(c.ws / "tracklets"),
                         "--global-manifest", str(c.ws / "global" / "global_clips.jsonl"),
                         "--shots", str(c.ws / "shots" / "shots.jsonl"),
                         "--global-captions", str(c.ws / "captions" / "global"),
                         "--shot-captions", str(c.ws / "captions" / "shot"),
                         "--segment-captions", str(c.ws / "captions" / "segment"),
                         "--output", str(c.ws / "global_alignment"),
                         "--concurrency", str(ns.align_concurrency)],
              lambda c: c.ws / "global_alignment" / "global_tracklet_alignment.jsonl",
              lambda c: [c.ws / "tracklets" / "manifest.json",
                         c.ws / "captions" / "segment" / "segment_captions.jsonl"]),
        Stage("voice_preview", "角色声音候选", "voice_preview",
              lambda c: [sys.executable, _py(HERE / "assets" / "voice" / "export_all_asr_audio_review.py"),
                         "--input", str(c.ws / "captions" / "global" / "speaker_attributed_asr.jsonl"),
                         "--output-dir", str(c.ws / "voice_by_role_src")],
              lambda c: c.ws / "voice_by_role_src" / "all_asr_manifest.jsonl",
              lambda c: [c.ws / "captions" / "global" / "speaker_attributed_asr.jsonl"]),
        Stage("character_pack", "角色素材物化", "character_pack",
              lambda c: [sys.executable, _py(HERE / "global_align_pipeline" / "global_character_materializer.py"),
                         "--alignment", str(c.ws / "global_alignment" / "global_tracklet_alignment.jsonl"),
                         "--tracklets", str(c.ws / "tracklets"),
                         "--voice-root", str(c.ws / "voice_by_role_src"),
                         "--output", str(c.ws),
                         "--face-top-k", str(ns.face_top_k), "--body-top-k", str(ns.body_top_k)]
                        + (["--identity-check", "--face-model", str(ns.face_model)]
                           if ns.identity_check and ns.face_model else []),
              lambda c: c.ws / "materialization_summary.json",
              lambda c: [c.ws / "global_alignment" / "global_tracklet_alignment.jsonl",
                         c.ws / "voice_by_role_src" / "all_asr_manifest.jsonl"]),
        Stage("object_queue", "关键物体候选(时间窗 v2)", "object_queue",
              lambda c: [sys.executable, _py(HERE / "assets" / "key_object" / "build_object_grounding_queue_v2.py"),
                         "--shot-captions", str(c.ws / "captions" / "shot" / "shot_captions.jsonl"),
                         "--output", str(c.ws / "key_object" / "grounding_queue.jsonl")],
              lambda c: c.ws / "key_object" / "grounding_queue.jsonl",
              lambda c: [c.ws / "captions" / "shot" / "shot_captions.jsonl"]),
        Stage("object_ground", "物体定位与分割(V4 逻辑)", "object_ground",
              lambda c: [sys.executable, _py(HERE / "assets" / "key_object" / "object_grounding_segmenter_v3.py"),
                         "--candidates", str(c.ws / "key_object" / "grounding_queue.jsonl"),
                         "--shots", str(c.ws / "shots" / "shots.jsonl"),
                         "--output", str(c.ws / "key_object" / "grounding")],
              lambda c: c.ws / "key_object" / "grounding" / "object_observations.jsonl",
              lambda c: [c.ws / "key_object" / "grounding_queue.jsonl",
                         c.ws / "shots" / "shots.jsonl"]),
        Stage("object_references", "物体多帧参考集", "object_ground",
              lambda c: [sys.executable, _py(HERE / "assets" / "key_object" / "object_grounding_segmenter_v3.py"),
                         "--build-reference-sets",
                         "--grounding-dir", str(c.ws / "key_object" / "grounding"),
                         "--candidates", str(c.ws / "key_object" / "grounding_queue.jsonl"),
                         "--output", str(c.ws / "key_object" / "grounding" / "reference_sets.json")],
              lambda c: c.ws / "key_object" / "grounding" / "reference_sets.json",
              lambda c: [c.ws / "key_object" / "grounding" / "object_observations.jsonl",
                         c.ws / "key_object" / "grounding_queue.jsonl"]),
        Stage("scene_select", "场景参考图", "scene_select",
              lambda c: [sys.executable, _py(HERE / "assets" / "scene" / "important_scene_frame_extractor.py"),
                         str(c.video),
                         "--output-dir", str(c.ws / "scene"), "--overwrite"],
              lambda c: c.ws / "scene" / c.video.stem / "scene_index.json",
              lambda c: [c.video]),
    ]


# ---------------------------------------------------------------------------
# 每视频作业:状态文件 + 阶段执行
# ---------------------------------------------------------------------------

class VideoJob:
    def __init__(self, video: Path, ws: Path, asr: Path):
        self.video, self.ws, self.asr = video, ws, asr
        self.state_path = ws / "pipeline_state.json"
        self.history_path = ws / "pipeline_state_history.jsonl"
        self.log_dir = ws / "logs"

    def load_state(self) -> dict:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"video": str(self.video), "workspace": str(self.ws), "asr": str(self.asr),
                "created_at": _now(), "stages": {}}

    def save_state(self, state: dict) -> None:
        self.ws.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def append_history(self, record: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_stage(self, stage: Stage, ns: argparse.Namespace) -> bool:
        state = self.load_state()
        entry = state["stages"].get(stage.id, {})
        marker = stage.produces(self)
        if (not ns.force and entry.get("status") == "succeeded" and marker.exists()):
            print(f"  = {stage.id:<16} 跳过(已成功且产物存在)", flush=True)
            return True
        missing = [str(p) for p in stage.requires(self) if not p.exists()]
        if missing:
            print(f"  ! {stage.id:<16} 上游产物缺失: {missing}", flush=True)
            self._record(state, stage, "failed", exit_code=None, note=f"missing upstream: {missing}")
            return False

        cmd = stage.cmd(self)
        entry.update({"status": "running", "started_at": _now(), "cmd": cmd,
                      "module_id": stage.module_id, "title": stage.title})
        state["stages"][stage.id] = entry
        self.save_state(state)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{stage.id}.log"
        print(f"  ▶ {stage.id:<16} {stage.title} → {log_path.name}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {_now()} =====\n{' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT,
                                  timeout=ns.stage_timeout or None)
        ok = proc.returncode == 0 and marker.exists()
        self._record(state, stage, "succeeded" if ok else "failed", exit_code=proc.returncode,
                     note=None if ok else ("产物标记缺失: " + str(marker) if proc.returncode == 0 else None))
        print(f"  {'✓' if ok else '✗'} {stage.id:<16} exit={proc.returncode}", flush=True)
        return ok

    def _record(self, state: dict, stage: Stage, status: str, exit_code: int | None, note: str | None) -> None:
        entry = state["stages"].get(stage.id, {})
        entry.update({"status": status, "ended_at": _now(), "exit_code": exit_code})
        if note:
            entry["note"] = note
        state["stages"][stage.id] = entry
        self.save_state(state)
        self.append_history({"time": _now(), "video": self.video.name, "stage": stage.id,
                             "status": status, "exit_code": exit_code, "note": note})

    def finished_all(self, stages: list[Stage]) -> bool:
        state = self.load_state()
        return all(state["stages"].get(s.id, {}).get("status") == "succeeded" and s.produces(self).exists()
                   for s in stages)


# ---------------------------------------------------------------------------
# viewer 绑定片段:工作区 → result_viewer/viewer_config.json 的 runs 条目
# ---------------------------------------------------------------------------

def write_viewer_binding(job: VideoJob, ns: argparse.Namespace) -> Path:
    run_id = job.ws.name
    rel = _rel_to_config(job.ws, ns)
    binding = {
        "video_id": job.video.stem,
        "title": f"{job.video.stem} 全流程主线运行({run_id})",
        "layout_profile": "custom",
        "route_id": "fixed-global-align",
        "produced_code_revision": None,
        "created_at": _dt.date.today().isoformat(),
        "workspace_root": rel,
        "source_roots": [{"root_id": "run_root", "path": rel}],
        "path_rewrites": [{"pattern": f"^.*/{re.escape(run_id)}/", "replacement": "{run_root}/"}],
        "artifact_bindings": {
            "shots": {"relative_path": "shots/shots.jsonl", "producer_module_id": "shot_split"},
            "globals": {"relative_path": "global/global_clips.jsonl", "producer_module_id": "global_group"},
            "global_captions": {"relative_path": "captions/global/global_captions.jsonl", "producer_module_id": "global_describe"},
            "attributed_asr": {"relative_path": "captions/global/speaker_attributed_asr.jsonl", "producer_module_id": "global_describe"},
            "shot_captions": {"relative_path": "captions/shot/shot_captions.jsonl", "producer_module_id": "shot_describe"},
            "segment_captions": {"relative_path": "captions/segment/segment_captions.jsonl", "producer_module_id": "segment_describe"},
            "segment_media": {"relative_path": "segments/segments.jsonl", "producer_module_id": "segment_export"},
            "tracklets": {"relative_path": "tracklets", "producer_module_id": "person_tracks"},
            "direct_alignment": {"relative_path": "global_alignment/global_tracklet_alignment.jsonl", "producer_module_id": "person_align"},
            "voice_previews": {"relative_path": "voice_by_role_src/all_asr_manifest.jsonl", "producer_module_id": "voice_preview"},
            "characters": {"relative_path": "characters", "producer_module_id": "character_pack"},
            "character_summary": {"relative_path": "materialization_summary.json", "producer_module_id": "character_pack"},
            "object_candidates": {"relative_path": "key_object/grounding_queue.jsonl", "producer_module_id": "object_queue"},
            "object_observations": {"relative_path": "key_object/grounding/object_observations.jsonl", "producer_module_id": "object_ground"},
            "object_references": {"relative_path": "key_object/grounding/reference_sets.json", "producer_module_id": "object_ground"},
            "scene_assets": {"relative_path": "scene", "producer_module_id": "scene_select"},
        },
    }
    out = job.ws / "viewer_binding.json"
    out.write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding="utf-8")

    if ns.update_viewer_config:
        cfg_path = Path(ns.update_viewer_config).resolve()
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["runs"][run_id] = binding
        if run_id not in cfg["allowed_runs"]:
            cfg["allowed_runs"].append(run_id)
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ◈ 已并入 viewer 配置 {cfg_path}(重启 serve.py 生效): run_id={run_id}", flush=True)
    return out


def _rel_to_config(ws: Path, ns: argparse.Namespace) -> str:
    cfg_dir = Path(ns.update_viewer_config).resolve().parent if ns.update_viewer_config \
        else REPO_ROOT / "result_viewer"
    try:
        return str(Path(os.path.relpath(ws, cfg_dir)).as_posix())
    except ValueError:  # 跨盘符
        return str(ws)


# ---------------------------------------------------------------------------
# 输入发现 / 预检
# ---------------------------------------------------------------------------

def discover_videos(ns: argparse.Namespace) -> list[Path]:
    if ns.inputs:
        return [Path(p).expanduser().resolve() for p in ns.inputs]
    root = Path(ns.input_dir).expanduser().resolve()
    it = root.rglob(ns.pattern) if ns.recursive else root.glob(ns.pattern)
    return sorted(p for p in it if p.is_file())


def find_asr(video: Path, ns: argparse.Namespace, asr_map: dict) -> Path | None:
    stem = video.stem
    if stem in asr_map:
        return Path(asr_map[stem]).expanduser().resolve()
    if ns.asr:
        return Path(ns.asr).expanduser().resolve()
    names = [f"{stem}.asr.jsonl", f"{stem}_asr.jsonl", f"{stem}.asr.json", f"{stem}_asr.json",
             f"{stem}.jsonl", f"{stem}.json"]
    search_dirs = [video.parent]
    if ns.asr_dir:
        search_dirs.insert(0, Path(ns.asr_dir).expanduser().resolve())
    for d in search_dirs:
        for name in names:
            cand = d / name
            if cand.is_file():
                return cand.resolve()
        for cand in sorted(d.glob(f"{stem}*.jsonl")) + sorted(d.glob(f"{stem}*.json")):
            if cand.is_file():
                return cand.resolve()
    return None


def preflight(ns: argparse.Namespace, videos: list[Path]) -> list[str]:
    problems = []
    if not videos:
        problems.append("没有找到任何输入视频(--input-dir/--pattern 或 --inputs)")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            problems.append(f"PATH 中找不到 {tool}(切段/归属拼接/声音预览都需要)")
    if not any(os.getenv(k) for k in VLM_BASE_ENVS):
        problems.append("未设置 VLM 端点环境变量(OPENAI_BASE_URL / OPENAI_API_BASE / API_BASE),"
                        "三级描述与归属阶段会失败或落到脚本默认值")
    if not os.getenv("MODEL_NAME"):
        print("[preflight] 提示: 未设置 MODEL_NAME,将使用 captioner 默认模型", flush=True)
    return problems


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    _stdout_utf8()
    ns = parse_args()
    if ns.list_stages:
        for s in _stages(ns):
            print(f"{s.id:<18} {s.title}  (模块 {s.module_id})")
        return 0

    asr_map: dict = {}
    if ns.asr_map:
        asr_map = json.loads(Path(ns.asr_map).expanduser().read_text(encoding="utf-8"))

    videos = discover_videos(ns)
    problems = preflight(ns, videos)
    if problems:
        for p in problems:
            print(f"[preflight] {p}", flush=True)
        if not ns.dry_run and problems and not ns.ignore_preflight:
            return 2

    stages = [s for s in _stages(ns) if not ns.only or s.id in ns.only]
    if ns.only:
        unknown = set(ns.only) - {s.id for s in _stages(ns)}
        if unknown:
            print(f"[args] 未知阶段: {sorted(unknown)};可用: {[s.id for s in _stages(ns)]}", flush=True)
            return 2

    failed_videos = 0
    for video in videos:
        stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", video.stem) or "video"
        ws = (Path(ns.output_root).expanduser().resolve() / stem)
        asr = find_asr(video, ns, asr_map)
        print(f"\n=== {video.name} → {ws}", flush=True)
        if asr is None:
            print("  ! 未找到 ASR 文件(speaker_v2 与声音候选必需);跳过该视频。"
                  "可用 --asr-map/--asr-dir/--asr 显式指定。", flush=True)
            failed_videos += 1
            continue
        print(f"    ASR: {asr}", flush=True)
        job = VideoJob(video, ws, asr)

        if ns.dry_run:
            for s in stages:
                print(f"    [{s.id}] {' '.join(s.cmd(job))}", flush=True)
            print("    (dry-run:全部成功后还会生成 viewer_binding.json)", flush=True)
            continue

        for s in stages:
            if not job.run_stage(s, ns) and ns.stop_on_error:
                print(f"  ■ 阶段 {s.id} 失败,--stop-on-error 中止剩余阶段", flush=True)
                break
        if job.finished_all(stages):
            write_viewer_binding(job, ns)
        state = job.load_state()
        done = sum(1 for e in state["stages"].values() if e.get("status") == "succeeded")
        print(f"  总结: {done}/{len(stages)} 阶段成功;状态: {job.state_path}", flush=True)
        if not job.finished_all(stages):
            failed_videos += 1

    print(f"\n[done] 视频 {len(videos)} 个,失败 {failed_videos} 个"
          + ("(dry-run,未执行)" if ns.dry_run else ""), flush=True)
    return 1 if failed_videos else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="全流程统一编排(最新主线逻辑,无 CHR 链)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("输入")
    src.add_argument("--input-dir", help="扫描该目录下的视频(默认 *.mp4)")
    src.add_argument("--pattern", default="*.mp4", help="input-dir 的文件 glob")
    src.add_argument("--recursive", action="store_true", help="递归扫描 input-dir")
    src.add_argument("--inputs", nargs="+", help="显式视频列表(优先于 input-dir)")
    src.add_argument("--asr", help="单一 ASR 文件(配合 --inputs 单视频)")
    src.add_argument("--asr-dir", help="ASR 文件目录(按 <视频名>*.jsonl/json 匹配)")
    src.add_argument("--asr-map", help="JSON 文件 {视频名: ASR 路径}")

    out = p.add_argument_group("输出与执行")
    out.add_argument("--output-root", default=str(HERE / "full_pipeline_runs"), help="每视频工作区的父目录")
    out.add_argument("--force", action="store_true", help="重跑所有阶段(忽略成功记录)")
    out.add_argument("--only", nargs="+", help="只跑指定阶段 id(--list-stages 查看)")
    out.add_argument("--stop-on-error", action="store_true", help="某阶段失败即中止(默认继续下一视频)")
    out.add_argument("--stage-timeout", type=int, default=None, help="单阶段超时秒数(默认不限)")
    out.add_argument("--dry-run", action="store_true", help="只打印计划,不执行")
    out.add_argument("--list-stages", action="store_true", help="列出阶段后退出")
    out.add_argument("--ignore-preflight", action="store_true", help="预检仅警告不拦截")

    cap = p.add_argument_group("模型与阶段参数(透传)")
    cap.add_argument("--detector", default="auto", help="切镜检测器")
    cap.add_argument("--min-global-sec", type=float, default=60.0)
    cap.add_argument("--target-global-sec", type=float, default=90.0)
    cap.add_argument("--max-global-sec", type=float, default=120.0)
    cap.add_argument("--caption-workers", type=int, default=2)
    cap.add_argument("--segment-workers", type=int, default=1)
    cap.add_argument("--align-concurrency", type=int, default=2)
    cap.add_argument("--retries", type=int, default=3)
    cap.add_argument("--segment-retries", type=int, default=6)
    cap.add_argument("--request-timeout", type=float, default=1800.0)
    cap.add_argument("--face-top-k", type=int, default=8)
    cap.add_argument("--body-top-k", type=int, default=8)
    cap.add_argument("--identity-check", action="store_true", help="物化时启用 InsightFace 校验")
    cap.add_argument("--face-model", type=Path, default=None, help="InsightFace 权重路径")

    viewer = p.add_argument_group("结果网站")
    viewer.add_argument("--update-viewer-config", default=None,
                        help="全部阶段成功后把本运行并入该 viewer_config.json(如 result_viewer/viewer_config.json),需重启 serve.py")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
