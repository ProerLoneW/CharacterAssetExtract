#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资产提取 orchestrator（浅层命名）。

输入：segments 目录（任一三路产物） + 全片视频路径。
输出（每种资产独立子目录）：

  assets/visual_person/<segment_id>/<obs_id>/face.jpg body_crop.jpg mask.png
  assets/voice/speakers.json + voice_assets/<speaker_id>/references/*.wav
  assets/key_object/<segment_id>/<object_id>/bbox.png mask.png evidence.json

每条记录带 source_segment_impl 字段，方便对比不同 segment 切分对
资产结果的影响。

注：entity_asset_extractor、speaker_voice_asset_pipeline、
object_grounding_segmenter_v3 都需要 GPU 与本地模型权重。本 runner
给出"调用入口 + 抽样 stub"两套路径：

  * --mode real：调用真实模块（需 GPU + 已下载模型）
  * --mode stub：不调模型，从 segment 时长/区间按确定性函数生成
                placeholder 资产 + manifest，方便用户先看产物结构

⚠️ 本 runner 的 real 模式是**早期 segment 级演示入口**。当前推荐的人物
视觉/音频资产链已切换为 global 级的 Scheme A 体系（统一 caption S-ID
命名空间），请直接按 ARCHITECTURE.md 4.12/4.13/4.14 节的脚本顺序运行：

  * 视觉：shot_tracklet_builder → semantic_role_tracklet_refiner_v3
    → visual_character_registry_builder → character_asset_bank
  * 音频：mimo_global_captioner_speaker_v2（caption 时归属 ASR）
    → export_all_asr_audio_review
  * ID 统一：character_role_aligner（CHR→S-ID 投票 + 角色目录物化）

用法::

    python -m extract_full_pipeline.assets_runner \
        --video <processed.mp4> \
        --segments-jsonl <segments_window/segments.jsonl> \
        --assets-root assets \
        --mode stub
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("assets_runner")


@dataclass
class AssetRecord:
    asset_id: str
    source_segment_id: str
    source_segment_impl: str
    kind: str           # visual_person | voice | key_object
    media_paths: List[str]
    meta: dict

    def to_dict(self) -> dict:
        return asdict(self)


def extract_keyframe(video_path: str, ts: float, dst: str, width: int = 320) -> str:
    """用 ffmpeg 抽一帧 JPEG。"""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{ts:.3f}", "-i", video_path,
        "-frames:v", "1", "-vf", f"scale={width}:-1", "-q:v", "3", dst,
    ]
    subprocess.run(cmd, check=True)
    return dst


def extract_audio_clip(video_path: str, start: float, end: float, dst: str) -> str:
    """用 ffmpeg 抽一段 wav（16k 单声道）。"""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", dst,
    ]
    subprocess.run(cmd, check=True)
    return dst


# ===========================================================================
# 1. 视觉人物资产（stub / real 入口）
# ===========================================================================

def visual_person_for_segment(
    video_path: str,
    segment: dict,
    impl: str,
    out_root: Path,
    mode: str,
) -> List[AssetRecord]:
    """对单个 segment 抽视觉人物资产。

    mode=stub：在 segment 中点抽一帧当 placeholder，并写 1 个 OBS 记录。
    mode=real：调 entity_asset_extractor 抽帧 + 检测 + SAM3.1 分割。
    """
    seg_id = segment["segment_id"]
    src = segment["source_time_range_sec"]
    mid_ts = (src[0] + src[1]) / 2
    seg_dir = out_root / "visual_person" / seg_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    records: List[AssetRecord] = []
    if mode == "stub":
        # Stub 1：抽 1 帧 + 写出 manifest（无真实模型结果）
        frame = seg_dir / "obs_stub_frame.jpg"
        extract_keyframe(video_path, mid_ts, str(frame), width=320)
        # 写 manifest placeholder
        manifest = seg_dir / "manifest.json"
        manifest.write_text(json.dumps({
            "segment_id": seg_id,
            "impl": impl,
            "timestamp_sec": mid_ts,
            "stub_reason": "stub mode (no GPU model loaded)",
            "frame": str(frame),
            "observations": [
                {"obs_id": "OBS_stub", "frame": str(frame),
                 "sa2va_bbox": None, "groundingdino_bbox": None,
                 "mask": None, "face": None,
                 "note": "run with --mode real + local models for real assets"},
            ],
        }, ensure_ascii=False, indent=2))
        records.append(AssetRecord(
            asset_id=f"{seg_id}/OBS_stub",
            source_segment_id=seg_id,
            source_segment_impl=impl,
            kind="visual_person",
            media_paths=[str(frame), str(manifest)],
            meta={"stub": True, "timestamp_sec": mid_ts},
        ))
    else:  # real
        # 真实调用 entity_asset_extractor.extract_frame_observations
        # 仅抽一帧 + 跑主函数（用户需已下载 Sa2VA / GDINO / SAM3.1 / InsightFace）
        from extract_full_pipeline.assets.visual_person import entity_asset_extractor as ea
        sample = ea.FrameSample(
            tag=f"seg_{seg_id}",
            percentile=0.5,
            frame_index=int(mid_ts * 25),
            timestamp_sec=mid_ts,
            image_path=str(seg_dir / "frame.jpg"),
        )
        extract_keyframe(video_path, mid_ts, sample.image_path, width=518)
        try:
            import numpy as np
            import cv2
            img = cv2.imread(sample.image_path)
            sa2va = ea.load_entity_model()
            gdino = ea.load_grounding_dino()
            face_app = ea.load_face_detector()
            sam = ea.load_sam() if not getattr(ea, "DEFAULT_DISABLE_SAM", False) else None
            observations, _ = ea.extract_frame_observations(
                frame_idx=int(mid_ts * 25),
                sample=sample,
                image_bgr=img,
                sa2va=sa2va, gdino=gdino, face_app=face_app,
                sam=sam, use_sam=True, output_root=seg_dir,
                crop_padding=0.05,
            )
            obs_records = [asdict(o) for o in observations]
            (seg_dir / "manifest.json").write_text(
                json.dumps({"segment_id": seg_id, "impl": impl,
                            "observations": obs_records},
                           ensure_ascii=False, indent=2)
            )
            for o in observations:
                records.append(AssetRecord(
                    asset_id=f"{seg_id}/{o.observation_id}",
                    source_segment_id=seg_id,
                    source_segment_impl=impl,
                    kind="visual_person",
                    media_paths=[p for p in [
                        o.body_crop_path, o.mask_path, o.face.get("crop_path") if o.face else None
                    ] if p],
                    meta={"sa2va_bbox": o.sa2va_bbox,
                          "groundingdino_bbox": o.groundingdino_bbox,
                          "face_detected": o.face.get("detected") if o.face else False},
                ))
        except Exception as e:
            LOGGER.warning("[%s] 视觉资产 real 模式失败：%s", seg_id, e)
    return records


# ===========================================================================
# 2. 音频说话人资产（聚合全片）
# ===========================================================================

def voice_pipeline(
    video_path: str,
    out_root: Path,
    mode: str,
    work_dir: Optional[Path] = None,
) -> List[AssetRecord]:
    """全片说话人资产（与 segment 无关；产出一个 speaker registry）。

    mode=stub：在每个 segment（来自 segments_compare）的中点抽 1 段 wav
              当 placeholder references。
    mode=real：调 speaker_voice_asset_pipeline.run。
    """
    segs_root = out_root.parent / "segments_compare"  # 假定叫 segments_compare
    if mode == "real":
        from extract_full_pipeline.assets.voice import speaker_voice_asset_pipeline as sv
        work = work_dir or out_root / "_voice_work"
        work.mkdir(parents=True, exist_ok=True)
        try:
            import argparse
            ns = argparse.Namespace(
                input=str(video_path),
                output=str(work),
                # 简化：禁用 Stage5 的重引擎，用最小可行 flag 集合
                enhance_overwrite=False,
                retries=3,
                turn_embed_padding=0.75,
                pooled_merge_threshold=0.62,
                pooled_possible_threshold=0.50,
                pooled_snorm_threshold=2.0,
                use_snorm=False,
                max_temporal_overlap=0.10,
                max_pool_seconds=40.0,
                min_pool_seconds=1.5,
                min_speaker_clean_seconds=2.0,
                min_coverage=0.5,
                max_contamination=0.5,
                workers=1,
                ffmpeg="ffmpeg",
            )
            summary = sv.run(ns)
            (out_root / "voice" / "speakers.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2))
            refs_dir = work / "consolidated" / "speaker_registry"
            if refs_dir.exists():
                for sp in (out_root / "voice").iterdir():
                    pass
                shutil.copytree(refs_dir, out_root / "voice" / "speaker_registry",
                                dirs_exist_ok=True)
            return [AssetRecord(
                asset_id="voice_pipeline",
                source_segment_id="*",
                source_segment_impl="*",
                kind="voice",
                media_paths=[str(out_root / "voice" / "speakers.json")],
                meta={"summary_keys": list(summary.keys())},
            )]
        except Exception as e:
            LOGGER.warning("voice real 模式失败：%s", e)
            return []

    # stub
    voice_dir = out_root / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    # 选 segments_compare/summary.json 中任一 impl 的前 3 个 segment 作为 placeholder
    summary = out_root.parent / "segments_compare" / "summary.json"
    segments_jsonl: Optional[Path] = None
    if summary.exists():
        info = json.loads(summary.read_text())
        first_impl = next(iter(info.keys()))
        segments_jsonl = Path(info[first_impl]["path"])
    if segments_jsonl is None or not segments_jsonl.exists():
        # 兜底：直接拿第一段
        for cand in (out_root.parent / "segments_compare").glob("segments_*/segments.jsonl"):
            segments_jsonl = cand
            break
    segs = [json.loads(l) for l in open(segments_jsonl) if l.strip()] if segments_jsonl else []
    refs: List[dict] = []
    for s in segs[:3]:
        wav = voice_dir / f"placeholder_spk_{s['segment_id']}.wav"
        extract_audio_clip(video_path, s["source_time_range_sec"][0],
                           min(s["source_time_range_sec"][1],
                               s["source_time_range_sec"][0] + 5.0),
                           str(wav))
        refs.append({"segment_id": s["segment_id"], "wav": str(wav),
                     "duration_sec": s["duration_sec"]})
    speakers = {"speakers": [
        {"speaker_id": "SPK001", "reference_audio": [r["wav"] for r in refs],
         "pooled_seconds": sum(r["duration_sec"] for r in refs),
         "stub_reason": "stub mode (no audio foreground separation / Stage5)"},
    ], "stub_mode": True, "n_segments_used": len(refs)}
    (voice_dir / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    return [AssetRecord(
        asset_id="voice_pipeline",
        source_segment_id="*",
        source_segment_impl="*",
        kind="voice",
        media_paths=[r["wav"] for r in refs] + [str(voice_dir / "speakers.json")],
        meta={"stub": True, "speakers": 1, "segments_used": len(refs)},
    )]


# ===========================================================================
# 3. 关键物体资产（基于 segment caption 的 key_object_candidates）
# ===========================================================================

def key_object_for_segment(
    video_path: str,
    segment: dict,
    impl: str,
    out_root: Path,
    mode: str,
) -> List[AssetRecord]:
    seg_id = segment["segment_id"]
    src = segment["source_time_range_sec"]
    mid_ts = (src[0] + src[1]) / 2
    seg_dir = out_root / "key_object" / seg_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    if mode == "stub":
        # 抽 1 帧 + 写 placeholder
        bbox_img = seg_dir / "bbox.png"
        extract_keyframe(video_path, mid_ts, str(bbox_img), width=320)
        evidence = {
            "segment_id": seg_id, "impl": impl,
            "stub_reason": "stub mode (no GDINO / SAM3.1)",
            "objects": [
                {"candidate_id": "OC_stub", "phrase": "stub-object",
                 "bbox_xyxy": [80, 60, 240, 180], "score": 0.5,
                 "mask_area_pixels": 0, "image": str(bbox_img)},
            ],
        }
        (seg_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        return [AssetRecord(
            asset_id=f"{seg_id}/OC_stub",
            source_segment_id=seg_id,
            source_segment_impl=impl,
            kind="key_object",
            media_paths=[str(bbox_img), str(seg_dir / "evidence.json")],
            meta={"stub": True, "phrase": "stub-object"},
        )]

    # real: 用 build_object_grounding_queue + object_grounding_segmenter_v3
    # 没有 shot_captions 时跳过
    from extract_full_pipeline.assets.key_object import (
        build_object_grounding_queue_v2 as q,
        object_grounding_segmenter_v3 as og,
    )
    # 占位：建一个 1 行的 shot_captions.jsonl 喂给 build_queue
    shot_caps = {
        "shots_path": Path(video_path).parent.parent / "shots" / "processed" / "shots.jsonl",
    }
    # 直接抽帧占位
    bbox_img = seg_dir / "bbox.png"
    extract_keyframe(video_path, mid_ts, str(bbox_img), width=320)
    evidence = {
        "segment_id": seg_id, "impl": impl,
        "real_run_skipped": True,
        "reason": "need shot_captions.jsonl with key_object_candidates for real mode",
        "image": str(bbox_img),
    }
    (seg_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    return [AssetRecord(
        asset_id=f"{seg_id}/OC_real_placeholder",
        source_segment_id=seg_id,
        source_segment_impl=impl,
        kind="key_object",
        media_paths=[str(bbox_img), str(seg_dir / "evidence.json")],
        meta={"real_run_skipped": True},
    )]


# ===========================================================================
# 入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--segments-jsonl", required=True)
    p.add_argument("--assets-root", required=True)
    p.add_argument("--mode", choices=["stub", "real"], default="stub")
    p.add_argument("--max-segments", type=int, default=0,
                   help="0 = 处理所有 segment；>0 = 仅前 N 个 segment")
    p.add_argument("--include-voice", action="store_true")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    with open(args.segments_jsonl) as f:
        segs = [json.loads(l) for l in f if l.strip()]
    LOGGER.info("loaded %d segments from %s", len(segs), args.segments_jsonl)

    # impl 取自 segments 文件名（segments_window -> "window"）
    impl = Path(args.segments_jsonl).parent.name.replace("segments_", "")

    if args.max_segments > 0:
        segs = segs[:args.max_segments]

    out_root = Path(args.assets_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_records: List[AssetRecord] = []
    for i, s in enumerate(segs, 1):
        if s["duration_sec"] < 1.0:  # skip tiny artifacts
            continue
        all_records += visual_person_for_segment(args.video, s, impl, out_root, args.mode)
        all_records += key_object_for_segment(args.video, s, impl, out_root, args.mode)
        if i % 5 == 0 or i == len(segs):
            LOGGER.info("processed %d/%d segments", i, len(segs))

    if args.include_voice:
        all_records += voice_pipeline(args.video, out_root, args.mode)

    # 总 manifest
    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    LOGGER.info("wrote %s  (%d records)", manifest_path, len(all_records))
    summary = {
        "mode": args.mode,
        "impl": impl,
        "n_segments": len(segs),
        "n_records": len(all_records),
        "by_kind": {},
    }
    for r in all_records:
        summary["by_kind"][r.kind] = summary["by_kind"].get(r.kind, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())