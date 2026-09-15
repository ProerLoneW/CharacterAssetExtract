# -*- coding: utf-8 -*-
"""⑦ Segment 聚合主流程（纯聚合，不含 caption/prompt）。

流程定位（调整后）：m07 只做**时长可控的 clip 聚合**，产出
segments.jsonl（clip 分组 + 质量统计）；caption 由 m06 统一完成
（m06 读 m07 的聚合结果，对 segment 和 clip 一并做 caption）。

划分（window.plan_segments）：按邻接表顺序把全片切成连续 segment，
时长由 target/max_segment_seconds 控制；视觉不一致（CLIP 与段质心
相似度低于阈值）可作为边界信号提前收刀，形成完整无重叠划分。

质量依据（来自 qc 打分卡）：
- grade_weight   = 段内 clip 等级权重均值（A/B/C 可配），训练加权采样用；
- unqualified_ratio = 段内 C 级/rejected clip 时长占比，如实记录不过滤。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from ..common.jsonl import load_records, upsert_records, write_jsonl
from ..common.workspace import Workspace
from ..schema import ClipRecord, GlobalRecord, SegmentRecord
from .consistency import (DINOv3Embedder, cosine, gray_hist_embedding,
                          window_consistency)
from .window import plan_segments


def grade_weight(window: List[ClipRecord], weight_map: Dict[str, float]) -> float:
    ws = [weight_map.get(c.quality.grade, 0.0) for c in window]
    return round(sum(ws) / max(len(ws), 1), 4)


class _BoundaryJudge:
    """段边界信号：下一个 clip 与当前段质心的视觉相似度低于阈值 → 收刀。

    - 同 parent 的再切割子 clip 必然连续，永不收刀；
    - 仅 DINOv3(CLIP) 可用时启用（直方图代理阈值不可靠，不硬卡）；
    - 抽帧失败的零向量跳过该门（不误切）。
    """

    def __init__(self, embedder: Optional[DINOv3Embedder], sim_thr: float):
        self.embedder = embedder
        self.sim_thr = sim_thr
        self._emb_cache: Dict[str, object] = {}
        self.stats = {"visual": 0}

    def _embed(self, c: ClipRecord):
        import numpy as np
        if c.clip_id not in self._emb_cache:
            try:
                if self.embedder is not None:
                    self._emb_cache[c.clip_id] = self.embedder.embed(c.video_path)
                else:
                    self._emb_cache[c.clip_id] = gray_hist_embedding(c.video_path)
            except Exception:
                dim = self.embedder.EMBED_DIM if self.embedder else 32
                self._emb_cache[c.clip_id] = np.zeros(dim, dtype=np.float32)
        return self._emb_cache[c.clip_id]

    def embeddings_of(self, seg: List[ClipRecord]) -> list:
        return [self._embed(c) for c in seg]

    def __call__(self, seg: List[ClipRecord], nxt: ClipRecord) -> bool:
        last = seg[-1]
        if nxt.prev.clip_id == last.clip_id and nxt.prev.continuous:
            return False                       # 再切割兄弟，必连续
        if self.embedder is not None and self.embedder.available():
            import numpy as np
            seg_embs = self.embeddings_of(seg)
            centroid = np.mean(np.stack(seg_embs), axis=0)
            nxt_emb = self._embed(nxt)
            if np.linalg.norm(centroid) > 1e-9 and np.linalg.norm(nxt_emb) > 1e-9 \
                    and cosine(nxt_emb, centroid) < self.sim_thr:
                self.stats["visual"] += 1
                return True
        return False


def run_segment(ws: Workspace, cfg: dict,
                embedder: Optional[DINOv3Embedder] = None) -> Dict[str, int]:
    m07 = cfg.get("m07_segment", {})
    target = float(m07.get("target_segment_seconds", 40.0))
    max_s = float(m07.get("max_segment_seconds", target * 1.5))
    min_s = float(m07.get("min_segment_seconds", target * 0.5))
    sim_thr = float(m07.get("dinov3_sim_threshold", 0.6))
    weight_map = m07.get("grade_weight", {"A": 1.0, "B": 0.6, "C": 0.0})

    if embedder is None:
        embedder = DINOv3Embedder(weights=m07.get("dinov3_weights", ""))
    use_visual = bool(m07.get("visual_boundary", True)) and embedder.available()
    judge = _BoundaryJudge(embedder if use_visual else None, sim_thr)

    clips = load_records(ws.clips_jsonl, ClipRecord)
    planned = plan_segments(clips, target, max_s, min_s,
                            should_break=judge)

    segments: List[SegmentRecord] = []
    for k, seg_clips in enumerate(planned, start=1):
        sim = window_consistency(judge.embeddings_of(seg_clips))
        segments.append(SegmentRecord(
            segment_id=f"seg_{k:04d}", video_id=ws.video_id,
            clip_ids=[c.clip_id for c in seg_clips],
            start_time=seg_clips[0].start_time, end_time=seg_clips[-1].end_time,
            duration=round(sum(c.duration for c in seg_clips), 3),
            unqualified_ratio=round(
                sum(c.duration for c in seg_clips
                    if c.quality.rejected or c.quality.grade == "C")
                / max(sum(c.duration for c in seg_clips), 1e-6), 4),
            dinov3_score=round(sim, 4), continuity_ok=True,
            segment_prompt="",               # caption 由 m06 统一产出
            grade_weight=grade_weight(seg_clips, weight_map),
            status="done",
        ))

    write_jsonl(ws.segments_jsonl, [s.to_dict() for s in segments])

    # 三级闭环的结构部分：segment_ids 写回 GlobalRecord（global_prompt 归 m06）
    globals_ = load_records(ws.global_jsonl, GlobalRecord)
    g = globals_[0] if globals_ else GlobalRecord(video_id=ws.video_id)
    g.segment_ids = [s.segment_id for s in segments]
    g.status = "done"
    upsert_records(ws.global_jsonl, GlobalRecord, [g], key="video_id")

    durations = [s.duration for s in segments]
    return {"segments": len(segments),
            "proxy": int(not use_visual),
            "bound_visual": judge.stats["visual"],
            "dur_min": round(min(durations), 1) if durations else 0,
            "dur_mean": round(sum(durations) / max(len(durations), 1), 1),
            "dur_max": round(max(durations), 1) if durations else 0}


def main():
    ap = argparse.ArgumentParser(description="模块⑦：Segment 聚合（纯聚合，时长可控）")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ws = Workspace(root=os.path.join(args.workspace, args.video_id), video_id=args.video_id)
    print(f"Segment 聚合完成：{run_segment(ws, cfg)}")


if __name__ == "__main__":
    main()
