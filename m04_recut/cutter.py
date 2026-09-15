# -*- coding: utf-8 -*-
"""④ 切刀点选择（纯逻辑，可单测）。

规则（框架图 + 说明文档）：
1. 优先在 ASR 句间静音间隙（>300ms）下刀（取间隙中点）；
2. 无静音间隙时在词边界下刀（词级时间戳相邻词的间隙）；
3. 持续说话 >20s（一段 ASR 横跨目标切点）→ 直接切断为两个 clip；
4. 切点左右各留 3–5 帧重叠（物理切文件时，防止音频削头削尾）。

贪心策略：从当前起点出发，目标切点 = 起点 + max_seconds，
在 [起点+min_segment, 起点+max_seconds] 窗口内选**最靠近目标**的静音间隙中点；
没有静音间隙找词边界；都没有则在目标点硬切。推进起点递归直到整段 ≤ max。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..schema import ASRSegment, ClipRecord


def silence_gaps(segments: List[ASRSegment], min_gap: float = 0.3) -> List[Tuple[float, float]]:
    """相邻说话段之间的静音间隙 [(gap_start, gap_end), ...]。"""
    segs = sorted(segments, key=lambda s: s.start)
    gaps = []
    for a, b in zip(segs, segs[1:]):
        if b.start - a.end >= min_gap:
            gaps.append((a.end, b.start))
    return gaps


def word_boundaries(segments: List[ASRSegment]) -> List[float]:
    """词级时间戳给出的可下刀边界（词与词之间）。"""
    bounds: List[float] = []
    for seg in segments:
        words = sorted(seg.words, key=lambda w: w.start)
        for w1, w2 in zip(words, words[1:]):
            if w2.start > w1.end:
                bounds.append((w1.end + w2.start) / 2)
    return bounds


def find_cut_points(duration: float, segments: List[ASRSegment],
                    max_seconds: float = 20.0, min_gap: float = 0.3,
                    min_segment: float = 2.0) -> List[float]:
    """返回 clip 内相对秒切点列表（升序）。"""
    if duration <= max_seconds:
        return []
    gaps = silence_gaps(segments, min_gap)
    gap_mids = [(s + e) / 2 for s, e in gaps]
    words = word_boundaries(segments)

    cuts: List[float] = []
    cursor = 0.0
    while duration - cursor > max_seconds:
        target = cursor + max_seconds
        lo = cursor + min_segment
        # 1) 静音间隙中点：窗口内最靠后（信息损失最小）
        cand = [m for m in gap_mids if lo <= m <= target]
        if cand:
            cut = max(cand)
        else:
            # 2) 词边界
            cand_w = [b for b in words if lo <= b <= target]
            cut = max(cand_w) if cand_w else target  # 3) 硬切
        cuts.append(round(cut, 3))
        cursor = cut
    return cuts
