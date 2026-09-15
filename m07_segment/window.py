# -*- coding: utf-8 -*-
"""⑦ 窗口/划分逻辑（纯逻辑）。

v2：plan_segments——时长驱动的连续划分，segment 大小由参数控制
（target/max/min_seconds），边界信号可提前收刀，全片无重叠完整划分。

旧版保留：slide_windows/valid_windows——按 clip 个数滑窗 + 不合格占比
规则（窗口内不合格 clip（C 级 / rejected / 无 caption）累计时长须
< segment 总时长的 20%，否则窗口作废）。
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

from ..schema import ClipRecord


def is_qualified(clip: ClipRecord) -> bool:
    """合格 clip：未被质检剔除且非 C 级。

    （v2 起不再以"有无 caption"论合格：caption 改由 m06 在 segment
    聚合之后统一产出，聚合阶段 caption 必然尚不存在。）
    """
    if clip.quality.rejected or clip.quality.grade == "C":
        return False
    return True


def unqualified_ratio(window: List[ClipRecord]) -> float:
    total = sum(c.duration for c in window)
    if total <= 0:
        return 1.0
    bad = sum(c.duration for c in window if not is_qualified(c))
    return bad / total


def slide_windows(clips: List[ClipRecord], size: int = 5,
                  stride: int = 2) -> Iterator[Tuple[int, List[ClipRecord]]]:
    """按邻接表顺序滑窗，产出 (起始下标, 窗口clip列表)。"""
    from ..common.clip_store import sort_key
    ordered = sorted(clips, key=sort_key)
    i = 0
    while i < len(ordered):
        win = ordered[i:i + size]
        if win:
            yield i, win
        i += stride


def valid_windows(clips: List[ClipRecord], size: int = 5, stride: int = 2,
                  max_unqualified: float = 0.2) -> List[List[ClipRecord]]:
    """过滤出合格窗口。"""
    out = []
    for _, win in slide_windows(clips, size, stride):
        if len(win) < 2:                     # 单 clip 不成 segment
            continue
        if unqualified_ratio(win) < max_unqualified:
            out.append(win)
    return out


# ---------------------------------------------------------------------------
# 时长驱动的连续划分（v2：segment 大小可控）
# ---------------------------------------------------------------------------

def plan_segments(clips: List[ClipRecord],
                  target_seconds: float = 40.0,
                  max_seconds: float = 60.0,
                  min_seconds: float = 20.0,
                  should_break=None) -> List[List[ClipRecord]]:
    """按邻接表顺序把全片划分成连续 segment，时长由参数控制。

    收刀规则（按优先级）：
    1. 当前段时长 ≥ max_seconds → 硬收刀；
    2. 当前段时长 ≥ target_seconds 且"现在收"比"加上下一个 clip 再收"
       更接近 target → 收刀（贴近目标，宁短不长）；
    3. should_break(seg, next_clip) 边界信号（场景不连续/视觉不一致），
       仅在段时长 ≥ min_seconds 后生效，防止切出过碎的小段。

    返回完整划分：所有 clip 恰好属于一个 segment（单 clip 段允许存在，
    场景边界夹着的独立长镜头不应被强行合并）。
    """
    from ..common.clip_store import sort_key
    ordered = sorted(clips, key=sort_key)
    segments: List[List[ClipRecord]] = []
    cur: List[ClipRecord] = []
    cur_dur = 0.0
    for c in ordered:
        if cur:
            hard = cur_dur >= max_seconds
            closer = (cur_dur >= target_seconds
                      and abs(cur_dur - target_seconds)
                      <= abs(cur_dur + c.duration - target_seconds))
            gate = (cur_dur >= min_seconds
                    and should_break is not None
                    and should_break(cur, c))
            if hard or closer or gate:
                segments.append(cur)
                cur, cur_dur = [], 0.0
        cur.append(c)
        cur_dur += c.duration
    if cur:
        segments.append(cur)
    return segments
