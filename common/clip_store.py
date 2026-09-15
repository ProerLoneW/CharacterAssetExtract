# -*- coding: utf-8 -*-
"""clip 序列的邻接表（双向链表）维护。

对应框架图 ②④ 的连续性标记规则：
  - 序列首 clip：prev = {clip_id: None, continuous: False}
  - 序列末 clip：next = {clip_id: None, continuous: False}
  - 相邻两个原始镜头（不同 parent）：continuous = False
  - ④ 再切割产生的相邻子 clip（同 parent_clip_id）：continuous = True
  - 切割时：第一个子 clip 的 prev 继承原 clip 的 prev；
           最后一个子 clip 的 next 继承原 clip 的 next。

把整片镜头序列当成双向链表：切割 = 链表节点替换为一段子链，
再调用 rebuild_links 统一重挂指针，避免手工改表出错。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..schema import AdjacentLink, ClipRecord


def sort_key(clip: ClipRecord):
    # 先按源片时间，再按 clip_id 稳定次序（002-1 < 002-2 < 002-10 需自然序）
    return (clip.start_frame, _natural_key(clip.clip_id))


def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def same_parent(a: ClipRecord, b: ClipRecord) -> bool:
    """两个 clip 是否为同一父 clip 再切割出的相邻子 clip。"""
    pa = a.parent_clip_id or a.clip_id
    pb = b.parent_clip_id or b.clip_id
    return (a.parent_clip_id is not None or b.parent_clip_id is not None) and pa == pb


def rebuild_links(clips: List[ClipRecord]) -> List[ClipRecord]:
    """按时间序重挂整链邻接指针（原地修改并返回排序后的列表）。"""
    clips = sorted(clips, key=sort_key)
    n = len(clips)
    for i, c in enumerate(clips):
        if i == 0:
            c.prev = AdjacentLink(clip_id=None, continuous=False)
        else:
            c.prev = AdjacentLink(clip_id=clips[i - 1].clip_id,
                                  continuous=same_parent(clips[i - 1], c))
        if i == n - 1:
            c.next = AdjacentLink(clip_id=None, continuous=False)
        else:
            c.next = AdjacentLink(clip_id=clips[i + 1].clip_id,
                                  continuous=same_parent(c, clips[i + 1]))
    return clips


def make_sub_clip(parent: ClipRecord, index: int,
                  start_frame: int, end_frame: int) -> ClipRecord:
    """由父 clip 派生子 clip（只填身份与帧区间，邻接指针交给 rebuild_links）。

    命名：002 -> 002-1 / 002-2；若对子 clip 再切（002-1 -> 002-1-1）同样成立。
    parent_clip_id 始终指向最初的那个镜头（002），保证 same_parent 判定正确。
    """
    sub = ClipRecord(
        clip_id=f"{parent.clip_id}-{index}",
        video_id=parent.video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=parent.fps,
        start_time=start_frame / parent.fps if parent.fps else 0.0,
        end_time=end_frame / parent.fps if parent.fps else 0.0,
        duration=(end_frame - start_frame) / parent.fps if parent.fps else 0.0,
        parent_clip_id=parent.parent_clip_id or parent.clip_id,
        video_path=parent.video_path,
        audio_path=parent.audio_path,
    )
    return sub


def split_clip(clips: List[ClipRecord], clip_id: str,
               cut_frames: List[int]) -> List[ClipRecord]:
    """链表插入：把 clip_id 替换为按 cut_frames 切出的子 clip 序列。

    cut_frames 为父 clip 帧区间内部的切点（不含端点）。
    返回重挂好邻接指针的新全量列表。
    """
    clips = sorted(clips, key=sort_key)
    idx = next((i for i, c in enumerate(clips) if c.clip_id == clip_id), None)
    if idx is None:
        raise KeyError(f"clip not found: {clip_id}")
    parent = clips[idx]
    for cf in cut_frames:
        if not (parent.start_frame < cf < parent.end_frame):
            raise ValueError(
                f"切点 {cf} 不在 clip {clip_id} 帧区间 "
                f"({parent.start_frame}, {parent.end_frame}) 内")
    bounds = [parent.start_frame] + sorted(cut_frames) + [parent.end_frame]
    subs = [
        make_sub_clip(parent, i + 1, bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]
    new_list = clips[:idx] + subs + clips[idx + 1:]
    return rebuild_links(new_list)


def clips_by_id(clips: List[ClipRecord]) -> Dict[str, ClipRecord]:
    return {c.clip_id: c for c in clips}
