# -*- coding: utf-8 -*-
"""④ clip 再切割主流程。

输入：③ 之后的 clip 宇宙（clips.jsonl，含 ASR）。
对时长 >20s 的 clip：
  1. find_cut_points 选切刀点（相对秒 → 源片帧号）；
  2. clip_store.split_clip 链表插入（002 → 002-1/002-2，邻接指针全链重挂）；
  3. 子 clip 继承父 clip 的 ASR（按区间裁剪 + 时间戳平移）与质检卡
     （时长项重判，标记 need_qc_recheck，闭环回质检）；
  4. 物理切出子 clip 的视频/48kHz 音频（切点两侧留重叠帧）。
产出：最终 clip 宇宙 v1 冻结（status=recut_done），此后所有 clip 级计算只对它跑。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from ..common import ffmpeg as ff
from ..common.clip_store import clips_by_id, split_clip
from ..common.jsonl import load_records, write_jsonl
from ..common.workspace import Workspace
from ..schema import ASRResult, ASRSegment, ClipRecord, QualityScore, WordTS
from .cutter import find_cut_points


def inherit_asr(parent: ClipRecord, sub: ClipRecord) -> ASRResult:
    """把父 clip 的 ASR 按子区间裁剪并平移为子 clip 内相对时间。"""
    shift = sub.start_time - parent.start_time
    a, b = shift, shift + sub.duration   # 子区间在父 clip 坐标系下
    res = ASRResult(engine=parent.asr.engine, main_lang=parent.asr.main_lang)
    for seg in parent.asr.segments:
        s, e = max(seg.start, a), min(seg.end, b)
        if e - s <= 0.05:                # 擦边碎片丢弃
            continue
        words = [WordTS(word=w.word, start=round(w.start - a, 3),
                        end=round(w.end - a, 3))
                 for w in seg.words if w.end > a and w.start < b]
        res.segments.append(ASRSegment(
            start=round(s - a, 3), end=round(e - a, 3), text=seg.text,
            words=words, confidence=seg.confidence,
            speaker_local=seg.speaker_local, lang=seg.lang))
    for ev in parent.asr.audio_events:
        s, e = max(ev.start, a), min(ev.end, b)
        if e - s > 0.05:
            from ..schema import AudioEvent
            res.audio_events.append(AudioEvent(
                start=round(s - a, 3), end=round(e - a, 3),
                label=ev.label, confidence=ev.confidence))
    return res


def inherit_quality(parent: ClipRecord, sub: ClipRecord, min_seconds: float = 2.0) -> None:
    """子 clip 继承父质检卡，仅时长项重判；标记回炉质检。"""
    import copy
    sub.quality = copy.deepcopy(parent.quality)
    hard = sub.duration < min_seconds
    sub.quality.duration = QualityScore(
        score=0.0 if hard else 1.0, hard_reject=hard,
        detail=f"duration={sub.duration:.2f}s (recut)")
    extra = dict(sub.quality.extra or {})
    extra["need_qc_recheck"] = True
    extra["need_recut"] = False
    sub.quality.extra = extra


def _recut_cut_job(job: tuple) -> tuple:
    """进程池 worker：切出子 clip 视频 + 抽 48kHz 音频。"""
    src, dst_video, dst_audio, rel_start, rel_end, fps, overlap = job
    try:
        ff.cut_video(src, dst_video, rel_start, rel_end, fps=fps, overlap=overlap)
        ff.extract_audio_48k(dst_video, dst_audio)
        return dst_video, ""
    except Exception as e:
        return dst_video, str(e)


def recut_workspace(ws: Workspace, cfg: dict, cut_files: bool = True,
                    source_video: str = "") -> List[ClipRecord]:
    m04 = cfg.get("m04_recut", {})
    max_s = m04.get("max_clip_seconds", 20.0)
    min_gap = m04.get("silence_gap_min_ms", 300) / 1000.0
    overlap_frames = m04.get("overlap_frames", 4)
    workers = int(m04.get("cut_workers", 1))

    clips = load_records(ws.clips_jsonl, ClipRecord)
    jobs: List[tuple] = []
    for c in list(clips):
        if c.quality.rejected or c.duration <= max_s:
            continue
        cuts = find_cut_points(c.duration, c.asr.segments, max_s, min_gap)
        if not cuts:
            continue
        cut_frames = [c.start_frame + int(round(t * c.fps)) for t in cuts]
        parent_backup = clips_by_id(clips)[c.clip_id]
        root_parent = parent_backup.parent_clip_id or parent_backup.clip_id
        clips = split_clip(clips, c.clip_id, cut_frames)
        subs = [x for x in clips
                if x.parent_clip_id == root_parent
                and x.clip_id.startswith(c.clip_id + "-")]
        for sub in subs:
            inherit_quality(parent_backup, sub)
            sub.asr = inherit_asr(parent_backup, sub)
            sub.status = "recut_done"
            sub.thumbnail = ""
            if cut_files and parent_backup.video_path:
                sub.video_path = ws.clip_video_path(sub.clip_id)
                ov = overlap_frames / (sub.fps or 25.0)
                # 父 clip 文件的本地时间轴从 0 开始，切点换算为父内相对秒
                rel_start = sub.start_time - parent_backup.start_time
                rel_end = sub.end_time - parent_backup.start_time
                sub.audio_path = ws.clip_audio_path(sub.clip_id)
                jobs.append((parent_backup.video_path, sub.video_path,
                             sub.audio_path, rel_start, rel_end,
                             int(sub.fps or 25), ov))

    if jobs:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                futs = [ex.submit(_recut_cut_job, j) for j in jobs]
                failed = 0
                for fu in as_completed(futs):
                    _, err = fu.result()
                    if err:
                        failed += 1
                        print(f"[m04] 切割失败：{err}")
            if failed:
                print(f"[m04] {failed}/{len(jobs)} 个子 clip 切割失败")
        else:
            for j in jobs:
                _recut_cut_job(j)
    write_jsonl(ws.clips_jsonl, [c.to_dict() for c in clips])
    return clips


def main():
    ap = argparse.ArgumentParser(description="模块④：clip 再切割（≤20s，链表更新）")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-cut-files", action="store_true", help="只更新标注不切文件")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ws = Workspace(root=os.path.join(args.workspace, args.video_id), video_id=args.video_id)
    clips = recut_workspace(ws, cfg, cut_files=not args.no_cut_files)
    over = [c.clip_id for c in clips
            if c.duration > cfg.get("m04_recut", {}).get("max_clip_seconds", 20.0)
            and not c.quality.rejected]
    print(f"再切割完成：{len(clips)} 个 clip；仍超长的: {over or '无'}")


if __name__ == "__main__":
    main()
