#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三路 segment 聚合（纯算法，离线）。

每路都把 global_clip_aggregator 产出的 global clip 当作输入，
按各自策略切分成 15-30s segment。三路产物落到独立目录
便于直接对比：

  segments_window/  ← 按时长驱动的连续划分（level3 m07 plan_segments）
  segments_v1/      ← mimo_segment_captioner legacy 思路（基于 shot 上下文、对话标签）
  segments_v2/      ← mimo_segment_captioner_v2 valid_boundary_guide 枚举

每路输出 segments.jsonl 行格式：
  {"segment_id": "g01_s01", "global_id": 1, "shot_ids": [1,2,3],
   "time_range_sec": [12.3, 27.6], "duration_sec": 15.3,
   "impl": "window|v1|v2", "score": 0.0}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

LOGGER = logging.getLogger("segmentation")


@dataclass
class Segment:
    segment_id: str
    global_id: int
    shot_ids: List[int]
    time_range_sec: List[float]   # global-local
    source_time_range_sec: List[float]  # 源片时间
    duration_sec: float
    impl: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# 路径 A：window-based（按时长驱动，连续划分）
# ===========================================================================

def segments_window(
    shots: List[dict],
    global_record: dict,
    target_seconds: float = 22.5,
    max_seconds: float = 30.0,
    min_seconds: float = 15.0,
) -> List[Segment]:
    """类似 level3 m07_segment.plan_segments：按时长贪心切。

    优先取最接近 target 的连续 shot 序列；超过 max 强制收；
    不足 min 时与下一个 shot 合并（除非会超 max）。
    """
    if not shots:
        return []
    g_start = global_record["time_range_sec"][0]
    out: List[Segment] = []
    cursor = 0
    while cursor < len(shots):
        # Try to find best window starting at cursor
        best_end = cursor + 1
        best_diff = float("inf")
        acc_dur = shots[cursor].get("duration_sec",
                                   shots[cursor]["time_range_sec"][1]
                                   - shots[cursor]["time_range_sec"][0])
        for end in range(cursor + 1, len(shots) + 1):
            sub = shots[cursor:end]
            acc_dur = sum(s.get("duration_sec",
                                s["time_range_sec"][1] - s["time_range_sec"][0])
                          for s in sub)
            if acc_dur > max_seconds:
                break
            diff = abs(acc_dur - target_seconds)
            if diff < best_diff:
                best_diff = diff
                best_end = end
        chosen = shots[cursor:best_end]
        dur_total = sum(s.get("duration_sec",
                              s["time_range_sec"][1] - s["time_range_sec"][0])
                        for s in chosen)
        if dur_total < min_seconds and best_end < len(shots):
            chosen = shots[cursor:best_end + 1]
            dur_total = sum(s.get("duration_sec",
                                  s["time_range_sec"][1] - s["time_range_sec"][0])
                            for s in chosen)
            # expand to include the next shot so we don't re-emit it
            best_end += 1
        # Use the shots' cumulative time_range_sec as the segment's source time.
        # dur_total always comes from the sum of real per-shot durations.
        out.append(Segment(
            segment_id=f"g{global_record['global_id']:04d}_w{len(out)+1:02d}",
            global_id=global_record["global_id"],
            shot_ids=[s["shot_id"] for s in chosen],
            time_range_sec=[
                round(chosen[0]["time_range_sec"][0] - g_start, 3),
                round(chosen[-1]["time_range_sec"][1] - g_start, 3),
            ],
            source_time_range_sec=[
                round(chosen[0]["time_range_sec"][0], 3),
                round(chosen[-1]["time_range_sec"][1], 3),
            ],
            duration_sec=round(dur_total, 3),
            impl="window",
            score=1.0 - best_diff / max(target_seconds, 1e-6),
        ))
        cursor = best_end if dur_total >= min_seconds else best_end + 1
    return out


# ===========================================================================
# 路径 B：V1 legacy（按 shot 上下文、对话标签）
# ===========================================================================

def segments_v1(
    shots: List[dict],
    global_record: dict,
    target_seconds: float = 22.5,
    max_seconds: float = 30.0,
    min_seconds: float = 15.0,
) -> List[Segment]:
    """V1 思路：尽量在 shot 边界切，且每个 segment 至少 1 个 shot。

    不像 window 那样贪心合并；如果单个 shot > max_seconds 不切分而是整体
    作为一段（保持"以 shot 为单元"的语义），由后续阶段细化。
    """
    if not shots:
        return []
    g_start = global_record["time_range_sec"][0]
    out: List[Segment] = []
    bucket: List[dict] = []
    bucket_dur = 0.0
    for s in shots:
        s_dur = s.get("duration_sec",
                      s["time_range_sec"][1] - s["time_range_sec"][0])
        if bucket and bucket_dur + s_dur > max_seconds:
            out.append(_emit_v1(bucket, g_start, global_record["global_id"], len(out) + 1))
            bucket = []
            bucket_dur = 0.0
        bucket.append(s)
        bucket_dur += s_dur
        if bucket_dur >= target_seconds:
            out.append(_emit_v1(bucket, g_start, global_record["global_id"], len(out) + 1))
            bucket = []
            bucket_dur = 0.0
    if bucket:
        out.append(_emit_v1(bucket, g_start, global_record["global_id"], len(out) + 1))
    if len(out) >= 2 and out[-1].duration_sec < min_seconds:
        last = out.pop()
        if out[-1].duration_sec + last.duration_sec <= max_seconds:
            out[-1] = _merge_segments(out[-1], last, impl="v1")
        else:
            out.append(last)
    return out


def _emit_v1(bucket: List[dict], g_start: float, gid: int, idx: int) -> Segment:
    bucket_dur = sum(s.get("duration_sec",
                           s["time_range_sec"][1] - s["time_range_sec"][0])
                     for s in bucket)
    first_s = bucket[0]["time_range_sec"][0]
    return Segment(
        segment_id=f"g{gid:04d}_v1{idx:02d}",
        global_id=gid,
        shot_ids=[s["shot_id"] for s in bucket],
        time_range_sec=[
            round(first_s - g_start, 3),
            round(first_s + bucket_dur - g_start, 3),
        ],
        source_time_range_sec=[
            round(first_s, 3),
            round(first_s + bucket_dur, 3),
        ],
        duration_sec=round(bucket_dur, 3),
        impl="v1",
        score=1.0,
    )


def _merge_segments(a: Segment, b: Segment, impl: str) -> Segment:
    return Segment(
        segment_id=a.segment_id + "+" + b.segment_id.split("_")[-1],
        global_id=a.global_id,
        shot_ids=a.shot_ids + b.shot_ids,
        time_range_sec=[a.time_range_sec[0], b.time_range_sec[1]],
        source_time_range_sec=[a.source_time_range_sec[0], b.source_time_range_sec[1]],
        duration_sec=round(b.time_range_sec[1] - a.time_range_sec[0], 3),
        impl=impl,
        score=min(a.score, b.score),
    )


# ===========================================================================
# 路径 C：V2 valid_boundary_guide（枚举所有合法切分，挑最佳）
# ===========================================================================

def segments_v2(
    shots: List[dict],
    global_record: dict,
    min_seconds: float = 15.0,
    max_seconds: float = 30.0,
    target_seconds: float = 22.5,
) -> List[Segment]:
    """V2 思路：枚举所有满足 15-30s 的连续 shot 段作为候选。

    复用 mimo_segment_captioner_v2.valid_boundary_guide 的语义（这里用
    同一段贪心近似：累加直到 ≥ min，然后在 ≤ max 时停止；如到末尾仍不足
    min 则合并到前一段）。
    """
    if not shots:
        return []
    g_start = global_record["time_range_sec"][0]
    out: List[Segment] = []
    cursor = 0
    while cursor < len(shots):
        def shot_dur(s):
            return s.get("duration_sec",
                         s["time_range_sec"][1] - s["time_range_sec"][0])
        if shot_dur(shots[cursor]) >= max_seconds:
            out.append(_emit_v2(shots[cursor:cursor + 1], g_start,
                                global_record["global_id"], len(out) + 1))
            cursor += 1
            continue
        best_end = cursor + 1
        best_diff = float("inf")
        end = cursor + 1
        dur = sum(shot_dur(shots[i]) for i in range(cursor, end))
        while end < len(shots) and dur < min_seconds:
            end += 1
            dur += shot_dur(shots[end - 1])
        for e in range(end, len(shots) + 1):
            d = sum(shot_dur(shots[i]) for i in range(cursor, e))
            if d > max_seconds:
                break
            diff = abs(d - target_seconds)
            if diff < best_diff:
                best_diff = diff
                best_end = e
        chosen = shots[cursor:best_end]
        out.append(_emit_v2(chosen, g_start, global_record["global_id"], len(out) + 1))
        cursor = best_end
    if len(out) >= 2 and out[-1].duration_sec < min_seconds:
        if out[-2].duration_sec + out[-1].duration_sec <= max_seconds:
            out[-2] = _merge_segments(out[-2], out[-1], impl="v2")
            out.pop()
    return out


def _emit_v2(bucket: List[dict], g_start: float, gid: int, idx: int) -> Segment:
    dur = sum(s.get("duration_sec",
                    s["time_range_sec"][1] - s["time_range_sec"][0])
              for s in bucket)
    first_s = bucket[0]["time_range_sec"][0]
    return Segment(
        segment_id=f"g{gid:04d}_v2{idx:02d}",
        global_id=gid,
        shot_ids=[s["shot_id"] for s in bucket],
        time_range_sec=[
            round(first_s - g_start, 3),
            round(first_s + dur - g_start, 3),
        ],
        source_time_range_sec=[
            round(first_s, 3),
            round(first_s + dur, 3),
        ],
        duration_sec=round(dur, 3),
        impl="v2",
        score=0.0,
    )


# ===========================================================================
# 主入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-clips", required=True, help="global_clips.jsonl")
    p.add_argument("--shots-jsonl", default="",
                   help="shots.jsonl（默认按目录约定推导，可显式覆盖）")
    p.add_argument("--out-dir", required=True, help="三路分段产物输出根")
    p.add_argument("--target-seconds", type=float, default=22.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--min-seconds", type=float, default=15.0)
    p.add_argument("--impls", nargs="+", default=["window", "v1", "v2"],
                   choices=["window", "v1", "v2"])
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    with open(args.global_clips) as f:
        globals_ = [json.loads(l) for l in f if l.strip()]
    LOGGER.info("loaded %d global clips", len(globals_))

    if args.shots_jsonl:
        shots_path = Path(args.shots_jsonl)
    else:
        # try sibling shots.jsonl first
        candidate = Path(args.global_clips).parent / "shots.jsonl"
        if candidate.exists():
            shots_path = candidate
        else:
            shots_path = Path(args.global_clips).parent.parent.parent / "shots" / "processed" / "shots.jsonl"
    if not shots_path.exists():
        LOGGER.error("shots.jsonl not found: %s", shots_path)
        return 1
    LOGGER.info("using shots manifest: %s", shots_path)
    with open(shots_path) as f:
        all_shots_raw = {s["shot_id"]: s for s in (json.loads(l) for l in f if l.strip())}
    LOGGER.info("loaded %d shots", len(all_shots_raw))

    # 标准化：每个 shot 的 time_range_sec 改成"按 shot.duration_sec 累计"
    # 这样 segment 输出的 source_time_range_sec 与真实 source time 一致（按 shot 序顺推）
    all_shots = {}
    cumulative = 0.0
    for sid in sorted(all_shots_raw.keys()):
        s = dict(all_shots_raw[sid])
        d = float(s.get("duration_sec", 0))
        s["time_range_sec"] = [round(cumulative, 6), round(cumulative + d, 6)]
        all_shots[sid] = s
        cumulative += d
    LOGGER.info("normalised shot time_range to cumulative; total=%.2fs", cumulative)

    # 把每个 global 的 time_range_sec 也按其内 shot 的真实累计重算
    for g in globals_:
        shots_in_g = [all_shots[sid] for sid in g["shot_ids"] if sid in all_shots]
        if shots_in_g:
            g["time_range_sec"] = [shots_in_g[0]["time_range_sec"][0],
                                   shots_in_g[-1]["time_range_sec"][1]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    impls = {
        "window": lambda gs, g: segments_window(
            gs, g, args.target_seconds, args.max_seconds, args.min_seconds),
        "v1": lambda gs, g: segments_v1(
            gs, g, args.target_seconds, args.max_seconds, args.min_seconds),
        "v2": lambda gs, g: segments_v2(
            gs, g, args.min_seconds, args.max_seconds, args.target_seconds),
    }

    summary = {}
    for impl in args.impls:
        impl_dir = out_dir / f"segments_{impl}"
        impl_dir.mkdir(parents=True, exist_ok=True)
        out_path = impl_dir / "segments.jsonl"
        all_segs: List[Segment] = []
        for g in globals_:
            shots_for_g = [all_shots[sid] for sid in g["shot_ids"] if sid in all_shots]
            if not shots_for_g:
                continue
            segs = impls[impl](shots_for_g, g)
            all_segs.extend(segs)
        with open(out_path, "w") as f:
            for s in all_segs:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        durs = [s.duration_sec for s in all_segs]
        summary[impl] = {
            "n_segments": len(all_segs),
            "duration_min": round(min(durs), 1) if durs else 0,
            "duration_mean": round(sum(durs) / max(len(durs), 1), 1),
            "duration_max": round(max(durs), 1) if durs else 0,
            "path": str(out_path),
        }
        LOGGER.info("impl=%s n=%d dur min/mean/max = %.1f/%.1f/%.1f",
                    impl, len(all_segs),
                    summary[impl]["duration_min"],
                    summary[impl]["duration_mean"],
                    summary[impl]["duration_max"])

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())