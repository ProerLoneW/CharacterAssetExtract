#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare MiMo-division vs window-division chains produced by run_compare.sh.

Reads the shared base (shots.jsonl, global_clips.jsonl) and both chains'
segment manifests/captions, then writes comparison_report.md/.json covering:

- division: segment count, duration distribution, timeline coverage,
  boundary snap rate to shot cuts, A-vs-B boundary agreement;
- captions: success rate, description length, dialogue coverage,
  cross-level S-ID consistency against each global's subject registry;
- cost: per-stage wall time from timings.tsv.

All segment times are normalised to SOURCE-absolute seconds before comparison
(window segments expose source_time_range_sec directly; MiMo-division segments
are global-local and are offset by their global's time_range_sec[0]).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SUBJECT_RE = re.compile(r"\bS[1-9][0-9]*\b")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def find_first(pattern: str, root: Path) -> Optional[Path]:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def durations_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.mean(values), 2),
        "max": round(max(values), 2),
    }


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_shots(run_dir: Path) -> List[dict]:
    path = find_first("shots.jsonl", run_dir / "shots") or find_first("shots.jsonl", run_dir)
    return load_jsonl(path) if path else []


def load_globals(run_dir: Path) -> List[dict]:
    path = find_first("global_clips.jsonl", run_dir / "global") or find_first(
        "global_clips.jsonl", run_dir)
    return load_jsonl(path) if path else []


def window_segments(run_dir: Path) -> List[dict]:
    """Chain B plan rows → {segment_id, global_id, t0, t1 (source-absolute)}."""
    path = find_first("segments.jsonl", run_dir / "window_division")
    rows = load_jsonl(path) if path else []
    out = []
    for row in rows:
        rng = row.get("source_time_range_sec") or row.get("time_range_sec") or []
        if len(rng) != 2:
            continue
        out.append({"segment_id": str(row.get("segment_id")),
                    "global_id": row.get("global_id"),
                    "t0": float(rng[0]), "t1": float(rng[1])})
    return out


def mimo_segments(run_dir: Path, globals_by_id: Dict[int, dict]) -> List[dict]:
    """Chain A: cutter segments.jsonl (global-local) → source-absolute.

    Falls back to the captioner's segment_captions.jsonl if the cutter was
    skipped (clip export not needed for division comparison).
    """
    path = find_first("segments.jsonl", run_dir / "mimo_division")
    rows = load_jsonl(path) if path else []
    if not rows:
        cap = find_first("segment_captions.jsonl", run_dir / "mimo_division")
        rows = []
        for record in load_jsonl(cap) if cap else []:
            payload = record.get("segment_caption") or record
            for seg in payload.get("segments", []):
                if not isinstance(seg, dict):
                    continue
                seg = dict(seg)
                seg["global_id"] = record.get("global_id", seg.get("global_id"))
                rows.append({"global_id": seg.get("global_id"),
                             "segment_id": seg.get("segment_id"),
                             "time_range_sec": seg.get("time_range_sec"),
                             "description": seg.get("description", ""),
                             "dialogue": seg.get("dialogue", [])})
    out = []
    for row in rows:
        gid = row.get("global_id")
        rng = row.get("time_range_sec") or []
        if len(rng) != 2:
            continue
        g = globals_by_id.get(int(gid)) if gid is not None else None
        offset = float(g["time_range_sec"][0]) if g else 0.0
        out.append({"segment_id": str(row.get("segment_id")),
                    "global_id": gid,
                    "t0": offset + float(rng[0]),
                    "t1": offset + float(rng[1])})
    return out


def segment_caption_rows(run_dir: Path, chain: str) -> List[dict]:
    """Flatten a chain's segment_captions.jsonl into one row per segment."""
    cap = find_first("segment_captions.jsonl", run_dir / f"{chain}_division")
    out = []
    for record in load_jsonl(cap) if cap else []:
        payload = record.get("segment_caption") or record
        if isinstance(payload, dict) and isinstance(
                payload.get("segment_caption"), dict):
            payload = payload["segment_caption"]
        for seg in payload.get("segments", []):
            if not isinstance(seg, dict):
                continue
            out.append({
                "global_id": record.get("global_id"),
                "segment_id": str(seg.get("segment_id")),
                "description": str(seg.get("description") or ""),
                "dialogue": seg.get("dialogue", []),
                "key_subject_ids": seg.get("key_subject_ids", []),
            })
    return out


def registry_subjects(global_rows: List[dict]) -> Dict[int, set]:
    by_global: Dict[int, set] = {}
    for row in global_rows:
        payload = row.get("global_caption") or {}
        if isinstance(payload, dict) and isinstance(
                payload.get("global_caption"), dict):
            payload = payload["global_caption"]
        found = set()
        for subject in payload.get("subject_registry", []) or []:
            if isinstance(subject, dict):
                match = SUBJECT_RE.fullmatch(str(subject.get("subject_id") or ""))
                if match:
                    found.add(match.group(0))
        by_global[int(row["global_id"])] = found
    return by_global


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def snap_rate(bounds: List[float], targets: List[float], tol: float) -> float:
    if not bounds:
        return 0.0
    hits = 0
    for value in bounds:
        if targets and min(abs(value - t) for t in targets) <= tol:
            hits += 1
    return round(hits / len(bounds), 4)


def boundary_agreement(a: List[float], b: List[float], tol: float) -> dict:
    """How many of A's boundaries have a B boundary within tol."""
    matched = [v for v in a if b and min(abs(v - w) for w in b) <= tol]
    return {"a_boundaries": len(a), "matched": len(matched),
            "rate": round(len(matched) / len(a), 4) if a else 0.0}


def caption_stats(rows: List[dict], registry: Dict[int, set]) -> dict:
    descriptions = [r["description"] for r in rows]
    dialogue_lines = sum(len(r.get("dialogue") or []) for r in rows)
    foreign_ids = 0
    total_ids = 0
    for row in rows:
        allowed = registry.get(int(row["global_id"]), set()) \
            if row.get("global_id") is not None else set()
        for match in SUBJECT_RE.findall(json.dumps(row, ensure_ascii=False)):
            total_ids += 1
            if match not in allowed:
                foreign_ids += 1
    return {
        "captioned_segments": len(rows),
        "mean_description_chars": round(statistics.mean(
            [len(d) for d in descriptions])) if descriptions else 0,
        "dialogue_lines": dialogue_lines,
        "subject_id_mentions": total_ids,
        "subject_id_consistency": round(1 - foreign_ids / total_ids, 4)
        if total_ids else None,
    }


def timing_rows(run_dir: Path) -> List[dict]:
    path = run_dir / "timings.tsv"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 2:
            out.append({"stage": parts[0], "seconds": int(parts[1])})
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-stem", default="video")
    parser.add_argument("--snap-tol", type=float, default=0.25)
    parser.add_argument("--agree-tol", type=float, default=1.0)
    args = parser.parse_args()

    shots = load_shots(args.run_dir)
    globals_rows = load_globals(args.run_dir)
    globals_by_id = {int(g["global_id"]): g for g in globals_rows if g.get("global_id") is not None}
    shot_bounds = sorted({round(float(r["time_range_sec"][0]), 3) for r in shots
                          if r.get("time_range_sec")} |
                         {round(float(r["time_range_sec"][1]), 3) for r in shots
                          if r.get("time_range_sec")})
    total_span = (shot_bounds[-1] - shot_bounds[0]) if shot_bounds else 0.0

    seg_a = mimo_segments(args.run_dir, globals_by_id)
    seg_b = window_segments(args.run_dir)
    registry = registry_subjects(globals_rows)
    captions_a = segment_caption_rows(args.run_dir, "mimo")
    captions_b = segment_caption_rows(args.run_dir, "window")

    def chain_block(segments: List[dict], captions: List[dict]) -> dict:
        durations = [s["t1"] - s["t0"] for s in segments]
        bounds = sorted({round(s["t0"], 3) for s in segments} |
                        {round(s["t1"], 3) for s in segments})
        return {
            "division": {
                **durations_stats(durations),
                "timeline_coverage": round(sum(durations) / total_span, 4)
                if total_span else None,
                "boundary_snap_to_shot_rate": snap_rate(bounds, shot_bounds,
                                                        args.snap_tol),
            },
            "captions": caption_stats(captions, registry),
        }

    block_a = chain_block(seg_a, captions_a)
    block_b = chain_block(seg_b, captions_b)
    agreement = boundary_agreement(
        sorted({round(s["t0"], 3) for s in seg_a} | {round(s["t1"], 3) for s in seg_a}),
        sorted({round(s["t0"], 3) for s in seg_b} | {round(s["t1"], 3) for s in seg_b}),
        args.agree_tol)

    report = {
        "video_stem": args.video_stem,
        "shared_base": {
            "shots": len(shots),
            "globals": len(globals_rows),
            "shot_count_per_global": round(len(shots) / len(globals_rows), 2)
            if globals_rows else None,
        },
        "mimo_division": block_a,
        "window_division": block_b,
        "boundary_agreement_a_vs_b": agreement,
        "timings": timing_rows(args.run_dir),
    }
    (args.run_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt_div(name: str, block: dict) -> str:
        d, c = block["division"], block["captions"]
        return (f"| {name} | {d['count']} | {d.get('median', '-')}s | "
                f"{d.get('max', '-')}s | {d.get('timeline_coverage', '-')} | "
                f"{d.get('boundary_snap_to_shot_rate', '-')} | "
                f"{c['captioned_segments']} | {c['mean_description_chars']} | "
                f"{c['dialogue_lines']} | {c['subject_id_consistency']} |")

    md = [
        f"# Division comparison — {args.video_stem}",
        "",
        f"Shared base: {len(shots)} shots, {len(globals_rows)} globals, "
        f"span {round(total_span, 1)}s (480p/15fps preprocessed).",
        "",
        "| chain | segments | median dur | max dur | coverage | snap@0.25s | "
        "captions | mean chars | dialogue lines | S-ID consistency |",
        "|---|---|---|---|---|---|---|---|---|---|",
        fmt_div("A MiMo", block_a),
        fmt_div("B window", block_b),
        "",
        f"Boundary agreement (A→B within {args.agree_tol}s): "
        f"{agreement['matched']}/{agreement['a_boundaries']} "
        f"({agreement['rate']}).",
        "",
        "Per-stage wall time (s):",
        "",
        "| stage | seconds |",
        "|---|---|",
        *[f"| {row['stage']} | {row['seconds']} |" for row in report["timings"]],
        "",
        "Reading guide: snap rate ≈1 means boundaries respect camera cuts; "
        "low S-ID consistency means segment captions invent subjects absent "
        "from the global registry; coverage ≈1 means no timeline gaps.",
    ]
    (args.run_dir / "comparison_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.run_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
