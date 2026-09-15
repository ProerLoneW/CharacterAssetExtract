#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a time-aware grounding queue from versioned shot captions.

V2 preserves each proposal's shot-relative visibility interval so downstream
grounding does not sample frames in which the proposed instance is absent.
It remains an unverified pre-Stage1.5 evaluation adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_object_grounding_queue_v1 as v1


def build_queue(
    rows: list[dict[str, Any]], priorities: set[str]
) -> list[dict[str, Any]]:
    queue = v1.build_queue(rows, priorities)
    source_candidates: dict[str, tuple[dict[str, Any], str]] = {}
    for row in rows:
        if row.get("success") is not True:
            continue
        shot_label = f"SHOT_{int(row['shot_id']):03d}"
        schema_version = str(row.get("schema_version") or "unknown")
        payload = row.get("shot_caption") or {}
        for index, candidate in enumerate(payload.get("key_object_candidates") or [], 1):
            local_id = str(candidate.get("local_object_id") or f"OC{index}")
            source_candidates[f"{shot_label}_{local_id}"] = (candidate, schema_version)

    for item in queue:
        candidate, schema_version = source_candidates[item["candidate_id"]]
        item["schema_version"] = "object-grounding-candidates-v2"
        item["visible_time_range_sec"] = candidate.get("visible_time_range_sec")
        item["source_caption_schema_version"] = schema_version
        item["source"] = "shot_caption_direct_time_aware_eval"
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-captions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--priorities", nargs="+", default=["must", "high"],
        choices=["must", "high", "optional", "ignore"],
    )
    args = parser.parse_args()
    rows = v1.read_jsonl(args.shot_captions)
    queue = build_queue(rows, set(args.priorities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"candidate_count": len(queue)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
