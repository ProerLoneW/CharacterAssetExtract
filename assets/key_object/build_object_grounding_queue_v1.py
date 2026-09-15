#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an unverified grounding queue directly from Shot Caption V4 output.

This adapter is intended for detector evaluation before Stage1.5 is available
for every shot.  It does not claim tracklet or character alignment and marks
every row unverified.  Production runs should still prefer the Stage1.5 queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFIDENCE_SCORE = {"high": 0.9, "medium": 0.65, "low": 0.35, "unknown": 0.5}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_queue(
    rows: list[dict[str, Any]], priorities: set[str]
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in rows:
        if row.get("success") is not True:
            continue
        shot_id = int(row["shot_id"])
        shot_label = f"SHOT_{shot_id:03d}"
        payload = row.get("shot_caption") or {}
        for index, candidate in enumerate(payload.get("key_object_candidates") or [], 1):
            priority = str(candidate.get("priority") or "optional").lower()
            if priority not in priorities:
                continue
            local_id = str(candidate.get("local_object_id") or f"OC{index}")
            confidence = str(candidate.get("confidence") or "unknown").lower()
            queue.append(
                {
                    "schema_version": "object-grounding-candidates-v1",
                    "candidate_id": f"{shot_label}_{local_id}",
                    "shot_id": shot_label,
                    "semantic_scope_id": None,
                    "canonical_name": candidate.get("canonical_name"),
                    "grounding_phrases": candidate.get("grounding_phrases") or [],
                    "priority": priority,
                    "related_semantic_subject_ids": candidate.get("related_subject_ids") or [],
                    "related_tracklet_ids": [],
                    "interactions": candidate.get("interactions") or [],
                    "visual_attributes": candidate.get("visual_attributes") or {},
                    "possible_states": candidate.get("possible_states") or [],
                    "observed_state": candidate.get("observed_state") or "unknown",
                    "expected_shots": [shot_label],
                    "recurring": bool(candidate.get("recurrence_hint", False)),
                    "stateful": len(candidate.get("possible_states") or []) > 1,
                    "proposal_confidence": CONFIDENCE_SCORE.get(confidence, 0.5),
                    "proposal_status": "caption_proposed",
                    "verification_status": "unverified",
                    "source": "shot_caption_v4_direct_eval",
                }
            )
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
    queue = build_queue(read_jsonl(args.shot_captions), set(args.priorities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"candidate_count": len(queue)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
