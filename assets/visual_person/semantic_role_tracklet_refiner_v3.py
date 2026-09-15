#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage1.5 V3 compatibility entry point with time-aware grounding export.

V2 already keeps proposal visibility intervals internally, but its grounding
queue projection omitted them. V3 preserves the field without modifying V2.
"""
from __future__ import annotations

from typing import Sequence

import semantic_role_tracklet_refiner_v2 as v2


SCHEMA_VERSION = "semantic-role-tracklet-object-alignment-v3"
GROUNDING_SCHEMA_VERSION = "object-grounding-candidates-v2"
_V2_BUILD_GROUNDING_QUEUE = v2.build_grounding_queue


def build_grounding_queue(results: Sequence[dict], min_confidence: float) -> list[dict]:
    queue = _V2_BUILD_GROUNDING_QUEUE(results, min_confidence)
    visibility_by_id = {
        proposal.get("candidate_id"): proposal.get("visible_time_range_sec")
        for result in results
        for proposal in result.get("object_proposals", [])
    }
    for row in queue:
        row["schema_version"] = GROUNDING_SCHEMA_VERSION
        row["visible_time_range_sec"] = visibility_by_id.get(row.get("candidate_id"))
    return queue


def main() -> int:
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.GROUNDING_SCHEMA_VERSION = GROUNDING_SCHEMA_VERSION
    v2.build_grounding_queue = build_grounding_queue
    return v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
