#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialise character asset directories from global-scope tracklet alignment.

Reads ``global_tracklet_alignment.jsonl`` (produced by
``global_tracklet_aligner.py``) and, for every (global scope, S-ID), copies:

visual/
    top-K face crops, body crops and RGBA cutouts from the observation
    galleries of every tracklet assigned to that S-ID, ranked by quality and
    de-duplicated per observation;
voice/
    the existing per-sentence voice previews exported by
    ``export_all_asr_audio_review.py`` (``voice_by_role_src/valid/<scope>/S#/``
    by default, optionally invalid too — pass ``--include-invalid-voice``).

No clustering, no voting, no threshold on cluster confidence: any tracklet the
global alignment assigned to an S-ID contributes evidence.  Signature
``crop_path`` values may point at another machine (the v4 run was produced on
a GPU server); paths are remapped onto the local tracklets root by the
``shots/<SHOT_ID>/observations/...`` suffix.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("global_character_materializer")
SCHEMA_VERSION = "global-character-materialization-v1"
OBS_SUFFIX_PATTERN = re.compile(r"shots[/\\].*[/\\]observations[/\\].*$")


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def remap_crop_path(crop_path: str, tracklets_root: Path) -> Optional[Path]:
    """Resolve a (possibly foreign-machine) crop path onto the local tree."""
    raw = Path(crop_path)
    if raw.exists():
        return raw
    match = OBS_SUFFIX_PATTERN.search(crop_path.replace("\\", "/"))
    if not match:
        return None
    local = tracklets_root / Path(match.group(0))
    return local if local.exists() else None


def collect_tracklets(tracklets_root: Path) -> Dict[str, dict]:
    """tracklet_id -> signature dict, discovered from shots/<SHOT>/."""
    result: Dict[str, dict] = {}
    shots_dir = tracklets_root / "shots"
    for signature_path in sorted(shots_dir.rglob("tracklet_signatures.json")):
        payload = json.loads(signature_path.read_text(encoding="utf-8"))
        for tracklet in payload.get("tracklet_signatures", []):
            result[str(tracklet.get("tracklet_id"))] = tracklet
    return result


def copy_visual_assets(character_dir: Path, tracklets: List[dict],
                       face_top_k: int, body_top_k: int,
                       tracklets_root: Path) -> dict:
    stats = {"face_copied": 0, "body_copied": 0, "rgba_copied": 0,
             "missing_files": 0, "contributing_tracklets": 0}

    faces = []
    bodies = []
    for tracklet in tracklets:
        contributed = False
        tracklet_id = str(tracklet.get("tracklet_id"))
        for entry in tracklet.get("face_gallery", []):
            faces.append((float(entry.get("quality_score", 0.0)), tracklet_id, entry))
        for entry in tracklet.get("body_gallery", []):
            bodies.append((float(entry.get("quality_score", 0.0)), tracklet_id, entry))
        if tracklet.get("face_gallery") or tracklet.get("body_gallery"):
            contributed = True
        if contributed:
            stats["contributing_tracklets"] += 1

    faces.sort(key=lambda item: item[0], reverse=True)
    bodies.sort(key=lambda item: item[0], reverse=True)

    def copy_group(ranked, sub_dir, key, counter_key, top_k):
        seen_obs = set()
        copied = 0
        for quality, tracklet_id, entry in ranked:
            if copied >= top_k:
                break
            source = remap_crop_path(str(entry.get(key) or ""), tracklets_root)
            if source is None:
                stats["missing_files"] += 1
                continue
            obs_id = str(entry.get("observation_id"))
            if obs_id in seen_obs:
                continue
            seen_obs.add(obs_id)
            target_dir = character_dir / "visual" / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{tracklet_id}_f{entry.get('frame_index', -1)}_{source.name}"
            shutil.copyfile(source, target)
            copied += 1
        stats[counter_key] += copied

    copy_group(faces, "face", "crop_path", "face_copied", face_top_k)
    copy_group(bodies, "body", "crop_path", "body_copied", body_top_k)
    copy_group(bodies, "rgba", "rgba_crop_path", "rgba_copied", body_top_k)
    return stats


def copy_voice_assets(character_dir: Path, voice_root: Path, scope_id: str,
                      subject_id: str, include_invalid: bool) -> dict:
    stats = {"voice_copied": 0, "voice_source": str(voice_root)}
    for bucket in (["valid", "invalid"] if include_invalid else ["valid"]):
        source_dir = voice_root / bucket / scope_id / subject_id
        if not source_dir.is_dir():
            continue
        target_dir = character_dir / "voice" / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        for wav in sorted(source_dir.glob("*.wav")):
            shutil.copyfile(wav, target_dir / wav.name)
            stats["voice_copied"] += 1
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True,
                        help="global_tracklet_alignment.jsonl")
    parser.add_argument("--tracklets", type=Path, required=True,
                        help="tracklets root (contains shots/<SHOT_ID>/)")
    parser.add_argument("--voice-root", type=Path, required=True,
                        help="voice_by_role_src root (contains valid/<scope>/S#/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-top-k", type=int, default=8)
    parser.add_argument("--body-top-k", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="minimum alignment confidence to contribute assets")
    parser.add_argument("--include-invalid-voice", action="store_true")
    args = parser.parse_args()

    tracklets_by_id = collect_tracklets(args.tracklets)
    characters_root = args.output / "characters"
    if characters_root.exists():
        shutil.rmtree(characters_root)
    characters_root.mkdir(parents=True)

    report = []
    assigned_total = 0
    for row in _load_jsonl(args.alignment):
        if row.get("status") != "ok":
            continue
        scope_id = str(row["global_scope_id"])
        by_subject: Dict[str, List[dict]] = defaultdict(list)
        for alignment in row.get("tracklet_alignments", []):
            subject_id = str(alignment.get("subject_id"))
            if subject_id in {"unknown", "new_subject_candidate"}:
                continue
            if float(alignment.get("confidence", 0.0)) < args.min_confidence:
                continue
            tracklet = tracklets_by_id.get(str(alignment.get("tracklet_id")))
            if tracklet is None:
                LOGGER.warning("unknown tracklet %s", alignment.get("tracklet_id"))
                continue
            tracklet["_alignment"] = alignment
            by_subject[subject_id].append(tracklet)

        for subject_id, tracklets in sorted(by_subject.items()):
            character_dir = characters_root / scope_id / subject_id
            character_dir.mkdir(parents=True, exist_ok=True)
            visual_stats = copy_visual_assets(character_dir, tracklets,
                                              args.face_top_k, args.body_top_k,
                                              args.tracklets)
            voice_stats = copy_voice_assets(character_dir, args.voice_root, scope_id,
                                            subject_id, args.include_invalid_voice)
            confidence_values = [float(t["_alignment"].get("confidence", 0.0))
                                 for t in tracklets]
            character_json = {
                "schema_version": SCHEMA_VERSION,
                "semantic_scope_id": scope_id,
                "subject_id": subject_id,
                "tracklet_count": len(tracklets),
                "tracklet_ids": [str(t.get("tracklet_id")) for t in tracklets],
                "mean_alignment_confidence": round(
                    sum(confidence_values) / len(confidence_values), 4)
                if confidence_values else 0.0,
                "visual_stats": visual_stats,
                "voice_stats": voice_stats,
                "source_alignment": str(args.alignment),
            }
            (character_dir / "role_character.json").write_text(
                json.dumps(character_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            for tracklet in tracklets:
                tracklet.pop("_alignment", None)
            assigned_total += len(tracklets)
            report.append({
                "scope_subject": f"{scope_id}:{subject_id}",
                "tracklet_count": len(tracklets),
                **visual_stats,
                "voice_copied": voice_stats["voice_copied"],
            })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "character_count": len(report),
        "assigned_tracklet_count": assigned_total,
        "characters": sorted(report, key=lambda item: item["scope_subject"]),
    }
    (args.output / "materialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("materialised %d characters from %d tracklets",
                len(report), assigned_total)


if __name__ == "__main__":
    main()
