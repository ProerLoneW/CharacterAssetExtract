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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_EFP_ROOT = Path(__file__).resolve().parents[1]
_VP_DIR = _EFP_ROOT / "assets" / "visual_person"
for _path in (_EFP_ROOT, _VP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

LOGGER = logging.getLogger("global_character_materializer")
SCHEMA_VERSION = "global-character-materialization-v2"
OBS_SUFFIX_PATTERN = re.compile(r"shots[/\\].*[/\\]observations[/\\].*$")
# Positive assertions of "not a physically present person" gate visual
# materialization; "uncertain" stays materialized (fail-open, recall first)
# unless --strict-entity-nature is given.
EXCLUDED_ENTITY_NATURES = {"depicted_person", "non_person"}


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
                       tracklets_root: Path,
                       face_excluded: Optional[set] = None,
                       visual_excluded: Optional[set] = None) -> dict:
    stats = {"face_copied": 0, "body_copied": 0, "rgba_copied": 0,
             "missing_files": 0, "contributing_tracklets": 0}
    face_excluded = face_excluded or set()
    visual_excluded = visual_excluded or set()

    faces = []
    bodies = []
    for tracklet in tracklets:
        tracklet_id = str(tracklet.get("tracklet_id"))
        drop_face = tracklet_id in face_excluded or tracklet_id in visual_excluded
        drop_body = tracklet_id in visual_excluded
        contributed = False
        if not drop_face:
            for entry in tracklet.get("face_gallery", []):
                faces.append((float(entry.get("quality_score", 0.0)), tracklet_id, entry))
        if not drop_body:
            for entry in tracklet.get("body_gallery", []):
                bodies.append((float(entry.get("quality_score", 0.0)), tracklet_id, entry))
        if (tracklet.get("face_gallery") and not drop_face) or \
                (tracklet.get("body_gallery") and not drop_body):
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


# --------------------------------------------------------------------------- #
# identity consistency check (optional, fail-open)
# --------------------------------------------------------------------------- #

def load_face_extractor(model_path: Path, device_id: int):
    """Reuse Stage 3's InsightFace FaceFeatureExtractor; None on failure."""
    try:
        import cv2  # noqa: F401 - availability probe
        from visual_character_registry_builder import FaceFeatureExtractor
        return FaceFeatureExtractor(model_path, device_id=device_id)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        LOGGER.warning("identity check disabled: cannot load face model %s (%s)",
                       model_path, exc)
        return None


def embed_tracklet_prototypes(tracklets: List[dict], tracklets_root: Path,
                              extractor, max_faces_per_tracklet: int = 3) -> Dict[str, dict]:
    """Quality-ranked mean face embedding per tracklet (None when unusable)."""
    import cv2
    import numpy as np

    prototypes: Dict[str, dict] = {}
    for tracklet in tracklets:
        tracklet_id = str(tracklet.get("tracklet_id"))
        gallery = sorted(tracklet.get("face_gallery", []),
                         key=lambda v: float(v.get("quality_score", 0.0)),
                         reverse=True)[:max_faces_per_tracklet]
        vectors = []
        used = []
        for entry in gallery:
            source = remap_crop_path(str(entry.get("crop_path") or ""), tracklets_root)
            if source is None:
                continue
            image = cv2.imread(str(source))
            if image is None:
                continue
            vector = extractor.embedding(image)
            if vector is None:
                continue
            vectors.append(vector)
            used.append(str(entry.get("observation_id")))
        if vectors:
            prototype = np.mean(np.stack(vectors, axis=0), axis=0)
            norm = float(np.linalg.norm(prototype))
            if norm > 0:
                prototype = prototype / norm
            else:
                continue
        prototypes[tracklet_id] = {
            "vector": prototype if vectors else None,
            "faces_embedded": len(vectors),
            "observation_ids": used,
        }
    return prototypes


def identity_consistency_check(tracklets: List[dict],
                               prototypes: Dict[str, dict],
                               sim_threshold: float) -> dict:
    """Group tracklets by face-embedding similarity; isolate outliers.

    Builds a graph over tracklet prototypes with edges where cosine
    similarity >= sim_threshold, takes the largest connected component as the
    identity core (ties broken by summed MiMo alignment confidence) and marks
    every other face-bearing tracklet as an outlier.  Characters whose faces
    share no edges at all are left untouched with status ``inconclusive``
    (fail-open: weak embeddings must not silently empty a gallery).
    """
    import numpy as np

    confidence = {str(t.get("tracklet_id")):
                  float(t.get("_alignment", {}).get("confidence", 0.0))
                  for t in tracklets}
    embedded = [tid for tid, item in prototypes.items()
                if item.get("vector") is not None]
    no_face = [str(t.get("tracklet_id")) for t in tracklets
               if str(t.get("tracklet_id")) not in prototypes
               or prototypes[str(t.get("tracklet_id"))].get("vector") is None]

    report: Dict[str, Any] = {
        "status": "not_checked",
        "core_tracklet_ids": [],
        "outlier_tracklets": [],
        "no_face_tracklet_ids": no_face,
        "face_tracklet_count": len(embedded),
        "similarity_threshold": sim_threshold,
    }
    if len(embedded) == 0:
        report["status"] = "no_face_evidence"
        return report
    if len(embedded) == 1:
        report["status"] = "single_face_tracklet_unchecked"
        report["core_tracklet_ids"] = embedded
        return report

    vectors = {tid: prototypes[tid]["vector"] for tid in embedded}
    ids = sorted(embedded)
    matrix = {a: {b: (float(np.dot(vectors[a], vectors[b])) if a != b else 1.0)
                  for b in ids} for a in ids}

    adjacency: Dict[str, set] = {tid: set() for tid in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if matrix[a][b] >= sim_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    components: List[set] = []
    seen: set = set()
    for tid in ids:
        if tid in seen:
            continue
        stack, component = [tid], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen |= component
        components.append(component)

    def component_weight(component: set) -> tuple:
        return (sum(confidence.get(tid, 0.0) for tid in component),
                len(component))

    components.sort(key=component_weight, reverse=True)
    core, rest = components[0], components[1:]
    ambiguous = bool(rest) and component_weight(rest[0]) == component_weight(core)

    if not adjacency_connected(components):  # no edges anywhere
        report["status"] = "inconclusive"
        report["core_tracklet_ids"] = ids
        report["similarity_matrix"] = matrix
        return report

    outliers = sorted({tid for component in rest for tid in component})
    report.update({
        "status": "ok_ambiguous_core" if ambiguous else "ok",
        "core_tracklet_ids": sorted(core),
        "outlier_tracklets": [{
            "tracklet_id": tid,
            "median_similarity_to_core": round(_median(
                [matrix[tid][other] for other in core]), 4),
        } for tid in outliers],
        "similarity_matrix": matrix,
    })
    return report


def adjacency_connected(components: List[set]) -> bool:
    """True when at least one component has more than one member (has edges)."""
    return any(len(component) > 1 for component in components)


def _median(values: List[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


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
    parser.add_argument("--identity-check", action="store_true",
                        help="verify with face embeddings that all tracklets "
                             "assigned to one S-ID are the same person; outlier "
                             "tracklets lose their face crops (fail-open when "
                             "the model or crops are unavailable)")
    parser.add_argument("--face-model", type=Path,
                        default=Path("local_models/insightface/models/buffalo_l/"
                                     "w600k_r50.onnx"),
                        help="same InsightFace pack Stage 3 uses")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--face-sim-threshold", type=float, default=0.40,
                        help="cosine similarity below which two tracklet face "
                             "prototypes are considered different people")
    parser.add_argument("--exclude-outlier-visual", action="store_true",
                        help="also drop outlier body/rgba crops (default keeps "
                             "them and only excludes faces)")
    parser.add_argument("--strict-entity-nature", action="store_true",
                        help="also withhold assets of tracklets whose "
                             "entity_nature is 'uncertain' (default "
                             "materializes them; depicted_person and "
                             "non_person are always withheld)")
    args = parser.parse_args()

    extractor = (load_face_extractor(args.face_model, args.device_id)
                 if args.identity_check else None)
    if args.identity_check and extractor is None:
        LOGGER.warning("continuing without identity check (model unavailable)")

    tracklets_by_id = collect_tracklets(args.tracklets)
    characters_root = args.output / "characters"
    if characters_root.exists():
        shutil.rmtree(characters_root)
    characters_root.mkdir(parents=True)

    report = []
    identity_conflicts = []
    assigned_total = 0
    excluded_natures = set(EXCLUDED_ENTITY_NATURES)
    if args.strict_entity_nature:
        excluded_natures.add("uncertain")
    nature_totals: Dict[str, int] = defaultdict(int)
    for row in _load_jsonl(args.alignment):
        status = str(row.get("status") or "")
        if status not in {"ok", "skip_model_call_fail_open"}:
            continue
        scope_id = str(row["global_scope_id"])
        by_subject: Dict[str, List[dict]] = defaultdict(list)
        depicted_by_subject: Dict[str, List[dict]] = defaultdict(list)
        for alignment in row.get("tracklet_alignments", []):
            nature = str(alignment.get("entity_nature") or "uncertain")
            nature_totals[nature] += 1
            subject_id = str(alignment.get("subject_id"))
            if subject_id in {"unknown"}:
                continue
            if nature in excluded_natures:
                # depicted/non-person tracks never enter the asset gallery,
                # but stay recorded as metadata under their asserted S-ID.
                if str(alignment.get("tracklet_id")) in tracklets_by_id:
                    depicted_by_subject[subject_id].append({
                        "tracklet_id": str(alignment.get("tracklet_id")),
                        "entity_nature": nature,
                        "confidence": float(alignment.get("confidence", 0.0)),
                        "reason": str(alignment.get("reason") or ""),
                    })
                continue
            if float(alignment.get("confidence", 0.0)) < args.min_confidence:
                continue
            tracklet = tracklets_by_id.get(str(alignment.get("tracklet_id")))
            if tracklet is None:
                LOGGER.warning("unknown tracklet %s", alignment.get("tracklet_id"))
                continue
            tracklet["_alignment"] = alignment
            by_subject[subject_id].append(tracklet)
        for subject_id, depicted in sorted(depicted_by_subject.items()):
            if subject_id not in by_subject:
                LOGGER.info("%s:%s has only depicted/non-person tracklets; "
                            "no character directory materialised",
                            scope_id, subject_id)

        for subject_id, tracklets in sorted(by_subject.items()):
            character_dir = characters_root / scope_id / subject_id
            character_dir.mkdir(parents=True, exist_ok=True)

            face_excluded: set = set()
            visual_excluded: set = set()
            identity_report: Dict[str, Any] = {"status": "not_requested"}
            if extractor is not None:
                prototypes = embed_tracklet_prototypes(tracklets, args.tracklets,
                                                       extractor)
                identity_report = identity_consistency_check(
                    tracklets, prototypes, args.face_sim_threshold)
                outlier_ids = {row["tracklet_id"]
                               for row in identity_report.get("outlier_tracklets", [])}
                visual_excluded = outlier_ids if args.exclude_outlier_visual else set()
                face_excluded = outlier_ids - visual_excluded
                if outlier_ids:
                    identity_conflicts.append({
                        "semantic_scope_id": scope_id,
                        "subject_id": subject_id,
                        "core_tracklet_ids": identity_report["core_tracklet_ids"],
                        "outlier_tracklets": identity_report["outlier_tracklets"],
                        "excluded_visual": args.exclude_outlier_visual,
                    })
                if identity_report.get("status") == "inconclusive":
                    identity_conflicts.append({
                        "semantic_scope_id": scope_id,
                        "subject_id": subject_id,
                        "review_only": True,
                        "reason": "face_tracklets_mutually_dissimilar",
                        "core_tracklet_ids": identity_report["core_tracklet_ids"],
                        "outlier_tracklets": [],
                    })
                LOGGER.info("%s:%s identity=%s outliers=%s", scope_id, subject_id,
                            identity_report.get("status"),
                            sorted(outlier_ids) or "none")

            visual_stats = copy_visual_assets(character_dir, tracklets,
                                              args.face_top_k, args.body_top_k,
                                              args.tracklets,
                                              face_excluded=face_excluded,
                                              visual_excluded=visual_excluded)
            voice_stats = copy_voice_assets(character_dir, args.voice_root, scope_id,
                                            subject_id, args.include_invalid_voice)
            confidence_values = [float(t["_alignment"].get("confidence", 0.0))
                                 for t in tracklets]
            nature_counts: Dict[str, int] = defaultdict(int)
            for t in tracklets:
                nature_counts[str(t["_alignment"].get("entity_nature")
                                  or "uncertain")] += 1
            character_json = {
                "schema_version": SCHEMA_VERSION,
                "semantic_scope_id": scope_id,
                "subject_id": subject_id,
                "tracklet_count": len(tracklets),
                "tracklet_ids": [str(t.get("tracklet_id")) for t in tracklets],
                "mean_alignment_confidence": round(
                    sum(confidence_values) / len(confidence_values), 4)
                if confidence_values else 0.0,
                "entity_nature_counts": dict(nature_counts),
                "depicted_references": depicted_by_subject.get(subject_id, []),
                "identity_check": identity_report,
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
        "entity_nature_totals": dict(nature_totals),
        "withheld_natures": sorted(excluded_natures),
        "identity_check_requested": bool(args.identity_check),
        "identity_check_active": extractor is not None,
        "identity_outlier_tracklet_count": sum(
            len(entry["outlier_tracklets"]) for entry in identity_conflicts),
        "characters": sorted(report, key=lambda item: item["scope_subject"]),
    }
    (args.output / "materialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.identity_check:
        (args.output / "identity_conflicts.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION,
                        "conflicts": identity_conflicts},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("materialised %d characters from %d tracklets",
                len(report), assigned_total)


if __name__ == "__main__":
    main()
