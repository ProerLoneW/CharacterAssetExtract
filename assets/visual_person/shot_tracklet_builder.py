#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build multi-frame, shot-local Tracklets from standalone shot videos.

One file, one responsibility: extend frame-local Observations into temporal
Tracklets inside a physical shot.  It uses SAM 3.1 image masks as periodic
anchors, sparse optical flow between anchors, Hungarian association, and an
ACTIVE/LOST/FINISHED state machine.  It never assigns cross-shot identity IDs.

Detection anchors are planned from the shot's usable frame count
(``_planned_redetections``) so short shots still get several independent
detection opportunities, plus a cold-start re-detection while no track exists
at all.  Admission has two lanes (``_merge_directions``): the standard
multi-detection bar, and a joint-evidence lane for shots too short to reach
it.  Rejected tracks are listed in the manifest's ``filtered_tracklets``.
Cannot-links carry ``cannot_link_evidence`` (source + detection frames) so
Stage 3 can tell detection-frame co-occurrence from propagation overlap.
Per-frame deduplication is evidence-aware (V4): an overlapping pair merges
only when independent face/mask evidence does not separate the instances,
and the decision rationale is appended to the surviving record's ``notes``.
Difficult frames additionally get a bounded local zoom-in re-detection pass
inside the observation extractor (``--disable-local-redetect`` to roll back).

Optional one-shot semantic feedback (V6, ``--semantic-feedback-requests``):
a Stage-1.5 ``semantic-feedback-requests-v1`` file lists shots whose
unmatched caption subjects carry shot-local visible-evidence windows.  The
builder then processes only those shots, converts the windows into extra
detection anchor frames merged into the normal detection plan, and writes
results to the (separate) output directory with a ``semantic_feedback``
audit record in each shot manifest.  Without the flag the pipeline is
byte-identical to the one-way chain.

Run with the project environment and SAM source on PYTHONPATH::

    PYTHONPATH="$PWD/third_party/sam3_src:$PWD/src:$PWD" \
    /path/to/general_video_annot/bin/python shot_tracklet_builder.py ...
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import entity_asset_extractor as obs_backend
import shot_tracklet_initializer as init_backend


SCHEMA_VERSION = "shot-tracklets-v2"
COLORS = [(0, 220, 255), (0, 255, 100), (255, 160, 0), (255, 0, 180),
          (180, 255, 0), (0, 120, 255), (220, 80, 80), (160, 80, 255)]


@dataclass
class TrajectoryPoint:
    frame_index: int
    timestamp_sec: float
    bbox: List[int]
    source: str                 # DETECTION or PROPAGATION
    confidence: float
    state: str


@dataclass
class RuntimeTracklet:
    tracklet_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    state: str
    observation_ids: List[str] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    last_bbox: Optional[List[int]] = None
    last_mask_path: Optional[str] = None
    missed_frames: int = 0
    age: int = 0
    face_assets: List[Dict[str, Any]] = field(default_factory=list)
    body_assets: List[Dict[str, Any]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    identity_id: Optional[str] = None
    last_seen_frame: int = 0
    propagation_confidence: float = 1.0
    _mask: Optional[np.ndarray] = field(default=None, repr=False)
    _appearance: Optional[np.ndarray] = field(default=None, repr=False)

    def export(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_mask", None)
        data.pop("_appearance", None)
        data["trajectory"] = [asdict(point) for point in self.trajectory]
        return data


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def bbox_containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box, useful for duplicate detections."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(min(aa, bb), 1e-6)


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def center_similarity(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    diagonal = max(math.hypot(a[2] - a[0], a[3] - a[1]), 1.0)
    normalized = float(np.linalg.norm(ac - bc) / diagonal)
    return max(0.0, 1.0 - normalized), normalized


def scale_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    bb = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
    return max(aa, bb) / min(aa, bb)


def association_score(
    track_box: Sequence[float], observation_box: Sequence[float],
    track_mask: Optional[np.ndarray] = None,
    observation_mask: Optional[np.ndarray] = None,
    track_appearance: Optional[np.ndarray] = None,
    observation_appearance: Optional[np.ndarray] = None,
    max_center_distance: float = 1.5, max_scale_ratio: float = 3.0,
) -> Optional[float]:
    """Hard-gated, missing-evidence-aware similarity."""
    center_score, distance = center_similarity(track_box, observation_box)
    if distance > max_center_distance or scale_ratio(track_box, observation_box) > max_scale_ratio:
        return None
    evidence = [(0.35, bbox_iou(track_box, observation_box)),
                (0.25, center_score)]
    miou = mask_iou(track_mask, observation_mask)
    if miou is not None:
        evidence.append((0.40, miou))
    if track_appearance is not None and observation_appearance is not None:
        appearance = float(np.dot(track_appearance, observation_appearance))
        if appearance < 0.15:
            return None
        evidence.append((0.20, max(0.0, min(appearance, 1.0))))
    weight = sum(item[0] for item in evidence)
    return sum(w * value for w, value in evidence) / weight


def hungarian_associate(
    tracks: Sequence[RuntimeTracklet], observations: Sequence[Dict[str, Any]],
    min_score: float = 0.25,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not tracks or not observations:
        return [], list(range(len(tracks))), list(range(len(observations)))
    scores = np.full((len(tracks), len(observations)), -1.0, np.float32)
    for ti, track in enumerate(tracks):
        for oi, obs in enumerate(observations):
            score = association_score(
                track.last_bbox, obs["bbox"], track._mask, obs.get("mask"),
                track._appearance, obs.get("appearance"),
            )
            if score is not None:
                scores[ti, oi] = score
    rows, cols = linear_sum_assignment(1.0 - scores)
    matches: List[Tuple[int, int, float]] = []
    used_t, used_o = set(), set()
    for ti, oi in zip(rows.tolist(), cols.tolist()):
        score = float(scores[ti, oi])
        if score >= min_score:
            matches.append((ti, oi, score)); used_t.add(ti); used_o.add(oi)
    return (matches,
            [i for i in range(len(tracks)) if i not in used_t],
            [i for i in range(len(observations)) if i not in used_o])


def _observation_face_box(observation: Dict[str, Any]) -> Optional[List[float]]:
    record = observation.get("record", {})
    face = record.get("face") or {}
    if not face.get("detected") or not face.get("bbox"):
        return None
    return list(face["bbox"])


def _instance_relation(new: Dict[str, Any], old: Dict[str, Any],
                       same_face_iou: float, mask_distinct_iou: float,
                       mask_same_iou: float) -> Optional[str]:
    """Independent evidence on whether two overlapping observations are one
    instance: "same", "distinct", or None when nothing informative exists.
    """
    new_face, old_face = _observation_face_box(new), _observation_face_box(old)
    if new_face is not None and old_face is not None:
        # A face is bound to at most one OBS (exclusive binding upstream), so
        # two different face boxes mean two different people; the same face
        # box on both records is a duplicated detection of one person.
        return "same" if bbox_iou(new_face, old_face) >= same_face_iou else "distinct"
    overlap = mask_iou(new.get("mask"), old.get("mask"))
    if overlap is None:
        return None
    if overlap <= mask_distinct_iou:
        return "distinct"
    if overlap >= mask_same_iou:
        return "same"
    return None


def _append_dedup_note(observation: Dict[str, Any], message: str) -> None:
    record = observation.get("record")
    if not isinstance(record, dict):
        return
    notes = record.get("notes")
    if not isinstance(notes, list):
        notes = []
        record["notes"] = notes
    notes.append(message)


def deduplicate_observations(observations: Sequence[Dict[str, Any]],
                             iou_threshold: float = .60,
                             containment_threshold: float = .85,
                             use_instance_evidence: bool = True,
                             same_face_iou: float = .50,
                             mask_distinct_iou: float = .30,
                             mask_same_iou: float = .60) -> List[Dict[str, Any]]:
    """Suppress multiple detector records describing the same person.

    Box overlap alone cannot tell a duplicate detection from two adjacent
    people, so with ``use_instance_evidence`` an overlapping pair is merged
    only when no independent evidence separates them: distinct owned faces
    (a face is bound to at most one OBS) or near-disjoint instance masks
    keep both records, while a shared face box or near-identical masks still
    merge.  Merge/keep decisions and their evidence are appended to the
    surviving record's ``notes``; no schema fields are added.
    """
    def confidence(item: Dict[str, Any]) -> float:
        record = item.get("record", {})
        face = record.get("face") or {}
        return max(float(record.get("grounding_score") or 0.0),
                   float(face.get("det_score") or 0.0))

    def obs_id(item: Dict[str, Any]) -> str:
        return str(item.get("record", {}).get("observation_id", "?"))

    kept: List[Dict[str, Any]] = []
    for observation in sorted(observations, key=confidence, reverse=True):
        absorbed_by: Optional[Tuple[Dict[str, Any], float, float,
                                    Optional[str]]] = None
        separations: List[Tuple[Dict[str, Any], float, float]] = []
        for old in kept:
            iou = bbox_iou(observation["bbox"], old["bbox"])
            containment = bbox_containment(observation["bbox"], old["bbox"])
            if iou < iou_threshold and containment < containment_threshold:
                continue
            relation = (_instance_relation(observation, old, same_face_iou,
                                           mask_distinct_iou, mask_same_iou)
                        if use_instance_evidence else None)
            if relation == "distinct":
                # Independent per-instance evidence: keep both records.
                separations.append((old, iou, containment))
                continue
            absorbed_by = (old, iou, containment, relation)
            break
        if absorbed_by is not None:
            old, iou, containment, relation = absorbed_by
            _append_dedup_note(
                old, f"dedup: absorbed {obs_id(observation)} "
                     f"(iou={iou:.2f}, containment={containment:.2f}, "
                     f"evidence={relation or 'box_only'})")
            continue
        for old, iou, containment in separations:
            _append_dedup_note(
                observation, f"dedup: kept despite overlap with {obs_id(old)} "
                             f"(iou={iou:.2f}, containment={containment:.2f}, "
                             "evidence=distinct_instance)")
        kept.append(observation)
    return kept


def reassociate_unmatched(tracks: Sequence[RuntimeTracklet], observations: Sequence[Dict[str, Any]],
                          track_indices: Sequence[int], observation_indices: Sequence[int],
                          min_score: float = .15) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Recover stale LOST tracks, but only with real spatial/appearance support."""
    candidates: List[Tuple[float, int, int]] = []
    for ti in track_indices:
        track = tracks[ti]
        for oi in observation_indices:
            obs = observations[oi]
            overlap = bbox_iou(track.last_bbox, obs["bbox"])
            appearance = None
            if track._appearance is not None and obs.get("appearance") is not None:
                appearance = float(np.dot(track._appearance, obs["appearance"]))
            if overlap < .08 and (appearance is None or appearance < .60):
                continue
            score = association_score(track.last_bbox, obs["bbox"], track._mask,
                                      obs.get("mask"), track._appearance,
                                      obs.get("appearance"), 2.5, 4.0)
            if score is not None and score >= min_score:
                candidates.append((score, ti, oi))
    matches, used_t, used_o = [], set(), set()
    for score, ti, oi in sorted(candidates, reverse=True):
        if ti in used_t or oi in used_o: continue
        matches.append((ti, oi, score)); used_t.add(ti); used_o.add(oi)
    return matches, [i for i in track_indices if i not in used_t], [i for i in observation_indices if i not in used_o]


def propagate_geometry(
    previous_gray: np.ndarray, current_gray: np.ndarray,
    bbox: Sequence[int], mask: Optional[np.ndarray],
) -> Tuple[List[int], Optional[np.ndarray], float]:
    """Propagate bbox/mask by robust median sparse optical flow."""
    h, w = current_gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi_mask = np.zeros_like(previous_gray, np.uint8)
    if mask is not None and mask.shape == previous_gray.shape:
        roi_mask[mask] = 255
    else:
        roi_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
    points = cv2.goodFeaturesToTrack(previous_gray, 120, 0.01, 5, mask=roi_mask)
    if points is None or len(points) < 4:
        return list(bbox), mask, 0.0
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    valid = status.reshape(-1) == 1
    if valid.sum() < 4:
        return list(bbox), mask, 0.0
    delta = moved[valid] - points[valid]
    dx, dy = np.median(delta.reshape(-1, 2), axis=0)
    residual = np.linalg.norm(delta.reshape(-1, 2) - [dx, dy], axis=1)
    inliers = residual <= max(2.0, float(np.median(residual) * 2.5))
    confidence = float(valid.mean() * inliers.mean() * math.exp(-np.median(residual) / 12.0))
    nx1 = int(np.clip(round(x1 + dx), 0, w - 1)); ny1 = int(np.clip(round(y1 + dy), 0, h - 1))
    nx2 = int(np.clip(round(x2 + dx), nx1 + 1, w)); ny2 = int(np.clip(round(y2 + dy), ny1 + 1, h))
    warped = None
    if mask is not None:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h),
                                flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return [nx1, ny1, nx2, ny2], warped, round(confidence, 6)


def _read_mask(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image > 0 if image is not None else None


def appearance_descriptor(frame: np.ndarray, bbox: Sequence[int],
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compact normalized HSV histogram for short-term appearance matching."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = obs_backend.clip_box(bbox, w, h)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_mask = None
    if mask is not None and mask.shape == (h, w):
        crop_mask = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], crop_mask, [16, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return (hist / norm).astype(np.float32) if norm > 0 else np.zeros(128, np.float32)


def observations_at_frame(
    shot_id: str, frame_index: int, fps: float, frame: np.ndarray,
    models: Dict[str, Any], output_root: Path, crop_padding: float,
    local_redetect: bool = True, max_local_rois: int = 3,
    dedup_instance_evidence: bool = True,
) -> List[Dict[str, Any]]:
    frame_path = output_root / "anchor_frames" / f"frame_{frame_index:06d}.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(frame_path), frame)
    sample = obs_backend.FrameSample("redetect", 0.0, frame_index, frame_index / fps, str(frame_path))
    raw, _debug = obs_backend.extract_frame_observations(
        frame_idx=frame_index, sample=sample, image_bgr=frame,
        sa2va=models["sa2va"], gdino=models["gdino"],
        face_app=models["face_app"], sam=models["sam"], use_sam=models["sam"] is not None,
        output_root=output_root, crop_padding=crop_padding,
        local_redetect=local_redetect, max_local_rois=max_local_rois,
    )
    out = []
    for index, item in enumerate(raw):
        record = asdict(item)
        record["observation_id"] = f"{shot_id}_F{frame_index:06d}_OBS{index:02d}"
        record["shot_id"] = shot_id; record["identity_id"] = None
        box = record.get("groundingdino_bbox") or record.get("sa2va_bbox")
        mask = _read_mask(record.get("mask_path"))
        out.append({"record": record, "bbox": box, "mask": mask,
                    "appearance": appearance_descriptor(frame, box, mask)})
    return deduplicate_observations(
        out, use_instance_evidence=dedup_instance_evidence)


def _add_observation(track: RuntimeTracklet, obs: Dict[str, Any], frame_index: int,
                     fps: float, score: float) -> None:
    record = obs["record"]; track.observation_ids.append(record["observation_id"])
    track.observations.append(record); track.last_bbox = list(obs["bbox"])
    track._mask = obs.get("mask"); track.last_mask_path = record.get("mask_path")
    current_appearance = obs.get("appearance")
    if current_appearance is not None:
        if track._appearance is None:
            track._appearance = current_appearance.copy()
        else:
            blended = 0.75 * track._appearance + 0.25 * current_appearance
            track._appearance = blended / max(float(np.linalg.norm(blended)), 1e-6)
    track.end_frame = frame_index; track.end_time = frame_index / fps
    track.last_seen_frame = frame_index; track.missed_frames = 0; track.state = "ACTIVE"
    track.propagation_confidence = 1.0
    point = TrajectoryPoint(frame_index, round(frame_index / fps, 6),
                            list(obs["bbox"]), "DETECTION", round(score, 6), "ACTIVE")
    if track.trajectory and track.trajectory[-1].frame_index == frame_index:
        track.trajectory[-1] = point
    else:
        track.trajectory.append(point)
    face = record.get("face") or {}
    if face.get("detected"):
        track.face_assets.append({"observation_id": record["observation_id"],
                                  "crop_path": face.get("crop_path"),
                                  "det_score": face.get("det_score")})
    if record.get("body_crop_path"):
        track.body_assets.append({"observation_id": record["observation_id"],
                                  "crop_path": record["body_crop_path"],
                                  "rgba_crop_path": record.get("rgba_crop_path"),
                                  "grounding_score": record.get("grounding_score")})


def _asset_quality(asset: Dict[str, Any], face: bool) -> float:
    path = asset.get("crop_path")
    image = cv2.imread(path) if path else None
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = image.shape[0] * image.shape[1]
    resolution = min(1.0, pixels / (160 * 160 if face else 500 * 700))
    sharpness = 1.0 - math.exp(-float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 250.0)
    confidence = float(asset.get("det_score" if face else "grounding_score") or 0.5)
    return round(0.35 * resolution + 0.35 * sharpness + 0.30 * confidence, 6)


def select_gallery(assets: Sequence[Dict[str, Any]], max_items: int,
                   face: bool = False, min_frame_gap: int = 8) -> List[Dict[str, Any]]:
    """Quality-ranked, temporally diverse gallery selection."""
    ranked = []
    for asset in assets:
        item = dict(asset); item["quality_score"] = _asset_quality(item, face)
        match = re.search(r"_F(\d+)_", item.get("observation_id", ""))
        item["frame_index"] = int(match.group(1)) if match else -1
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["quality_score"], item["frame_index"]))
    selected: List[Dict[str, Any]] = []
    for item in ranked:
        if item["quality_score"] <= 0:
            continue
        if any(abs(item["frame_index"] - kept["frame_index"]) < min_frame_gap
               for kept in selected):
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _new_track(shot_id: str, number: int, obs: Dict[str, Any], frame_index: int,
               fps: float) -> RuntimeTracklet:
    track = RuntimeTracklet(f"{shot_id}_T{number:03d}", shot_id, frame_index, frame_index,
                            frame_index / fps, frame_index / fps, "ACTIVE",
                            age=1, last_seen_frame=frame_index)
    _add_observation(track, obs, frame_index, fps, 1.0)
    return track


def render_frame(frame: np.ndarray, tracks: Sequence[RuntimeTracklet], frame_index: int) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        if track.state == "FINISHED" or not track.last_bbox: continue
        number = int(track.tracklet_id.rsplit("T", 1)[-1]); color = COLORS[number % len(COLORS)]
        x1, y1, x2, y2 = track.last_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"{track.tracklet_id} {track.state} {track.propagation_confidence:.2f}",
                    (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    cv2.putText(out, f"frame {frame_index}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    return out


def _snapshot(tracks: Sequence[RuntimeTracklet]) -> List[Dict[str, Any]]:
    return [{"tracklet_id": t.tracklet_id, "bbox": list(t.last_bbox), "state": t.state,
             "confidence": t.propagation_confidence}
            for t in tracks if t.state != "FINISHED" and t.last_bbox]


def _planned_redetections(anchor_index: int, frame_count: int,
                          redetect_interval: int, min_anchors: int) -> List[int]:
    """Deduplicated detection anchors sized to the shot's usable frames.

    Every direction gets at least ``min_anchors`` independent detection
    opportunities when it has that many frames, so a real person in a short
    shot is not filtered out for want of a third detection.  Long shots keep
    roughly the old one-per-``redetect_interval`` budget.
    """
    planned: set = set()
    for lo, hi in ((anchor_index + 1, frame_count - 1), (0, anchor_index - 1)):
        available = hi - lo + 1
        if available <= 0:
            continue
        count = min(available, max(min_anchors, math.ceil(available / redetect_interval)))
        planned.update(int(round(value)) for value in np.linspace(lo, hi, count))
    return sorted(planned)


FEEDBACK_REQUESTS_SCHEMA = "semantic-feedback-requests-v1"
MAX_FEEDBACK_ROUND = 1


def _feedback_anchor_frames(windows_sec: Sequence[Sequence[float]], fps: float,
                            frame_count: int, max_anchors: int = 6,
                            anchors_per_window: int = 3) -> List[int]:
    """Convert shot-local evidence windows into extra detection anchor frames.

    Each window contributes up to ``anchors_per_window`` evenly spaced frames
    (start/middle/end); the merged, deduplicated set is capped at
    ``max_anchors`` so a hallucination-heavy caption cannot blow up the
    per-shot detection budget.
    """
    frames: set = set()
    for window in windows_sec or []:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        try:
            start, end = float(window[0]), float(window[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            continue
        for value in np.linspace(start, end, max(1, anchors_per_window)):
            frames.add(min(max(int(round(value * fps)), 0), frame_count - 1))
    return sorted(frames)[:max_anchors]


def load_feedback_requests(path: Path | str) -> Dict[str, Any]:
    """Load and validate a Stage-1.5 feedback request file (at most one round)."""
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != FEEDBACK_REQUESTS_SCHEMA:
        raise ValueError(f"expected {FEEDBACK_REQUESTS_SCHEMA} requests file: {path}")
    round_no = int(payload.get("round", 1))
    if round_no > MAX_FEEDBACK_ROUND:
        raise ValueError(f"semantic feedback is limited to {MAX_FEEDBACK_ROUND} round(s); "
                         f"got round {round_no}")
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("feedback requests must be a list")
    return payload


def _process_direction(
    video_path: Path, frame_indices: Sequence[int], anchor_frame: np.ndarray,
    anchor_index: int, tracks: List[RuntimeTracklet], next_number: List[int],
    shot: init_backend.ShotInput, shot_root: Path, models: Dict[str, Any],
    snapshots: Dict[int, List[Dict[str, Any]]],
    cannot_links: Dict[Tuple[str, str], Dict[str, Any]],
    redetect_interval: int, max_missed_frames: int, min_match_score: float,
    min_flow_confidence: float, crop_padding: float, redetect_cooldown: int,
    planned_redetections: set, detection_frames: List[int],
    local_redetect: bool, max_local_rois: int, dedup_instance_evidence: bool,
) -> List[RuntimeTracklet]:
    previous_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    last_redetect = anchor_index
    cold_start_gap = max(2, redetect_interval // 2)
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); low_flow = False
            active = [track for track in tracks if track.state != "FINISHED"]
            for track in active:
                if track.state != "ACTIVE":
                    continue
                box, mask, conf = propagate_geometry(
                    previous_gray, gray, track.last_bbox, track._mask)
                track.last_bbox, track._mask = box, mask
                track.propagation_confidence = conf
                track.trajectory.append(TrajectoryPoint(
                    frame_index, round(frame_index / shot.fps, 6), box,
                    "PROPAGATION", conf, track.state))
                low_flow |= conf < min_flow_confidence

            planned = frame_index in planned_redetections
            cooled_down = abs(frame_index - last_redetect) >= redetect_cooldown
            # Cold start: with no track at all (e.g. the anchor frame had no
            # detection) there is no optical-flow signal to trigger a
            # re-detection, so force one every few frames until something
            # is found or the planned anchors take over.
            cold_start = not tracks and abs(frame_index - last_redetect) >= cold_start_gap
            redetect = planned or (low_flow and cooled_down) or cold_start
            if redetect:
                last_redetect = frame_index
                detection_frames.append(frame_index)
                observations = observations_at_frame(
                    shot.shot_id, frame_index, shot.fps, frame,
                    models, shot_root, crop_padding,
                    local_redetect=local_redetect,
                    max_local_rois=max_local_rois,
                    dedup_instance_evidence=dedup_instance_evidence)
                candidates_tracks = [t for t in tracks if t.state != "FINISHED"]
                matches, unmatched_t, unmatched_o = hungarian_associate(
                    candidates_tracks, observations, min_match_score)
                recovered, unmatched_t, unmatched_o = reassociate_unmatched(
                    candidates_tracks, observations, unmatched_t, unmatched_o)
                matches.extend(recovered)
                observed_ids: List[str] = []
                observed_boxes: Dict[str, Sequence[int]] = {}
                for ti, oi, score in matches:
                    track = candidates_tracks[ti]
                    _add_observation(track, observations[oi], frame_index, shot.fps, score)
                    observed_ids.append(track.tracklet_id)
                    observed_boxes[track.tracklet_id] = observations[oi]["bbox"]
                for ti in unmatched_t:
                    track = candidates_tracks[ti]
                    distance = abs(frame_index - track.last_seen_frame)
                    track.missed_frames = distance
                    track.state = "LOST" if distance <= max_missed_frames else "FINISHED"
                    if track.trajectory and track.trajectory[-1].frame_index == frame_index:
                        track.trajectory[-1].state = track.state
                for oi in unmatched_o:
                    track = _new_track(
                        shot.shot_id, next_number[0], observations[oi],
                        frame_index, shot.fps)
                    next_number[0] += 1; tracks.append(track)
                    observed_ids.append(track.tracklet_id)
                    observed_boxes[track.tracklet_id] = observations[oi]["bbox"]
                for left in range(len(observed_ids)):
                    for right in range(left + 1, len(observed_ids)):
                        a, b = observed_ids[left], observed_ids[right]
                        # Heavy overlap can be a duplicate track, reflection or
                        # detector split; it is not safe hard-negative evidence.
                        if bbox_containment(observed_boxes[a], observed_boxes[b]) < .50:
                            pair = tuple(sorted((a, b)))
                            evidence = cannot_links.setdefault(
                                pair, {"source": "detection_cooccurrence", "frames": []})
                            evidence["frames"].append(frame_index)
            snapshots[frame_index] = _snapshot(tracks)
            previous_gray = gray
    finally:
        cap.release()
    return tracks


def _best_detection_confidence(track: RuntimeTracklet) -> float:
    best = 0.0
    for record in track.observations:
        face = record.get("face") or {}
        best = max(best, float(record.get("grounding_score") or 0.0),
                   float(face.get("det_score") or 0.0))
    return best


def _merge_directions(forward: List[RuntimeTracklet], backward: List[RuntimeTracklet],
                      anchor_ids: set, fps: float, min_track_detections: int = 3,
                      min_track_frames: int = 8,
                      shot_frame_count: Optional[int] = None,
                      short_track_min_confidence: float = .5,
                      short_track_min_coverage: float = .5
                      ) -> Tuple[List[RuntimeTracklet], List[Dict[str, Any]]]:
    merged = {track.tracklet_id: track for track in forward}
    for other in backward:
        if other.tracklet_id not in anchor_ids:
            merged[other.tracklet_id] = other
            continue
        target = merged[other.tracklet_id]
        target.observations.extend(record for record in other.observations
                                   if record["frame_index"] < target.start_frame)
        target.face_assets.extend(asset for asset in other.face_assets
                                  if asset not in target.face_assets)
        target.body_assets.extend(asset for asset in other.body_assets
                                  if asset not in target.body_assets)
        points = target.trajectory + [point for point in other.trajectory
                                      if point.frame_index < target.start_frame]
        by_frame: Dict[int, TrajectoryPoint] = {}
        for point in points:
            old = by_frame.get(point.frame_index)
            if old is None or point.source == "DETECTION":
                by_frame[point.frame_index] = point
        target.trajectory = [by_frame[index] for index in sorted(by_frame)]
        target.observations.sort(key=lambda record: record["frame_index"])
        target.observation_ids = [record["observation_id"] for record in target.observations]
    result = list(merged.values())
    for track in result:
        track.trajectory.sort(key=lambda point: point.frame_index)
        if not track.trajectory:
            continue
        track.start_frame = track.trajectory[0].frame_index
        track.end_frame = track.trajectory[-1].frame_index
        track.start_time = track.start_frame / fps; track.end_time = track.end_frame / fps
        track.age = len(track.trajectory); track.state = "FINISHED"
        track.trajectory[-1].state = "FINISHED"
        track.face_assets = select_gallery(track.face_assets, 3, face=True)
        track.body_assets = select_gallery(track.body_assets, 5, face=False)
        flow = [point.confidence for point in track.trajectory
                if point.source == "PROPAGATION"]
        track.quality = {
            "detection_observations": len(track.observations),
            "trajectory_frames": len(track.trajectory),
            "coverage_ratio": round(len(track.trajectory) /
                                    max(track.end_frame - track.start_frame + 1, 1), 6),
            "mean_propagation_confidence": round(float(np.mean(flow or [1.])), 6),
        }
    # Two admission lanes.  The standard lane still requires multiple detector
    # anchors over a real trajectory.  Shots too short to ever reach that bar
    # get a joint-evidence lane instead of a deterministic drop: at least one
    # solid detection, an independent face sighting, and real temporal
    # coverage.  Propagation frames never count as detection evidence, and
    # rejected tracks are reported rather than silently dropped.
    kept: List[RuntimeTracklet] = []
    filtered: List[Dict[str, Any]] = []
    short_shot = shot_frame_count is not None and shot_frame_count < 2 * min_track_frames
    for track in result:
        detections = len(track.observations)
        frames = len(track.trajectory)
        if detections >= min_track_detections and frames >= min_track_frames:
            track.quality["admission"] = "standard"
            kept.append(track)
            continue
        reasons = []
        if detections < min_track_detections:
            reasons.append("few_detection_observations")
        if frames < min_track_frames:
            reasons.append("short_trajectory")
        coverage = float(track.quality.get("coverage_ratio") or 0.0)
        confidence = _best_detection_confidence(track)
        if (short_shot and detections >= 1 and frames >= 3
                and track.face_assets
                and confidence >= short_track_min_confidence
                and coverage >= short_track_min_coverage):
            track.quality["admission"] = "short_track_joint_evidence"
            kept.append(track)
            continue
        filtered.append({"tracklet_id": track.tracklet_id,
                         "detection_observations": detections,
                         "trajectory_frames": frames,
                         "coverage_ratio": track.quality.get("coverage_ratio"),
                         "best_detection_confidence": round(confidence, 6),
                         "has_face": bool(track.face_assets),
                         "reasons": reasons})
    kept.sort(key=lambda track: track.tracklet_id)
    return kept, filtered


def _write_overlay_video(video_path: Path, output_path: Path,
                         snapshots: Dict[int, List[Dict[str, Any]]], fps: float) -> None:
    cap = cv2.VideoCapture(str(video_path)); writer = None; frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: break
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
            for item in snapshots.get(frame_index, []):
                number = int(item["tracklet_id"].rsplit("T", 1)[-1])
                color = COLORS[number % len(COLORS)]; x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f'{item["tracklet_id"]} {item["state"]} {item["confidence"]:.2f}',
                            (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            cv2.putText(frame, f"frame {frame_index}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
            writer.write(frame); frame_index += 1
    finally:
        cap.release()
        if writer is not None: writer.release()


def _signature(track: RuntimeTracklet,
               cannot_links: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    linked = sorted({b if a == track.tracklet_id else a for a, b in cannot_links
                     if track.tracklet_id in (a, b)})
    # Per-pair provenance: which detection frames showed the two tracklets as
    # independent instances.  Stage 3 uses this to keep hard cannot-links
    # traceable to detection evidence instead of propagation-range overlap.
    evidence = {}
    for a, b in cannot_links:
        if track.tracklet_id not in (a, b):
            continue
        other = b if a == track.tracklet_id else a
        record = cannot_links[(a, b)]
        evidence[other] = {"source": record.get("source"),
                           "frames": sorted(record.get("frames") or [])}
    return {"tracklet_id": track.tracklet_id, "shot_id": track.shot_id,
            "start_frame": track.start_frame, "end_frame": track.end_frame,
            "start_time": round(track.start_time, 6), "end_time": round(track.end_time, 6),
            "observation_ids": track.observation_ids,
            "face_gallery": track.face_assets, "body_gallery": track.body_assets,
            "face_embeddings": [], "body_embeddings": [],
            "appearance_state": None, "cannot_links": linked,
            "cannot_link_evidence": evidence,
            "quality": track.quality, "identity_id": None}


def build_shot_tracklets(video_path: Path, output_root: Path, models: Dict[str, Any],
                         redetect_interval: int = 8, max_missed_frames: int = 12,
                         min_match_score: float = .25, min_flow_confidence: float = .25,
                         crop_padding: float = .05, selection_window_sec: float = 1.5,
                         candidate_frames: int = 12, redetect_cooldown: int = 4,
                         min_track_detections: int = 3, min_track_frames: int = 8,
                         short_track_min_confidence: float = .5,
                         short_track_min_coverage: float = .5,
                         local_redetect: bool = True, max_local_rois: int = 3,
                         dedup_instance_evidence: bool = True,
                         feedback_windows_sec: Optional[Sequence[Sequence[float]]] = None,
                         semantic_feedback: Optional[Dict[str, Any]] = None,
                         max_feedback_anchors: int = 6) -> Dict[str, Any]:
    shot = init_backend.inspect_shot(video_path)
    shot_root = output_root / "shots" / shot.shot_id; shot_root.mkdir(parents=True, exist_ok=True)
    start, anchor_frame, quality, candidates = init_backend.select_initialization_frame(
        video_path, fps=shot.fps, frame_count=shot.frame_count,
        selection_window_sec=selection_window_sec, candidate_frames=candidate_frames)
    anchor_observations = observations_at_frame(
        shot.shot_id, start, shot.fps, anchor_frame, models, shot_root, crop_padding,
        local_redetect=local_redetect, max_local_rois=max_local_rois,
        dedup_instance_evidence=dedup_instance_evidence)
    next_number = [0]; anchor_tracks = []
    for observation in anchor_observations:
        anchor_tracks.append(_new_track(
            shot.shot_id, next_number[0], observation, start, shot.fps))
        next_number[0] += 1
    anchor_ids = {track.tracklet_id for track in anchor_tracks}
    # Co-occurrence at a detection frame is the only hard-negative evidence:
    # two independently detected instances in the same frame are two people.
    cannot_links: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for i, a in enumerate(anchor_tracks):
        for b in anchor_tracks[i + 1:]:
            if bbox_containment(a.last_bbox, b.last_bbox) < .50:
                cannot_links[tuple(sorted((a.tracklet_id, b.tracklet_id)))] = {
                    "source": "detection_cooccurrence", "frames": [start]}
    planned = _planned_redetections(start, shot.frame_count, redetect_interval,
                                    min_track_detections)
    # V6 semantic feedback: shot-local evidence windows from Stage-1.5 become
    # extra detection anchors on top of the normal plan.  New observations
    # still pass through the unchanged detection/admission gates below; the
    # caption claim itself never creates a tracklet.
    added_anchor_frames: List[int] = []
    if feedback_windows_sec:
        added_anchor_frames = [frame for frame in _feedback_anchor_frames(
            feedback_windows_sec, shot.fps, shot.frame_count, max_feedback_anchors)
            if frame not in planned]
        if added_anchor_frames:
            planned = sorted(set(planned) | set(added_anchor_frames))
    detection_frames = [start]
    snapshots = {start: _snapshot(anchor_tracks)}
    forward = _process_direction(
        video_path, range(start + 1, shot.frame_count), anchor_frame, start,
        copy.deepcopy(anchor_tracks), next_number, shot, shot_root, models,
        snapshots, cannot_links, redetect_interval, max_missed_frames,
        min_match_score, min_flow_confidence, crop_padding, redetect_cooldown,
        set(planned), detection_frames,
        local_redetect, max_local_rois, dedup_instance_evidence)
    backward = _process_direction(
        video_path, range(start - 1, -1, -1), anchor_frame, start,
        copy.deepcopy(anchor_tracks), next_number, shot, shot_root, models,
        snapshots, cannot_links, redetect_interval, max_missed_frames,
        min_match_score, min_flow_confidence, crop_padding, redetect_cooldown,
        set(planned), detection_frames,
        local_redetect, max_local_rois, dedup_instance_evidence)
    tracks, filtered = _merge_directions(
        forward, backward, anchor_ids, shot.fps,
        min_track_detections, min_track_frames,
        shot_frame_count=shot.frame_count,
        short_track_min_confidence=short_track_min_confidence,
        short_track_min_coverage=short_track_min_coverage)
    video_out = shot_root / "tracklet_overlay.mp4"
    _write_overlay_video(video_path, video_out, snapshots, shot.fps)
    retained_ids = {track.tracklet_id for track in tracks}
    links = {pair: evidence for pair, evidence in cannot_links.items()
             if set(pair) <= retained_ids}
    signatures = [_signature(track, links) for track in tracks]
    signature_path = shot_root / "tracklet_signatures.json"
    init_backend._write_json_atomic(
        {"schema_version": SCHEMA_VERSION, "shot_id": shot.shot_id,
         "tracklet_signatures": signatures}, signature_path)
    payload = {"schema_version": SCHEMA_VERSION, "status": "ok", "shot": asdict(shot),
               "initialization_frame": {"frame_index": start, "quality": quality},
               "tracking_direction": "bidirectional",
               "frame_quality_candidates": candidates,
               "detection_plan": planned,
               "detection_frames": sorted(set(detection_frames)),
               "tracklets": [track.export() for track in tracks],
               "filtered_tracklets": filtered,
               "tracklet_signatures": str(signature_path),
               "cannot_links": [list(link) for link in sorted(links)],
               "cannot_link_evidence": [{"pair": list(pair), **evidence}
                                        for pair, evidence in sorted(links.items())],
               "overlay_video": str(video_out),
               "settings": {"redetect_interval": redetect_interval,
                            "max_missed_frames": max_missed_frames,
                            "min_match_score": min_match_score,
                            "min_flow_confidence": min_flow_confidence,
                            "redetect_cooldown": redetect_cooldown,
                            "min_track_detections": min_track_detections,
                            "min_track_frames": min_track_frames,
                            "short_track_min_confidence": short_track_min_confidence,
                            "short_track_min_coverage": short_track_min_coverage,
                            "local_redetect": local_redetect,
                            "max_local_rois": max_local_rois,
                            "dedup_instance_evidence": dedup_instance_evidence,
                            "face_gallery_top_k": 3, "body_gallery_top_k": 5}}
    if semantic_feedback is not None:
        # Feedback rounds are audited inside the existing shot-tracklets-v2
        # schema (no version bump, no caption schema change); the key is
        # absent entirely when the one-way chain runs.
        payload["semantic_feedback"] = {
            **semantic_feedback,
            "added_anchor_frames": added_anchor_frames,
            "evidence_windows_sec": [list(window) for window in (feedback_windows_sec or [])],
        }
    init_backend._write_json_atomic(payload, shot_root / "manifest.json")
    return payload


def load_models(disable_sam: bool = False) -> Dict[str, Any]:
    # Local GroundingDINO uses the Transformers 4.x BERT helper signatures;
    # the project environment currently carries Transformers 5.x.
    import torch
    from transformers import BertModel
    if not hasattr(BertModel, "get_head_mask"):
        def legacy_head(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None: return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask[None, None, :, None, None].expand(
                    num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask[:, None, :, None, None]
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask
        BertModel.get_head_mask = legacy_head
    original_extended = BertModel.get_extended_attention_mask
    if not getattr(original_extended, "_tracklet_compat", False):
        def legacy_extended(self, attention_mask, input_shape, device_or_dtype=None, dtype=None):
            # dtype 以关键字传入是合法的：entity_asset_extractor 的兼容 shim 与
            # 部分 transformers 内部调用都用 dtype= 形式（EFP 补丁，上游未改）。
            if dtype is None:
                dtype = self.dtype if isinstance(device_or_dtype, torch.device) else device_or_dtype
            if dtype is None:
                dtype = self.dtype
            return original_extended(self, attention_mask, input_shape, dtype=dtype)
        legacy_extended._tracklet_compat = True
        BertModel.get_extended_attention_mask = legacy_extended
    return {"sa2va": obs_backend.load_entity_model(), "gdino": obs_backend.load_grounding_dino(),
            "face_app": obs_backend.load_face_detector(),
            "sam": None if disable_sam else obs_backend.load_sam()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = init_backend.discover_shots(args.shots_dir, args.pattern, args.recursive)
    feedback_meta: Optional[Dict[str, Any]] = None
    requests_by_shot: Dict[str, Dict[str, Any]] = {}
    if args.semantic_feedback_requests:
        payload = load_feedback_requests(args.semantic_feedback_requests)
        round_no = int(payload.get("round", 1))
        for request in payload.get("requests", []):
            if isinstance(request, dict) and request.get("shot_id"):
                requests_by_shot[str(request["shot_id"])] = request
        by_id = {init_backend.canonical_shot_id(path.stem): path for path in paths}
        missing = sorted(set(requests_by_shot) - set(by_id))
        paths = [by_id[shot_id] for shot_id in sorted(requests_by_shot) if shot_id in by_id]
        feedback_meta = {"round": round_no,
                         "requests_source": str(Path(args.semantic_feedback_requests)
                                                .expanduser().resolve()),
                         "requested_shot_ids": sorted(requests_by_shot),
                         "missing_shot_ids": missing}
        for missing_id in missing:
            print(f"semantic feedback: requested shot {missing_id} not found under "
                  f"{args.shots_dir}", flush=True)
        # Never overwrite the original one-way-chain products: a feedback run
        # must target a separate output directory.  Re-running into the same
        # feedback directory is idempotent and allowed.
        output_pre = Path(args.output).expanduser().resolve()
        for path in paths:
            shot_id = init_backend.canonical_shot_id(path.stem)
            existing = output_pre / "shots" / shot_id / "manifest.json"
            if not existing.is_file():
                continue
            try:
                old = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(old, dict) or "semantic_feedback" not in old:
                raise RuntimeError(
                    f"{existing} already contains non-feedback tracklets; "
                    "point --output at a separate feedback directory")
    if args.limit: paths = paths[:args.limit]
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    models = load_models(args.disable_sam); results, errors = [], []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] tracking {path.name}", flush=True)
        try:
            shot_id = init_backend.canonical_shot_id(path.stem)
            request = requests_by_shot.get(shot_id)
            kwargs: Dict[str, Any] = {}
            if request is not None and feedback_meta is not None:
                kwargs = {"feedback_windows_sec": request.get("all_evidence_windows_sec"),
                          "semantic_feedback": {
                              "round": feedback_meta["round"],
                              "requests_source": feedback_meta["requests_source"],
                              "requested_subjects": list(request.get("requested_subjects", []))},
                          "max_feedback_anchors": args.max_feedback_anchors}
            results.append(build_shot_tracklets(path, output, models,
                args.redetect_interval, args.max_missed_frames, args.min_match_score,
                args.min_flow_confidence, args.crop_padding,
                args.selection_window_sec, args.candidate_frames,
                args.redetect_cooldown, args.min_track_detections,
                args.min_track_frames, args.short_track_min_confidence,
                args.short_track_min_coverage,
                not args.disable_local_redetect, args.max_local_rois,
                not args.disable_dedup_instance_evidence, **kwargs))
        except Exception as exc:
            import traceback
            errors.append({"video": str(path), "error": str(exc), "traceback": traceback.format_exc()})
            if not args.continue_on_error: break
    manifest = {"schema_version": SCHEMA_VERSION,
                "status": "ok" if not errors else "partial",
                "shots_completed": len(results),
                "tracklet_count": sum(len(x["tracklets"]) for x in results),
                "filtered_tracklet_count": sum(len(x["filtered_tracklets"]) for x in results),
                "shot_manifests": [str(output/"shots"/x["shot"]["shot_id"]/"manifest.json") for x in results],
                "errors": errors}
    if feedback_meta is not None:
        feedback_meta = {**feedback_meta,
                         "processed_shot_ids": [x["shot"]["shot_id"] for x in results]}
        manifest["semantic_feedback"] = feedback_meta
    init_backend._write_json_atomic(manifest, output / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi-frame shot-local Tracklets.")
    p.add_argument("--shots-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="*.mp4"); p.add_argument("--recursive", action="store_true")
    p.add_argument("--limit", type=int); p.add_argument("--redetect-interval", type=int, default=8)
    p.add_argument("--max-missed-frames", type=int, default=12)
    p.add_argument("--min-match-score", type=float, default=.25)
    p.add_argument("--min-flow-confidence", type=float, default=.25)
    p.add_argument("--redetect-cooldown", type=int, default=4)
    p.add_argument("--min-track-detections", type=int, default=3)
    p.add_argument("--min-track-frames", type=int, default=8)
    p.add_argument("--short-track-min-confidence", type=float, default=.5,
                   help="joint-evidence admission for short shots: best detection score floor")
    p.add_argument("--short-track-min-coverage", type=float, default=.5,
                   help="joint-evidence admission for short shots: trajectory coverage floor")
    p.add_argument("--crop-padding", type=float, default=.05)
    p.add_argument("--selection-window-sec", type=float, default=1.5)
    p.add_argument("--candidate-frames", type=int, default=12)
    p.add_argument("--disable-local-redetect", action="store_true",
                   help="disable the V4 difficult-frame local zoom-in re-detection "
                        "pass in the per-frame observation extractor")
    p.add_argument("--max-local-rois", type=int, default=3,
                   help="maximum zoom-in ROIs re-detected per difficult frame")
    p.add_argument("--disable-dedup-instance-evidence", action="store_true",
                   help="deduplicate observations by box overlap only, ignoring "
                        "independent face/mask instance evidence (rollback)")
    p.add_argument("--disable-sam", action="store_true")
    p.add_argument("--semantic-feedback-requests",
                   help="V6: path to a Stage-1.5 semantic-feedback-requests-v1 "
                        "file; only requested shots are rebuilt with extra "
                        "evidence-window detection anchors, and --output must "
                        "be a separate feedback directory")
    p.add_argument("--max-feedback-anchors", type=int, default=6,
                   help="maximum extra detection anchor frames injected per "
                        "shot from semantic feedback evidence windows")
    p.add_argument("--continue-on-error", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.semantic_feedback_requests:
        if not Path(args.semantic_feedback_requests).expanduser().is_file():
            raise SystemExit(
                f"--semantic-feedback-requests not found: {args.semantic_feedback_requests}")
        if args.max_feedback_anchors < 1:
            raise SystemExit("--max-feedback-anchors must be positive")
    result = run(args)
    print(json.dumps({k: result[k] for k in ("status", "shots_completed", "tracklet_count",
                                             "filtered_tracklet_count")},
                     ensure_ascii=False), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
