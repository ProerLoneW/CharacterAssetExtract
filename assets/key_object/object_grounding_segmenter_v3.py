#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-aware GroundingDINO + real SAM3.1 object grounding V3/V4.

Unlike V2, frame sampling is constrained to each caption proposal's
``visible_time_range_sec`` when present. The interval is shot-relative.

V4 (default; ``--window-resample-max 0`` rolls back to exact V3 behavior):
when the first pass over the sampled frames fails or stays ambiguous, up to
``--window-resample-max`` extra frames are re-sampled *inside the same visible
window*, preferring sharp frames because blur/occlusion causes most first-pass
failures.  The window itself, the caption candidates and their priorities are
never changed.  Instance arbitration then links per-frame clusters into
cross-frame tracks and selects by positional continuity plus detection
evidence instead of a single-frame best cluster; low consistency stays
ambiguous and emits no asset.  V4 rows keep every V3 field and add
``sampling_pass``/``frame_sharpness``/``resample``/``cross_frame_support``
records under schema ``object-observation-v4``; the input queue stays
``object-grounding-candidates-v2``.

``--build-reference-sets`` switches to a post-hoc mode that never runs
detection or segmentation: it reads an existing grounding output directory,
embeds each frame's RGBA foreground with the local DINOv3 weights, and writes
an ordered 3-5 frame reference set per candidate to ``reference_sets.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import object_grounding_segmenter_v1 as common
import object_grounding_segmenter_v2 as v2


def frame_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_video_frames_in_window(
    video_path: Path,
    count: int,
    visible_time_range_sec: Any = None,
    *,
    exclude_indices: Any = None,
    prefer_sharp: bool = False,
    scan_limit: int = 64,
) -> tuple[list[dict[str, Any]], dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open shot video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"video has no frames: {video_path}")

    duration = (frame_count - 1) / fps
    start_sec, end_sec = 0.0, duration
    range_applied = False
    if isinstance(visible_time_range_sec, (list, tuple)) and len(visible_time_range_sec) == 2:
        try:
            requested_start = float(visible_time_range_sec[0])
            requested_end = float(visible_time_range_sec[1])
        except (TypeError, ValueError):
            requested_start, requested_end = 0.0, duration
        if np.isfinite(requested_start) and np.isfinite(requested_end) and requested_end >= requested_start:
            start_sec = max(0.0, min(duration, requested_start))
            end_sec = max(start_sec, min(duration, requested_end))
            range_applied = True

    start_frame = int(np.ceil(start_sec * fps))
    end_frame = int(np.floor(end_sec * fps))
    start_frame = max(0, min(frame_count - 1, start_frame))
    end_frame = max(start_frame, min(frame_count - 1, end_frame))
    excluded = {int(value) for value in (exclude_indices or [])}
    sharpness_by_index: dict[int, float] = {}
    meta_extra: dict[str, Any] = {}
    if prefer_sharp:
        pool = [
            index
            for index in range(start_frame, end_frame + 1)
            if index not in excluded
        ]
        if len(pool) > scan_limit:
            pool = sorted(
                {pool[int(round(value))] for value in np.linspace(0, len(pool) - 1, scan_limit)}
            )
        scored: list[tuple[int, float]] = []
        for frame_index in pool:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if ok and image is not None:
                scored.append((frame_index, frame_sharpness(image)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        wanted = sorted(frame_index for frame_index, _ in scored[:count])
        sharpness_by_index = dict(scored)
        meta_extra = {
            "sharpness_preferred": True,
            "excluded_frame_indices": sorted(excluded),
            "candidate_pool_size": len(pool),
        }
    else:
        wanted = sorted(
            {
                int(round(value))
                for value in np.linspace(start_frame, end_frame, count)
            }
            - excluded
        )
    frames: list[dict[str, Any]] = []
    for frame_index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if ok and image is not None:
            frame = {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 6),
                "image_bgr": image,
            }
            if frame_index in sharpness_by_index:
                frame["sharpness"] = round(sharpness_by_index[frame_index], 6)
            frames.append(frame)
    capture.release()
    return frames, {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 6),
        "visibility_range_applied": range_applied,
        "effective_sample_range_sec": [
            round(start_frame / fps, 6), round(end_frame / fps, 6)
        ],
        **meta_extra,
    }


def _detect_clusters(
    dino: Any, phrases: Sequence[str], frame: dict[str, Any], cluster_iou: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_detections: list[dict[str, Any]] = []
    for phrase in phrases:
        raw_detections.extend(dino.detect(frame["image_bgr"], phrase))
    clusters = common.cluster_detections(raw_detections, cluster_iou)
    return raw_detections, clusters


def _select_track(
    records: Sequence[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict | None, str, bool]:
    tracks = common.link_clusters_across_frames(
        [(record["frame"]["frame_index"], record["clusters"]) for record in records],
        args.track_link_iou,
    )
    # Continuity is only measurable when multiple frames were sampled; a
    # one-frame window keeps single-frame evidence admissible.
    min_support = min(args.min_track_support_frames, max(1, len(records)))
    return common.select_cluster_track(
        tracks, min_support, args.track_ambiguity_tolerance
    )


def _base_observation(
    schema_version: str,
    candidate: dict[str, Any],
    candidate_id: str,
    shot_id: str,
    video_path: Path,
    requested_range: Any,
    sampling: dict[str, Any],
    frame: dict[str, Any],
    phrases: Sequence[str],
    raw_detections: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    strategy: str,
    ambiguous: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "candidate_id": candidate_id,
        "semantic_scope_id": candidate.get("semantic_scope_id"),
        "canonical_name": candidate.get("canonical_name"),
        "visual_attributes": candidate.get("visual_attributes"),
        "possible_states": candidate.get("possible_states"),
        "observed_state": candidate.get("observed_state"),
        "visible_time_range_sec": candidate.get("visible_time_range_sec"),
        "shot_id": shot_id,
        "shot_video_path": str(video_path.resolve()),
        "frame_index": frame["frame_index"],
        "timestamp_sec": frame["timestamp_sec"],
        "requested_visible_time_range_sec": requested_range,
        "effective_sample_range_sec": sampling["effective_sample_range_sec"],
        "visibility_range_applied": sampling["visibility_range_applied"],
        "grounding_phrases": list(phrases),
        "all_grounding_detections": raw_detections,
        "grounding_clusters": clusters,
        "selection_strategy": strategy,
        "ambiguous_instance": ambiguous,
        "segmentation_backend": "sam3.1_image_geometric_prompt",
        "sam31_checkpoint": str(args.sam31_checkpoint.resolve()),
    }


def _no_selection(base: dict[str, Any], status: str) -> None:
    base.update({
        "status": status,
        "selected_cluster_id": None,
        "selected_bbox_xyxy": None,
        "mask_bbox_xyxy": None,
        "mask_area_pixels": 0,
        "mask_area_ratio": 0.0,
        "sam_score": None,
        "sam_details": {},
        "assets": {},
    })


def _segment_selection(
    base: dict[str, Any],
    sam: Any,
    args: argparse.Namespace,
    frame: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    mask, sam_score, sam_details = sam.segment(
        frame["image_bgr"], selected["bbox_xyxy"]
    )
    stem = f"{base['candidate_id']}_{base['shot_id']}_F{int(frame['frame_index']):06d}"
    assets = common.save_assets(
        args.output, stem, frame["image_bgr"], selected["bbox_xyxy"], mask
    )
    area = int(mask.sum())
    base.update({
        "status": "ok" if area else "no_mask",
        "selected_cluster_id": selected["cluster_id"],
        "selected_bbox_xyxy": selected["bbox_xyxy"],
        "selected_grounding_score": selected["max_score"],
        "selected_phrase_vote_count": selected["phrase_vote_count"],
        "mask_bbox_xyxy": common.mask_bbox(mask),
        "mask_area_pixels": area,
        "mask_area_ratio": round(area / mask.size, 8),
        "sam_score": round(sam_score, 6),
        "sam_details": sam_details,
        "assets": assets,
    })


# ---------------------------------------------------------------------------
# Post-hoc reference-set builder (--build-reference-sets).
#
# Reads an existing grounding output directory (object_observations.jsonl plus
# the rendered assets) without re-running detection or segmentation, embeds
# each frame's segmented foreground with the local DINOv3 weights, and picks
# an ordered 3-5 frame reference set per candidate.  The normal segmentation
# flow above never calls this code.
# ---------------------------------------------------------------------------

DEFAULT_DINOV3_DIR = v2.ROOT / "local_models/dinov3-vitl16"
GRAY_BACKGROUND_VALUE = 128


def composite_rgba_on_gray(rgba_path: Path, gray: int = GRAY_BACKGROUND_VALUE):
    """Flatten a saved RGBA cutout onto a fixed gray background, in memory."""
    from PIL import Image

    with Image.open(rgba_path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    rgb = rgba[:, :, :3] * alpha + float(gray) * (1.0 - alpha)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


class Dinov3ForegroundEmbedder:
    """Offline DINOv3 embeddings of segmented foregrounds from local weights.

    Mirrors the loading and pooling pattern of
    ``assets/scene/important_scene_frame_extractor.Dinov3SceneEmbedder``:
    Hugging Face ``local_files_only`` weights, one CLS token plus four
    register tokens skipped, spatial patch tokens mean-pooled and
    L2-normalized.
    """

    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"DINOv3 model directory does not exist: {model_path}"
            )
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for local DINOv3"
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DINO device {device!r} requested but CUDA is unavailable"
            )
        self.model_path = model_path.resolve()
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True
        ).eval().to(self.device)

    def embed_rgba_paths(self, paths: Sequence[Path]) -> np.ndarray:
        torch = self.torch
        embeddings: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = [
                composite_rgba_on_gray(path)
                for path in paths[start:start + self.batch_size]
            ]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {
                name: value.to(self.device) for name, value in inputs.items()
            }
            with torch.inference_mode():
                tokens = self.model(**inputs).last_hidden_state
                features = tokens[:, 5:, :].mean(dim=1)
                features = torch.nn.functional.normalize(features, dim=1)
            embeddings.append(features.float().cpu().numpy())
        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(embeddings, axis=0).astype(np.float32)


def _relative_asset(path_value: Any, root: Path) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _pairwise_cosine_distance(features: np.ndarray) -> np.ndarray:
    similarity = np.clip(features @ features.T, -1.0, 1.0)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _flag_mask_area_outliers(
    ratios: Sequence[float], full_frame_ratio: float
) -> list[list[str]]:
    """Tukey IQR fences within the group, plus an absolute full-frame rule."""
    reasons: list[list[str]] = [[] for _ in ratios]
    values = np.asarray(ratios, dtype=np.float64)
    for index, ratio in enumerate(ratios):
        if ratio > full_frame_ratio:
            reasons[index].append(
                f"mask_area_ratio={ratio:.4f} exceeds the full-frame dominance "
                f"threshold {full_frame_ratio}"
            )
    if len(values) >= 2:
        q1, q3 = (float(v) for v in np.percentile(values, [25.0, 75.0]))
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for index, ratio in enumerate(ratios):
                if ratio < low or ratio > high:
                    reasons[index].append(
                        f"mask_area_ratio={ratio:.4f} outside the group IQR "
                        f"fences [{low:.4f}, {high:.4f}] (Q1={q1:.4f}, "
                        f"Q3={q3:.4f}, IQR={iqr:.4f})"
                    )
    return reasons


def _select_references(
    frames: list[dict[str, Any]],
    distance_matrix: np.ndarray,
    near_duplicate_similarity: float,
    max_references: int,
    duplicate_penalty: float,
) -> list[dict[str, Any]]:
    """Order eligible frames into a 3-5 (fewer when scarce) reference list.

    Eligibility excludes mask-area outliers and ambiguous-instance frames.
    Rank 1 is the eligible frame with the smallest median foreground distance
    to the other ok frames (ties broken by SAM score).  Later ranks greedily
    maximize the minimum distance to the already-selected set, with the gain
    of near-duplicates (max similarity to selected > threshold) scaled down
    by ``duplicate_penalty``.
    """
    for position, frame in enumerate(frames):
        frame["_position"] = position
    eligible = [
        frame
        for frame in frames
        if not frame["mask_area_outlier"] and not frame["ambiguous_instance"]
    ]
    if not eligible:
        return []
    n_frames = len(frames)

    def median_distance(frame: dict[str, Any]) -> float:
        if n_frames < 2:
            return 0.0
        distances = [
            float(distance_matrix[frame["_position"], other["_position"]])
            for other in frames
            if other is not frame
        ]
        return float(np.median(distances))

    def sam_score(frame: dict[str, Any]) -> float:
        value = frame.get("sam_score")
        return float(value) if value is not None else 0.0

    rank1 = min(eligible, key=lambda f: (median_distance(f), -sam_score(f)))
    rank1["_selection_note"] = (
        "rank1: most representative eligible frame, min median foreground "
        f"DINO distance to other ok frames = {median_distance(rank1):.4f}; "
        f"sam_score={sam_score(rank1):.4f}"
    )
    selected = [rank1]
    remaining = [frame for frame in eligible if frame is not rank1]
    while remaining and len(selected) < max_references:
        best_frame = None
        best_key = None
        for frame in remaining:
            distances = [
                float(distance_matrix[frame["_position"], chosen["_position"]])
                for chosen in selected
            ]
            max_similarity = 1.0 - min(distances)
            gain = min(distances)
            downweighted = max_similarity > near_duplicate_similarity
            if downweighted:
                gain *= duplicate_penalty
            key = (gain, sam_score(frame), -frame["frame_index"])
            if best_key is None or key > best_key:
                best_key = key
                best_frame = (frame, min(distances), max_similarity, downweighted)
        frame, min_distance, max_similarity, downweighted = best_frame
        note = (
            f"greedy marginal info: min foreground DINO distance to selected "
            f"= {min_distance:.4f}, max similarity to selected = "
            f"{max_similarity:.4f}"
        )
        if downweighted:
            note += (
                f"; near-duplicate of an already-selected frame (similarity > "
                f"{near_duplicate_similarity}), gain downweighted x{duplicate_penalty}"
            )
        frame["_selection_note"] = note
        selected.append(frame)
        remaining = [item for item in remaining if item is not frame]
    return selected


def build_reference_sets(
    grounding_dir: Path,
    candidates_path: Path | None,
    output_path: Path,
    embedder: Dinov3ForegroundEmbedder,
    *,
    full_frame_ratio: float = 0.8,
    near_duplicate_similarity: float = 0.95,
    max_references: int = 5,
    duplicate_penalty: float = 0.1,
    overwrite: bool = False,
) -> dict[str, Any]:
    grounding_dir = grounding_dir.resolve()
    observations_path = grounding_dir / "object_observations.jsonl"
    if not observations_path.exists():
        raise FileNotFoundError(observations_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact {output_path}; "
            "pass --overwrite to replace it"
        )

    observations = common.read_jsonl(observations_path)
    queue_by_candidate: dict[str, dict[str, Any]] = {}
    if candidates_path is not None and candidates_path.exists():
        for row in common.read_jsonl(candidates_path):
            queue_by_candidate[str(row.get("candidate_id", "UNKNOWN"))] = row

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        groups.setdefault(str(row.get("candidate_id", "UNKNOWN")), []).append(row)

    candidate_ids = list(queue_by_candidate)
    for candidate_id in groups:
        if candidate_id not in queue_by_candidate:
            candidate_ids.append(candidate_id)

    result_candidates: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        queue_row = queue_by_candidate.get(candidate_id, {})
        rows = sorted(
            groups.get(candidate_id, []),
            key=lambda row: int(row.get("frame_index", 0)),
        )
        ok_rows = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("assets", {}).get("rgba_path")
            and Path(str(row["assets"]["rgba_path"])).exists()
        ]
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "shot_id": queue_row.get("shot_id") or (ok_rows[0].get("shot_id") if ok_rows else None),
            "canonical_name": queue_row.get("canonical_name"),
            "visual_attributes": queue_row.get("visual_attributes"),
            "possible_states": queue_row.get("possible_states"),
            "observed_state": queue_row.get("observed_state"),
            "visible_time_range_sec": queue_row.get("visible_time_range_sec"),
        }
        if not ok_rows:
            entry.update({
                "references": [],
                "pairwise": {"frame_indices": [], "cosine_distance": []},
                "group_stats": {
                    "ok_frame_count": 0,
                    "observation_row_count": len(rows),
                    "pair_count": 0,
                    "near_duplicate_pair_count": 0,
                    "near_duplicate_pair_ratio": None,
                },
                "frame_evidence": [],
                "completeness": {
                    "status": "empty",
                    "selected_count": 0,
                    "eligible_frame_count": 0,
                    "ok_frame_count": 0,
                    "excluded_frames": [],
                    "notes": "no status=ok observation with an existing rgba "
                             "asset; nothing to select",
                },
            })
            result_candidates.append(entry)
            continue

        rgba_paths = [Path(str(row["assets"]["rgba_path"])) for row in ok_rows]
        features = embedder.embed_rgba_paths(rgba_paths)
        distance_matrix = _pairwise_cosine_distance(features)
        similarity_matrix = np.clip(1.0 - distance_matrix, -1.0, 1.0)
        outlier_reasons = _flag_mask_area_outliers(
            [float(row.get("mask_area_ratio", 0.0)) for row in ok_rows],
            full_frame_ratio,
        )

        frames: list[dict[str, Any]] = []
        for position, (row, reasons) in enumerate(zip(ok_rows, outlier_reasons)):
            others = [
                float(distance_matrix[position, other])
                for other in range(len(ok_rows))
                if other != position
            ]
            assets = row.get("assets", {})
            frames.append({
                "frame_index": int(row.get("frame_index", 0)),
                "timestamp_sec": row.get("timestamp_sec"),
                "status": row.get("status"),
                "mask_area_ratio": row.get("mask_area_ratio"),
                "mask_area_pixels": row.get("mask_area_pixels"),
                "sam_score": row.get("sam_score"),
                "selected_grounding_score": row.get("selected_grounding_score"),
                "selection_strategy": row.get("selection_strategy"),
                "ambiguous_instance": bool(row.get("ambiguous_instance")),
                "mask_area_outlier": bool(reasons),
                "mask_area_outlier_reasons": reasons,
                "foreground_dino_distances": {
                    str(int(ok_rows[other].get("frame_index", 0))): round(
                        float(distance_matrix[position, other]), 6
                    )
                    for other in range(len(ok_rows))
                    if other != position
                },
                "foreground_dino_mean_distance": (
                    round(float(np.mean(others)), 6) if others else None
                ),
                "foreground_dino_median_distance": (
                    round(float(np.median(others)), 6) if others else None
                ),
                "foreground_dino_max_similarity_to_others": (
                    round(float(1.0 - min(others)), 6) if others else None
                ),
                "assets": {
                    name: _relative_asset(assets.get(f"{name}_path"), grounding_dir)
                    for name in ("rgba", "crop", "frame", "mask", "overlay")
                },
            })

        selected = _select_references(
            frames,
            distance_matrix,
            near_duplicate_similarity,
            max_references,
            duplicate_penalty,
        )
        selected_positions = {frame["_position"] for frame in selected}
        references: list[dict[str, Any]] = []
        for rank, frame in enumerate(selected, 1):
            references.append({
                "rank": rank,
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "asset": frame["assets"],
                "evidence": {
                    key: frame[key]
                    for key in (
                        "mask_area_ratio",
                        "mask_area_outlier",
                        "ambiguous_instance",
                        "sam_score",
                        "selected_grounding_score",
                        "foreground_dino_median_distance",
                        "foreground_dino_max_similarity_to_others",
                    )
                },
                "selection_note": frame["_selection_note"],
            })

        frame_evidence: list[dict[str, Any]] = []
        excluded_frames: list[dict[str, Any]] = []
        for frame in frames:
            exclusion: list[str] = []
            if frame["mask_area_outlier"]:
                exclusion.extend(frame["mask_area_outlier_reasons"])
            if frame["ambiguous_instance"]:
                exclusion.append(
                    "ambiguous_instance=true in the source observation; "
                    "ambiguous frames are never auto-selected"
                )
            if frame["_position"] in selected_positions:
                exclusion = []
            elif not exclusion:
                exclusion.append("eligible but beyond the reference-count budget")
            record = {
                key: frame[key]
                for key in (
                    "frame_index",
                    "timestamp_sec",
                    "mask_area_ratio",
                    "mask_area_outlier",
                    "mask_area_outlier_reasons",
                    "ambiguous_instance",
                    "sam_score",
                    "foreground_dino_mean_distance",
                    "foreground_dino_median_distance",
                    "foreground_dino_max_similarity_to_others",
                    "foreground_dino_distances",
                    "assets",
                )
            }
            record["selected_rank"] = (
                references and next(
                    (
                        reference["rank"]
                        for reference in references
                        if reference["frame_index"] == frame["frame_index"]
                    ),
                    None,
                )
            )
            record["exclusion_reasons"] = exclusion
            frame_evidence.append(record)
            if frame["_position"] not in selected_positions:
                excluded_frames.append({
                    "frame_index": frame["frame_index"],
                    "reasons": exclusion,
                })

        upper = [
            float(similarity_matrix[left, right])
            for left in range(len(ok_rows))
            for right in range(left + 1, len(ok_rows))
        ]
        near_duplicate_count = sum(
            value > near_duplicate_similarity for value in upper
        )
        eligible_count = sum(
            not frame["mask_area_outlier"] and not frame["ambiguous_instance"]
            for frame in frames
        )
        if not references:
            status = "empty"
        elif len(references) < 3:
            status = "reduced"
        else:
            status = "full"
        notes: list[str] = []
        if status == "reduced":
            notes.append(
                f"only {len(references)} eligible frame(s) after excluding "
                "mask-area outliers and ambiguous-instance frames; reference "
                "count reduced below the 3-reference floor"
            )
        if status == "empty":
            notes.append(
                "every ok frame is a mask-area outlier or ambiguous; no "
                "reference auto-selected"
            )
        if near_duplicate_count:
            notes.append(
                f"{near_duplicate_count}/{len(upper)} foreground pairs are "
                f"near-duplicates (cosine similarity > "
                f"{near_duplicate_similarity}); near-duplicate frames were "
                "downweighted during greedy selection"
            )

        entry.update({
            "references": references,
            "pairwise": {
                "frame_indices": [frame["frame_index"] for frame in frames],
                "cosine_distance": [
                    [round(float(value), 6) for value in row_values]
                    for row_values in distance_matrix.tolist()
                ],
            },
            "group_stats": {
                "ok_frame_count": len(ok_rows),
                "observation_row_count": len(rows),
                "pair_count": len(upper),
                "near_duplicate_pair_count": near_duplicate_count,
                "near_duplicate_pair_ratio": (
                    round(near_duplicate_count / len(upper), 6) if upper else None
                ),
                "max_pair_similarity": (
                    round(max(upper), 6) if upper else None
                ),
                "min_pair_similarity": (
                    round(min(upper), 6) if upper else None
                ),
            },
            "frame_evidence": frame_evidence,
            "completeness": {
                "status": status,
                "selected_count": len(references),
                "eligible_frame_count": eligible_count,
                "ok_frame_count": len(ok_rows),
                "excluded_frames": excluded_frames,
                "notes": " ".join(notes) if notes else "all ok frames eligible",
            },
        })
        for frame in frames:
            frame.pop("_position", None)
            frame.pop("_selection_note", None)
        result_candidates.append(entry)

    result: dict[str, Any] = {
        "schema_version": "object-reference-sets-v1",
        "grounding_dir": str(grounding_dir),
        "candidates_path": (
            str(candidates_path.resolve())
            if candidates_path is not None and candidates_path.exists()
            else None
        ),
        "foreground_embedder": {
            "model": "dinov3-vitl16",
            "model_path": str(embedder.model_path),
            "input": "rgba foreground composited on a fixed gray background "
                     f"(value {GRAY_BACKGROUND_VALUE}), in memory only",
            "pooling": "patch tokens (CLS + 4 register tokens skipped) "
                       "mean-pooled, L2-normalized",
        },
        "rules": {
            "mask_area_outlier": (
                f"frame flagged when mask_area_ratio > {full_frame_ratio} "
                "(full-frame dominance), or when it falls outside the "
                "group's Tukey IQR fences [Q1 - 1.5*IQR, Q3 + 1.5*IQR] over "
                "the candidate's ok frames"
            ),
            "near_duplicate": (
                f"pairwise foreground cosine similarity > "
                f"{near_duplicate_similarity}; near-duplicate frames get "
                f"their greedy marginal gain scaled by {duplicate_penalty}"
            ),
            "eligibility": (
                "only status=ok frames that are neither mask_area_outlier "
                "nor ambiguous_instance are auto-selected; excluded frames "
                "stay listed under frame_evidence with reasons"
            ),
            "ranking": (
                "rank1 = eligible frame with the min median foreground DINO "
                "distance to the other ok frames (SAM score breaks ties); "
                "later ranks greedily maximize the min distance to the "
                "already-selected set; target 3-5 references, fewer when "
                "eligible frames are scarce (see completeness)"
            ),
        },
        "candidate_count": len(result_candidates),
        "candidates": result_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_reference_set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc reference-set builder: reads an existing grounding "
            "output directory and writes reference_sets.json without "
            "re-running detection or segmentation."
        )
    )
    parser.add_argument(
        "--grounding-dir",
        type=Path,
        required=True,
        help="Grounding output dir holding object_observations.jsonl and assets.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="grounding_queue.jsonl joined in by candidate_id; defaults to "
        "<grounding-dir>/../grounding_queue.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to <grounding-dir>/reference_sets.json.",
    )
    parser.add_argument(
        "--dinov3-model-dir", type=Path, default=DEFAULT_DINOV3_DIR
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-frame-ratio", type=float, default=0.8)
    parser.add_argument("--near-duplicate-similarity", type=float, default=0.95)
    parser.add_argument("--max-references", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing reference_sets.json.",
    )
    return parser


def reference_sets_main(argv: Sequence[str]) -> int:
    args = build_reference_set_parser().parse_args(argv)
    candidates_path = args.candidates
    if candidates_path is None:
        candidates_path = args.grounding_dir.parent / "grounding_queue.jsonl"
    output_path = args.output
    if output_path is None:
        output_path = args.grounding_dir / "reference_sets.json"
    if args.max_references < 1:
        raise ValueError("--max-references must be positive")
    embedder = Dinov3ForegroundEmbedder(
        args.dinov3_model_dir, args.device, args.batch_size
    )
    result = build_reference_sets(
        args.grounding_dir,
        candidates_path,
        output_path,
        embedder,
        full_frame_ratio=args.full_frame_ratio,
        near_duplicate_similarity=args.near_duplicate_similarity,
        max_references=args.max_references,
        overwrite=args.overwrite,
    )
    summary = {
        "output_path": str(output_path.resolve()),
        "candidate_count": result["candidate_count"],
        "reference_count_distribution": {
            str(count): sum(
                len(candidate["references"]) == count
                for candidate in result["candidates"]
            )
            for count in sorted({
                len(candidate["references"]) for candidate in result["candidates"]
            })
        },
        "completeness_status_counts": {
            status: sum(
                candidate["completeness"]["status"] == status
                for candidate in result["candidates"]
            )
            for status in ("full", "reduced", "empty")
        },
        "candidates_with_near_duplicates": sum(
            bool(candidate["group_stats"]["near_duplicate_pair_count"])
            for candidate in result["candidates"]
        ),
        "outlier_frame_count": sum(
            sum(
                frame["mask_area_outlier"]
                for frame in candidate["frame_evidence"]
            )
            for candidate in result["candidates"]
        ),
        "ambiguous_frame_count": sum(
            sum(
                frame["ambiguous_instance"]
                for frame in candidate["frame_evidence"]
            )
            for candidate in result["candidates"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = v2.build_parser()
    parser.add_argument(
        "--window-resample-max",
        type=int,
        default=4,
        help="Max extra frames re-sampled inside the same visible window when "
        "the first pass fails or stays ambiguous; 0 disables resampling and "
        "multi-frame selection, restoring exact V3 behavior and schema.",
    )
    parser.add_argument(
        "--window-resample-scan",
        type=int,
        default=64,
        help="Max unsampled in-window frames scanned for sharpness per resample.",
    )
    parser.add_argument(
        "--track-link-iou",
        type=float,
        default=0.3,
        help="IoU threshold linking per-frame clusters into instance tracks.",
    )
    parser.add_argument(
        "--min-track-support-frames",
        type=int,
        default=2,
        help="Cross-frame support required before an instance track emits assets.",
    )
    parser.add_argument(
        "--track-ambiguity-tolerance",
        type=float,
        default=0.05,
        help="Mean-score tolerance for treating equal-support tracks as ambiguous.",
    )
    return parser


def main() -> int:
    if "--build-reference-sets" in sys.argv:
        return reference_sets_main(
            [arg for arg in sys.argv[1:] if arg != "--build-reference-sets"]
        )
    args = build_parser().parse_args()
    v2.validate_inputs(args)
    if args.window_resample_max < 0:
        raise ValueError("--window-resample-max must be >= 0")
    if args.window_resample_scan < 1:
        raise ValueError("--window-resample-scan must be positive")
    if args.min_track_support_frames < 1:
        raise ValueError("--min-track-support-frames must be positive")
    if not 0.0 <= args.track_link_iou <= 1.0:
        raise ValueError("--track-link-iou must be within [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = common.read_jsonl(args.candidates)
    shots = {
        common.normalized_shot_id(row.get("shot_id")): row
        for row in common.read_jsonl(args.shots)
    }
    dino = common.GroundingDinoBackend(
        args.dino_model_dir, args.box_threshold, args.text_threshold
    )
    sam = v2.Sam31Backend(
        args.sam31_source, args.sam31_checkpoint, args.sam31_bpe
    )
    observations: list[dict[str, Any]] = []
    run_started = time.time()
    resample_enabled = args.window_resample_max > 0
    row_schema = "object-observation-v4" if resample_enabled else "object-observation-v3"
    resample_stats = {
        "triggered_candidate_shots": 0,
        "added_frame_total": 0,
        "window_exhausted_candidate_shots": 0,
    }

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "UNKNOWN"))
        phrases = common.unique_phrases(candidate)
        expected = candidate.get("expected_shots") or [candidate.get("shot_id")]
        for raw_shot_id in expected:
            shot_id = common.normalized_shot_id(raw_shot_id)
            shot = shots.get(shot_id)
            if shot is None:
                observations.append({
                    "schema_version": row_schema,
                    "candidate_id": candidate_id,
                    "shot_id": shot_id,
                    "status": "shot_missing",
                })
                continue
            video_path = Path(str(shot.get("clip_path") or shot.get("video_path")))
            requested_range = candidate.get("visible_time_range_sec")
            frames, sampling = sample_video_frames_in_window(
                video_path, args.samples_per_shot, requested_range
            )
            records: list[dict[str, Any]] = []
            for frame in frames:
                raw_detections, clusters = _detect_clusters(
                    dino, phrases, frame, args.cluster_iou
                )
                records.append({
                    "frame": frame,
                    "sampling_pass": "initial",
                    "raw_detections": raw_detections,
                    "clusters": clusters,
                })

            if not resample_enabled:
                for record in records:
                    frame = record["frame"]
                    selected, strategy, ambiguous = common.select_cluster(
                        record["clusters"]
                    )
                    base = _base_observation(
                        "object-observation-v3", candidate, candidate_id, shot_id,
                        video_path, requested_range, sampling, frame, phrases,
                        record["raw_detections"], record["clusters"], strategy,
                        ambiguous, args,
                    )
                    if selected is None:
                        _no_selection(base, "no_detection")
                        observations.append(base)
                        continue
                    _segment_selection(base, sam, args, frame, selected)
                    observations.append(base)
                continue

            track, strategy, ambiguous = _select_track(records, args)
            resample_record: dict[str, Any] = {
                "enabled": True,
                "budget_max_frames": args.window_resample_max,
                "triggered": False,
                "trigger_reason": None,
                "rounds": 0,
                "added_frame_indices": [],
                "window_limited": True,
                "window_exhausted": False,
            }
            if track is None or ambiguous:
                resample_record["triggered"] = True
                resample_record["trigger_reason"] = (
                    "no_detection" if track is None else "ambiguous"
                )
                sampled = [record["frame"]["frame_index"] for record in records]
                extra_frames, _ = sample_video_frames_in_window(
                    video_path,
                    args.window_resample_max,
                    requested_range,
                    exclude_indices=sampled,
                    prefer_sharp=True,
                    scan_limit=args.window_resample_scan,
                )
                if extra_frames:
                    resample_record["rounds"] = 1
                    resample_record["added_frame_indices"] = [
                        int(frame["frame_index"]) for frame in extra_frames
                    ]
                    for frame in extra_frames:
                        raw_detections, clusters = _detect_clusters(
                            dino, phrases, frame, args.cluster_iou
                        )
                        records.append({
                            "frame": frame,
                            "sampling_pass": "resample",
                            "raw_detections": raw_detections,
                            "clusters": clusters,
                        })
                    records.sort(key=lambda record: record["frame"]["frame_index"])
                    track, strategy, ambiguous = _select_track(records, args)
                else:
                    resample_record["window_exhausted"] = True
                resample_stats["triggered_candidate_shots"] += 1
                resample_stats["added_frame_total"] += len(
                    resample_record["added_frame_indices"]
                )
                if resample_record["window_exhausted"]:
                    resample_stats["window_exhausted_candidate_shots"] += 1

            member_by_frame = (
                track.get("member_by_frame", {})
                if track is not None and not ambiguous
                else {}
            )
            support_record = None
            if track is not None:
                support_record = {
                    "track_id": track["track_id"],
                    "support_frame_indices": [
                        int(index) for index in track["frame_indices"]
                    ],
                    "support_frame_count": int(track["support_frame_count"]),
                    "sampled_frame_count": len(records),
                    "mean_max_score": track["mean_max_score"],
                    "phrase_vote_total": track["phrase_vote_total"],
                    "min_support_frames": min(
                        args.min_track_support_frames, max(1, len(records))
                    ),
                    "consistency_ok": not ambiguous,
                }
            for record in records:
                frame = record["frame"]
                clusters = record["clusters"]
                selected = member_by_frame.get(int(frame["frame_index"]))
                base = _base_observation(
                    "object-observation-v4", candidate, candidate_id, shot_id,
                    video_path, requested_range, sampling, frame, phrases,
                    record["raw_detections"], clusters, strategy,
                    bool(ambiguous and clusters), args,
                )
                base["sampling_pass"] = record["sampling_pass"]
                base["frame_sharpness"] = frame.get("sharpness")
                base["resample"] = resample_record
                base["cross_frame_support"] = support_record
                if ambiguous:
                    _no_selection(
                        base, "ambiguous_instance" if clusters else "no_detection"
                    )
                elif track is None:
                    _no_selection(base, "no_detection")
                elif selected is None:
                    _no_selection(
                        base, "no_detection" if not clusters else "instance_track_absent"
                    )
                else:
                    _segment_selection(base, sam, args, frame, selected)
                observations.append(base)

    common.write_jsonl(args.output / "object_observations.jsonl", observations)
    successful = [row for row in observations if row.get("status") == "ok"]
    summary = {
        "schema_version": (
            "object-grounding-segmentation-run-v4"
            if resample_enabled
            else "object-grounding-segmentation-run-v3"
        ),
        "candidate_count": len(candidates),
        "observation_count": len(observations),
        "successful_observation_count": len(successful),
        "ambiguous_observation_count": sum(
            bool(row.get("ambiguous_instance")) for row in observations
        ),
        "no_detection_count": sum(
            row.get("status") == "no_detection" for row in observations
        ),
        "no_mask_count": sum(row.get("status") == "no_mask" for row in observations),
        "visibility_range_applied_count": sum(
            bool(row.get("visibility_range_applied")) for row in observations
        ),
        "grounding_backend": "GroundingDINO_SwinT_OGC",
        "segmentation_backend": "sam3.1_image_geometric_prompt",
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "cluster_iou": args.cluster_iou,
        "elapsed_seconds": round(time.time() - run_started, 3),
    }
    if resample_enabled:
        summary["window_resample"] = {
            "enabled": True,
            "budget_max_frames": args.window_resample_max,
            **resample_stats,
        }
        summary["instance_track_absent_count"] = sum(
            row.get("status") == "instance_track_absent" for row in observations
        )
        summary["min_track_support_frames"] = args.min_track_support_frames
        summary["track_link_iou"] = args.track_link_iou
        summary["track_ambiguity_tolerance"] = args.track_ambiguity_tolerance
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
