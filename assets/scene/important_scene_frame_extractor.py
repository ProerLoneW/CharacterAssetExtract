#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover all important scene views in one medium/long video.

This is a deliberately standalone implementation of the key-frame ranking
part described in ``关键场景帧提取与场景设定图生成_开发文档_V1.md``.  It uses
only local plug-in models: DINOv3 groups visually related scene views and
YOLOv8n/OpenCV rank how usable each frame is.  There are no prompts, remote
APIs, LLMs, or VLMs in this path, and this script does not perform shot-cut
detection or require shot/segment/global annotations.

For every input video the script writes::

    <output>/<video_stem>/
      all_scenes_preview.jpg
      candidate_scores.json
      scene_index.json
      scene_001/
        anchor_main.jpg
        anchor_aux_01.jpg ...
        person_mask_main.png     # conservative bbox-union proxy mask
        anchor_main_debug.jpg
        topk_preview.jpg
        scene_meta.json
      scene_002/ ...

The detector is run once on each full frame for people and structural COCO
objects.  Optional overlapping tiles add small-person recall.  Ranking is a
weighted fusion of person occupancy, structural occlusion, sharpness,
exposure, composition, visual representativeness, temporal stability and a
light parallax proxy.  Each aux frame must share the main frame's DINO cluster
and pass a direct similarity threshold, so unrelated frames never fill quota.

Example::

    python prompts/important_scene_frame_extractor.py \
      /path/to/medium_video.mp4 --output-dir /path/to/output \
      --max-scenes 12 --frames-per-scene 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
# 迁移后位于 extract_full_pipeline/assets/scene/；v2 仓库根在上三级。
PROJECT_ROOT = SCRIPT_PATH.parents[3]
WORKSPACE_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "local_models"
    / "asset_detectors"
    / "yolov8n"
    / "yolov8n.pt"
)
DEFAULT_DINO_MODEL = (
    PROJECT_ROOT
    / "local_models"
    / "dinov3-vitl16"
)

# COCO objects that expose useful indoor/outdoor scene structure.  They are
# intentionally conservative: movable props such as cups and phones do not
# make a frame a better scene reference.
STRUCTURAL_CLASSES = {
    "bench",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "potted plant",
    "book",
    "clock",
    "vase",
}

DEFAULT_WEIGHTS = {
    "person_clear": 0.22,
    "structure_clear": 0.22,
    "sharpness": 0.16,
    "exposure": 0.10,
    "composition": 0.10,
    "representativeness": 0.15,
    "stability": 0.03,
    "parallax": 0.02,
}

# Optional shot-manifest alignment snaps sampled timestamps onto the union of
# every t0/t1 boundary found in shots.jsonl.
SHOT_SNAP_TOLERANCE_SEC = 0.6
# --max-scenes auto searches this inclusive k grid by mean cosine silhouette.
AUTO_SCENE_GRID = range(2, 13)
# Silhouette is evaluated on at most this many embeddings for speed.
SILHOUETTE_SAMPLE_LIMIT = 2000


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: list[int]
    source: str


@dataclass
class Candidate:
    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray = field(repr=False)
    scene_cluster_id: int | None = None
    detections: list[Detection] = field(default_factory=list)
    person_boxes: list[list[int]] = field(default_factory=list)
    structural_boxes: list[list[int]] = field(default_factory=list)
    structural_objects: list[dict[str, Any]] = field(default_factory=list)
    person_area_ratio: float = 0.0
    structure_area_ratio: float = 0.0
    structure_occlusion_ratio: float = 0.0
    sharpness_raw: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    composition_score: float = 0.0
    representativeness_score: float = 0.0
    stability_score: float = 0.0
    parallax_potential_score: float = 0.0
    quality_gate: float = 0.0
    person_gate: float = 0.0
    total_score: float = 0.0
    selection_score: float | None = None
    selected_rank: int | None = None
    dino_similarity_to_main: float | None = None
    same_scene_as_main: bool = False
    max_sim_to_selected: float | None = None
    selection_note: str | None = None
    descriptor: np.ndarray | None = field(default=None, repr=False)
    dino_embedding: np.ndarray | None = field(default=None, repr=False)
    person_mask: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        # Do not call ``asdict(self)`` here: it would deepcopy every full-HD
        # image before we discard it, multiplying memory use on long batches.
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "scene_cluster_id": self.scene_cluster_id,
            "detections": [asdict(item) for item in self.detections],
            "person_boxes": self.person_boxes,
            "structural_boxes": self.structural_boxes,
            "structural_objects": self.structural_objects,
            "person_area_ratio": self.person_area_ratio,
            "structure_area_ratio": self.structure_area_ratio,
            "structure_occlusion_ratio": self.structure_occlusion_ratio,
            "sharpness_raw": self.sharpness_raw,
            "sharpness_score": self.sharpness_score,
            "exposure_score": self.exposure_score,
            "composition_score": self.composition_score,
            "representativeness_score": self.representativeness_score,
            "stability_score": self.stability_score,
            "parallax_potential_score": self.parallax_potential_score,
            "quality_gate": self.quality_gate,
            "person_gate": self.person_gate,
            "total_score": self.total_score,
            "selection_score": self.selection_score,
            "selected_rank": self.selected_rank,
            "dino_similarity_to_main": self.dino_similarity_to_main,
            "same_scene_as_main": self.same_scene_as_main,
        }


@dataclass
class SceneCluster:
    cluster_id: int
    members: list[Candidate]
    centroid: np.ndarray = field(repr=False)
    importance_score: float = 0.0
    selected: list[Candidate] = field(default_factory=list)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0


def _robust_normalize(values: Sequence[float], neutral: float = 0.5) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    low, high = np.percentile(array, [10.0, 90.0])
    if high - low < 1e-9:
        return [neutral] * len(array)
    return [float(np.clip((item - low) / (high - low), 0.0, 1.0)) for item in array]


def _box_area(box: Sequence[int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _nms(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            old.label == detection.label
            and _box_iou(old.bbox_xyxy, detection.bbox_xyxy) >= iou_threshold
            for old in kept
        ):
            continue
        kept.append(detection)
    return kept


def _mask_from_boxes(shape: tuple[int, int], boxes: Sequence[Sequence[int]]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 255
    return mask


def _scene_descriptor(image_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Compact background-biased colour/texture descriptor, no learned model."""
    small = cv2.resize(image_bgr, (256, 144), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(person_mask, (256, 144), interpolation=cv2.INTER_NEAREST)
    valid = (mask == 0).astype(np.uint8) * 255
    # If people cover nearly the whole image, keep enough pixels for a stable
    # descriptor rather than returning an empty histogram.
    if cv2.countNonZero(valid) < valid.size * 0.08:
        valid[:] = 255
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], valid, [16, 8], [0, 180, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    hist /= float(hist.sum() + 1e-8)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    means: list[float] = []
    stds: list[float] = []
    for row in range(3):
        for col in range(4):
            y0, y1 = row * 48, (row + 1) * 48
            x0, x1 = col * 64, (col + 1) * 64
            cell = gray[y0:y1, x0:x1]
            cell_valid = valid[y0:y1, x0:x1] > 0
            pixels = cell[cell_valid] if np.any(cell_valid) else cell.reshape(-1)
            means.append(float(np.mean(pixels)) / 255.0)
            stds.append(float(np.std(pixels)) / 128.0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.floor(angle / 22.5).astype(np.int32) % 8
    orientation = np.asarray(
        [float(magnitude[(bins == index) & (valid > 0)].sum()) for index in range(8)],
        dtype=np.float32,
    )
    orientation /= float(orientation.sum() + 1e-8)
    descriptor = np.concatenate(
        [hist, np.asarray(means, np.float32), np.asarray(stds, np.float32), orientation]
    )
    descriptor /= float(np.linalg.norm(descriptor) + 1e-8)
    return descriptor


def _exposure_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(thumb))
    p05, p95 = np.percentile(thumb, [5.0, 95.0])
    clipped = float(np.mean(thumb <= 5) + np.mean(thumb >= 250))
    midpoint = math.exp(-((mean - 128.0) / 78.0) ** 2)
    tonal_range = _clip01(float(p95 - p05) / 150.0)
    unclipped = _clip01(1.0 - clipped * 2.5)
    return _clip01(0.45 * midpoint + 0.30 * tonal_range + 0.25 * unclipped)


def _composition_score(
    image_bgr: np.ndarray, structural_boxes: Sequence[Sequence[int]]
) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (360, 202), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(thumb, 60, 160) > 0
    masses: list[float] = []
    for row in range(3):
        for col in range(3):
            cell = edges[row * 67 : (row + 1) * 67, col * 120 : (col + 1) * 120]
            masses.append(float(cell.sum()) + 1.0)
    probs = np.asarray(masses) / sum(masses)
    spread = float(-(probs * np.log(probs)).sum() / math.log(9.0))

    height, width = image_bgr.shape[:2]
    if structural_boxes:
        complete = []
        for x0, y0, x1, y1 in structural_boxes:
            margin_x, margin_y = 0.015 * width, 0.015 * height
            touches_edge = x0 < margin_x or y0 < margin_y or x1 > width - margin_x or y1 > height - margin_y
            complete.append(0.35 if touches_edge else 1.0)
        structure_completeness = float(np.mean(complete))
    else:
        structure_completeness = 0.5
    return _clip01(0.72 * spread + 0.28 * structure_completeness)


class YoloSceneDetector:
    """Local YOLO detector with optional overlapping small-person tiles."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        confidence: float,
        structure_confidence: float,
        image_size: int,
        tiled_people: bool,
        tile_overlap: float,
        batch_size: int,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO weights do not exist: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required: pip install ultralytics") from exc
        self.model_path = model_path.resolve()
        self.model = YOLO(str(model_path))
        self.names: dict[int, str] = dict(self.model.names)
        self.device = device
        self.confidence = confidence
        self.structure_confidence = structure_confidence
        self.image_size = image_size
        self.tiled_people = tiled_people
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.person_class = next(
            index for index, name in self.names.items() if name == "person"
        )
        self.structure_classes = {
            index for index, name in self.names.items() if name in STRUCTURAL_CLASSES
        }
        self.full_classes = sorted({self.person_class} | self.structure_classes)

    def _predict(self, images: Sequence[np.ndarray], classes: list[int], size: int):
        all_results = []
        for chunk in _chunks(images, self.batch_size):
            results = self.model.predict(
                list(chunk),
                imgsz=size,
                conf=self.confidence,
                iou=0.55,
                classes=classes,
                device=self.device,
                verbose=False,
            )
            all_results.extend(results)
        return all_results

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        detected: list[list[Detection]] = [[] for _ in frames]
        full_results = self._predict(frames, self.full_classes, self.image_size)
        for frame_index, result in enumerate(full_results):
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            height, width = frames[frame_index].shape[:2]
            for box, class_id, confidence in zip(boxes, classes, confidences):
                label = self.names[int(class_id)]
                if label != "person" and float(confidence) < self.structure_confidence:
                    continue
                detected[frame_index].append(
                    Detection(
                        label=label,
                        confidence=round(float(confidence), 6),
                        bbox_xyxy=self._clip_box(box, width, height),
                        source="full_frame",
                    )
                )

        if self.tiled_people:
            tile_images: list[np.ndarray] = []
            tile_meta: list[tuple[int, int, int]] = []
            for frame_index, frame in enumerate(frames):
                for x0, y0, x1, y1 in self._tiles(frame.shape[1], frame.shape[0]):
                    tile_images.append(frame[y0:y1, x0:x1])
                    tile_meta.append((frame_index, x0, y0))
            tile_results = self._predict(
                tile_images, [self.person_class], max(640, self.image_size // 2)
            )
            for (frame_index, offset_x, offset_y), result in zip(tile_meta, tile_results):
                if result.boxes is None:
                    continue
                height, width = frames[frame_index].shape[:2]
                for box, confidence in zip(
                    result.boxes.xyxy.detach().cpu().numpy(),
                    result.boxes.conf.detach().cpu().numpy(),
                ):
                    shifted = np.asarray(box) + np.asarray(
                        [offset_x, offset_y, offset_x, offset_y]
                    )
                    clipped = self._clip_box(shifted, width, height)
                    # Tiny boxes below roughly 8x12 pixels at 1080p are mostly
                    # detector noise and have negligible scene occupancy.
                    if _box_area(clipped) < max(64, width * height * 0.00004):
                        continue
                    detected[frame_index].append(
                        Detection(
                            label="person",
                            confidence=round(float(confidence), 6),
                            bbox_xyxy=clipped,
                            source="overlap_tile",
                        )
                    )

        return [_nms(items) for items in detected]

    @staticmethod
    def _clip_box(box: Sequence[float], width: int, height: int) -> list[int]:
        x0, y0, x1, y1 = [float(value) for value in box]
        return [
            max(0, min(width - 1, int(math.floor(x0)))),
            max(0, min(height - 1, int(math.floor(y0)))),
            max(1, min(width, int(math.ceil(x1)))),
            max(1, min(height, int(math.ceil(y1)))),
        ]

    def _tiles(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        # A 2x2 layout with overlap.  Full-frame inference is always retained,
        # so the tiles only need to improve small-person recall.
        tile_width = min(width, int(math.ceil(width * (0.5 + self.tile_overlap / 2))))
        tile_height = min(height, int(math.ceil(height * (0.5 + self.tile_overlap / 2))))
        x_values = sorted({0, width - tile_width})
        y_values = sorted({0, height - tile_height})
        return [
            (x0, y0, x0 + tile_width, y0 + tile_height)
            for y0 in y_values
            for x0 in x_values
        ]


class Dinov3SceneEmbedder:
    """Offline DINOv3 scene embeddings from local Hugging Face weights."""

    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"DINOv3 model directory does not exist: {model_path}")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for local DINOv3") from exc
        if device.isdigit():
            device = f"cuda:{device}"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"DINO device {device!r} requested but CUDA is unavailable")
        self.model_path = model_path.resolve()
        self.device = device
        self.batch_size = max(1, batch_size)
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True
        ).eval().to(self.device)

    @staticmethod
    def _background_image(candidate: Candidate) -> np.ndarray:
        """Reduce actor identity influence while retaining surrounding layout."""
        rgb = cv2.cvtColor(candidate.image_bgr, cv2.COLOR_BGR2RGB)
        if candidate.person_mask is None or not np.any(candidate.person_mask):
            return rgb
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=22.0, sigmaY=22.0)
        # Feathering avoids creating an artificial hard rectangle at bbox edges.
        mask = cv2.GaussianBlur(candidate.person_mask, (0, 0), sigmaX=9.0) / 255.0
        mixed = rgb.astype(np.float32) * (1.0 - mask[..., None])
        mixed += blurred.astype(np.float32) * mask[..., None]
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def embed(self, candidates: Sequence[Candidate]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        torch = self.torch
        for chunk in _chunks(candidates, self.batch_size):
            images = [self._background_image(item) for item in chunk]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                tokens = self.model(**inputs).last_hidden_state
                # DINOv3 ViT has one CLS and four register tokens.  Mean-pool
                # spatial patch tokens to represent the full environment.
                features = tokens[:, 5:, :].mean(dim=1)
                features = torch.nn.functional.normalize(features, dim=1)
            embeddings.append(features.float().cpu().numpy())
        matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
        for candidate, embedding in zip(candidates, matrix):
            candidate.dino_embedding = embedding
        return matrix


def _load_shot_boundaries(path: Path) -> list[float]:
    """Read a shots.jsonl manifest into a sorted unique boundary list.

    Every non-empty line must be a JSON object carrying
    ``time_range_sec: [t0, t1]``; both endpoints of each shot join the
    boundary set that sampled timestamps may snap onto.
    """
    if not path.is_file():
        raise FileNotFoundError(f"shots manifest does not exist: {path}")
    boundaries: set[float] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                start_sec, end_sec = record["time_range_sec"]
                boundaries.add(float(start_sec))
                boundaries.add(float(end_sec))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid shots manifest line {line_number} in {path}: {stripped}"
                ) from exc
    if not boundaries:
        raise ValueError(f"shots manifest contains no time_range_sec entries: {path}")
    return sorted(boundaries)


def _align_samples_to_shots(
    frame_positions: np.ndarray,
    boundaries: Sequence[float],
    tolerance_sec: float,
    fps: float,
    frame_count: int,
    desired: int,
) -> tuple[np.ndarray, int]:
    """Snap sample positions (in frames) onto the closest shot boundary.

    ``boundaries`` must be sorted ascending (as returned by
    ``_load_shot_boundaries``).  A sample moves only when the nearest
    boundary lies within ``tolerance_sec``; otherwise it keeps its original
    timestamp.  Samples collapsing onto one boundary are deduplicated and
    the pool is topped back up towards ``desired`` with the nearest unused
    in-range boundaries, so alignment can never push the candidate count
    below the min/max-candidate target while spare boundaries remain.
    """

    def clamp_index(position: float) -> int:
        return max(0, min(frame_count - 1, int(round(position))))

    boundary_positions = np.asarray([value * fps for value in boundaries], dtype=np.float64)
    requested = [float(value) for value in frame_positions]
    kept: list[float] = []
    used: set[int] = set()
    snapped_count = 0
    for position in requested:
        insert = int(np.searchsorted(boundary_positions, position))
        nearest: float | None = None
        if insert > 0:
            nearest = float(boundary_positions[insert - 1])
        if insert < boundary_positions.size:
            upper = float(boundary_positions[insert])
            if nearest is None or abs(upper - position) < abs(nearest - position):
                nearest = upper
        if nearest is not None and abs(nearest - position) <= tolerance_sec * fps:
            position = nearest
            snapped_count += 1
        index = clamp_index(position)
        if index in used:
            continue
        used.add(index)
        kept.append(position)
    if len(kept) < desired:
        spare = sorted(
            (
                float(value) * fps
                for value in boundaries
                if 0.0 <= value * fps <= frame_count - 1
            ),
            key=lambda item: min(abs(item - old) for old in requested),
        )
        for position in spare:
            index = clamp_index(position)
            if index in used:
                continue
            used.add(index)
            kept.append(position)
            if len(kept) >= desired:
                break
    return np.asarray(sorted(kept), dtype=np.float64), snapped_count


def _sample_video(
    video_path: Path,
    sample_interval: float,
    min_candidates: int,
    max_candidates: int,
    shot_boundaries: Sequence[float] | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"video contains no readable frames: {video_path}")
    duration = max(0.0, (frame_count - 1) / fps)
    margin = min(1.0, max(0.15, duration * 0.02))
    start, end = margin, max(margin, duration - margin)
    desired = max(min_candidates, int(math.ceil(max(0.0, end - start) / sample_interval)) + 1)
    desired = max(1, min(max_candidates, desired, frame_count))
    frame_positions = np.linspace(start * fps, end * fps, desired)
    snapped_count = 0
    if shot_boundaries is not None:
        # Flag-gated alignment to externally provided shot boundaries; the
        # default path feeds the raw linspace samples to the decoder below.
        frame_positions, snapped_count = _align_samples_to_shots(
            frame_positions,
            shot_boundaries,
            SHOT_SNAP_TOLERANCE_SEC,
            fps,
            frame_count,
            desired,
        )
    frame_indices = sorted(
        {
            max(0, min(frame_count - 1, int(round(value))))
            for value in frame_positions
        }
    )
    candidates: list[Candidate] = []
    used_indices: set[int] = set()
    for frame_index in frame_indices:
        image = None
        decoded_index = frame_index
        # Random seeking in long-GOP MP4s can occasionally fail on one exact
        # frame.  Retry immediate neighbours instead of silently reducing the
        # candidate pool.
        for offset in (0, -1, 1, -2, 2):
            retry_index = max(0, min(frame_count - 1, frame_index + offset))
            if retry_index in used_indices:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, retry_index)
            ok, retry_image = capture.read()
            if ok and retry_image is not None:
                image = retry_image
                decoded_index = retry_index
                break
        if image is None:
            continue
        used_indices.add(decoded_index)
        candidates.append(
            Candidate(
                frame_index=decoded_index,
                timestamp_sec=round(decoded_index / fps, 6),
                image_bgr=image,
            )
        )
    capture.release()
    if not candidates:
        raise RuntimeError(f"failed to decode candidate frames: {video_path}")
    video_info: dict[str, Any] = {
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_sec": round(duration, 6),
        "width": width,
        "height": height,
        "candidate_count": len(candidates),
        "sampling_margin_sec": round(margin, 6),
    }
    if shot_boundaries is not None:
        video_info["shot_boundary_count"] = len(shot_boundaries)
        video_info["snap_tolerance_sec"] = SHOT_SNAP_TOLERANCE_SEC
        video_info["snapped_timestamp_count"] = snapped_count
    return candidates, video_info


def _measure_candidates(candidates: list[Candidate], weights: dict[str, float]) -> None:
    sharpness_values: list[float] = []
    for candidate in candidates:
        image = candidate.image_bgr
        height, width = image.shape[:2]
        candidate.person_boxes = [
            detection.bbox_xyxy
            for detection in candidate.detections
            if detection.label == "person"
        ]
        structure_detections = [
            detection
            for detection in candidate.detections
            if detection.label in STRUCTURAL_CLASSES
        ]
        candidate.structural_boxes = [item.bbox_xyxy for item in structure_detections]
        candidate.structural_objects = [
            {
                "category": item.label,
                "bbox_xyxy": item.bbox_xyxy,
                "confidence": item.confidence,
                "source": item.source,
            }
            for item in structure_detections
        ]
        person_mask = _mask_from_boxes((height, width), candidate.person_boxes)
        structure_mask = _mask_from_boxes((height, width), candidate.structural_boxes)
        candidate.person_mask = person_mask
        candidate.person_area_ratio = round(float(np.mean(person_mask > 0)), 8)
        candidate.structure_area_ratio = round(float(np.mean(structure_mask > 0)), 8)
        structure_pixels = int(np.count_nonzero(structure_mask))
        overlap_pixels = int(np.count_nonzero((structure_mask > 0) & (person_mask > 0)))
        candidate.structure_occlusion_ratio = round(
            overlap_pixels / structure_pixels if structure_pixels else 0.0, 8
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        analysis_gray = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        candidate.sharpness_raw = round(
            float(cv2.Laplacian(analysis_gray, cv2.CV_64F).var()), 6
        )
        sharpness_values.append(math.log1p(candidate.sharpness_raw))
        candidate.exposure_score = round(_exposure_score(image), 6)
        candidate.composition_score = round(
            _composition_score(image, candidate.structural_boxes), 6
        )
        candidate.descriptor = _scene_descriptor(image, person_mask)

    sharpness_scores = _robust_normalize(sharpness_values)
    descriptors = np.stack([item.descriptor for item in candidates])
    similarities = np.clip(descriptors @ descriptors.T, -1.0, 1.0)
    count = len(candidates)
    neighbour_count = max(1, min(6, count - 1)) if count > 1 else 1
    raw_representativeness: list[float] = []
    centroid = np.median(descriptors, axis=0)
    centroid /= float(np.linalg.norm(centroid) + 1e-8)
    for index in range(count):
        if count == 1:
            local_similarity = 1.0
        else:
            others = np.delete(similarities[index], index)
            local_similarity = float(np.mean(np.sort(others)[-neighbour_count:]))
        centroid_similarity = _clip01((float(descriptors[index] @ centroid) + 1.0) / 2.0)
        raw_representativeness.append(0.65 * local_similarity + 0.35 * centroid_similarity)
    representative_scores = _robust_normalize(raw_representativeness, neutral=1.0)

    for index, candidate in enumerate(candidates):
        candidate.sharpness_score = round(sharpness_scores[index], 6)
        candidate.representativeness_score = round(representative_scores[index], 6)
        neighbour_sims = []
        neighbour_dists = []
        if index > 0:
            neighbour_sims.append(float(similarities[index, index - 1]))
            neighbour_dists.append(1.0 - float(similarities[index, index - 1]))
        if index + 1 < count:
            neighbour_sims.append(float(similarities[index, index + 1]))
            neighbour_dists.append(1.0 - float(similarities[index, index + 1]))
        candidate.stability_score = round(
            _clip01(max(neighbour_sims) if neighbour_sims else 1.0), 6
        )
        # Histogram/texture change is only a proxy for view variation.  The
        # bell-shaped mapping favours modest changes and rejects hard cuts.
        change = min(neighbour_dists) if neighbour_dists else 0.04
        parallax = math.exp(-((change - 0.045) / 0.055) ** 2)
        candidate.parallax_potential_score = round(_clip01(parallax), 6)

        person_clear = math.exp(-4.0 * candidate.person_area_ratio)
        # Absence of a known COCO structural class is unknown, not proof of a
        # fully exposed scene.  A neutral value prevents empty/dark frames from
        # collecting a free perfect structure score.
        structure_clear = (
            1.0 - candidate.structure_occlusion_ratio
            if candidate.structural_boxes
            else 0.65
        )
        base_score = (
            weights["person_clear"] * person_clear
            + weights["structure_clear"] * structure_clear
            + weights["sharpness"] * candidate.sharpness_score
            + weights["exposure"] * candidate.exposure_score
            + weights["composition"] * candidate.composition_score
            + weights["representativeness"] * candidate.representativeness_score
            + weights["stability"] * candidate.stability_score
            + weights["parallax"] * candidate.parallax_potential_score
        )
        # A scene reference must remain usable.  This soft gate makes very dark
        # or very blurred frames lose even when they happen to contain no one.
        candidate.quality_gate = round(
            0.35
            + 0.65
            * math.sqrt(candidate.sharpness_score * candidate.exposure_score),
            6,
        )
        # Treat <=8% person occupancy as tolerable (a small person at an edge),
        # then rapidly suppress medium/close-up portraits.  This prevents an
        # exceptionally sharp face close-up from beating a usable room view.
        candidate.person_gate = round(
            math.exp(-2.0 * max(0.0, candidate.person_area_ratio - 0.08)), 6
        )
        candidate.total_score = round(
            base_score * candidate.quality_gate * candidate.person_gate, 6
        )


def _fit_scene_labels(
    embeddings: np.ndarray, n_clusters: int | None, distance_threshold: float | None
) -> np.ndarray:
    """Average-linkage cosine agglomerative clustering labels.

    Exactly one of ``n_clusters`` / ``distance_threshold`` drives the stop
    criterion; they mirror the mutually exclusive scikit-learn parameters,
    so the default threshold path and the ``--max-scenes auto`` grid path
    share one fitter.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for scene clustering") from exc
    kwargs: dict[str, Any] = {"linkage": "average"}
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    else:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = distance_threshold
    try:
        return AgglomerativeClustering(metric="cosine", **kwargs).fit_predict(embeddings)
    except TypeError:  # scikit-learn < 1.2
        return AgglomerativeClustering(affinity="cosine", **kwargs).fit_predict(embeddings)


def _choose_auto_scene_count(
    embeddings: np.ndarray,
) -> tuple[int | None, dict[str, Any] | None]:
    """Pick the cluster count on the 2..12 grid by mean cosine silhouette.

    ``--max-scenes auto`` path only: every k is fitted with the same
    average-linkage cosine clustering used elsewhere and scored with
    ``silhouette_score`` on the very embeddings that feed the clustering,
    deterministically subsampled to at most ``SILHOUETTE_SAMPLE_LIMIT`` rows
    (evenly spaced, no RNG) when the pool is large.  Ties keep the smaller k
    so equally good scores resolve to fewer, coarser scenes.  Returns
    ``None`` when fewer than three candidates exist; the caller then falls
    back to the threshold-driven default clustering.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for --max-scenes auto") from exc
    count = len(embeddings)
    grid = [k for k in AUTO_SCENE_GRID if k < count]
    if not grid:
        return None, None
    if count > SILHOUETTE_SAMPLE_LIMIT:
        sample_indices = np.unique(
            np.linspace(0, count - 1, SILHOUETTE_SAMPLE_LIMIT).round().astype(np.int64)
        )
        silhouette_embeddings = embeddings[sample_indices]
    else:
        sample_indices = None
        silhouette_embeddings = embeddings
    silhouette_by_k: dict[int, float] = {}
    for k in grid:
        labels = _fit_scene_labels(embeddings, k, None)
        if sample_indices is None:
            score = silhouette_score(embeddings, labels, metric="cosine")
        else:
            score = silhouette_score(
                silhouette_embeddings, labels[sample_indices], metric="cosine"
            )
        silhouette_by_k[k] = round(float(score), 6)
    # Ascending grid + max() keeps the first maximum, so equal rounded
    # scores resolve to the smaller k.
    best_k = max(grid, key=lambda value: silhouette_by_k[value])
    selection = {
        "mode": "auto",
        "criterion": "mean cosine silhouette over the clustering embeddings",
        "search_grid": grid,
        "silhouette_sample_count": int(len(silhouette_embeddings)),
        "silhouette_by_k": {str(value): silhouette_by_k[value] for value in grid},
        "selected_k": best_k,
    }
    return best_k, selection


def _cluster_scenes(
    candidates: list[Candidate],
    embeddings: np.ndarray,
    scene_similarity: float,
    min_scene_frames: int,
    min_scene_quality: float,
    max_scenes: int | str,
) -> tuple[list[SceneCluster], list[SceneCluster], dict[str, Any] | None]:
    """Cluster recurring visual environments; no cut detection is involved."""
    selection: dict[str, Any] | None = None
    resolved_max_scenes: int | None = None if max_scenes == "auto" else max_scenes
    if len(candidates) == 1:
        labels = np.zeros(1, dtype=np.int32)
    else:
        if max_scenes == "auto":
            # Grid path: silhouette picks k, then the final partition is the
            # n_clusters=best_k fit.  An empty grid (very small pool) keeps
            # the threshold-driven default, hence the None fallback below.
            resolved_max_scenes, selection = _choose_auto_scene_count(embeddings)
            labels = _fit_scene_labels(
                embeddings,
                resolved_max_scenes,
                None if resolved_max_scenes is not None else 1.0 - scene_similarity,
            )
        else:
            labels = _fit_scene_labels(embeddings, None, 1.0 - scene_similarity)

    raw: list[tuple[list[Candidate], np.ndarray]] = []
    for label in sorted(set(int(value) for value in labels)):
        indices = np.flatnonzero(labels == label)
        members = [candidates[int(index)] for index in indices]
        centroid = embeddings[indices].mean(axis=0)
        centroid /= float(np.linalg.norm(centroid) + 1e-8)
        raw.append((members, centroid.astype(np.float32)))
    raw.sort(key=lambda item: min(member.timestamp_sec for member in item[0]))

    clusters: list[SceneCluster] = []
    for cluster_id, (members, centroid) in enumerate(raw, 1):
        for member in members:
            member.scene_cluster_id = cluster_id
        clusters.append(SceneCluster(cluster_id, members, centroid))

    candidate_count = max(1, len(candidates))
    for cluster in clusters:
        quality = float(np.mean(sorted(
            (item.total_score for item in cluster.members), reverse=True
        )[: min(3, len(cluster.members))]))
        support = _clip01(math.sqrt(len(cluster.members) / candidate_count) * 2.2)
        cohesion = float(np.mean([
            _cosine(item.dino_embedding, cluster.centroid) for item in cluster.members
        ]))
        other_centroids = [
            other.centroid for other in clusters if other.cluster_id != cluster.cluster_id
        ]
        nearest_other = max(
            (_cosine(cluster.centroid, other) for other in other_centroids), default=0.0
        )
        distinctiveness = _clip01((1.0 - nearest_other) / 0.65)
        cluster.importance_score = round(
            0.55 * quality
            + 0.20 * support
            + 0.15 * _clip01(cohesion)
            + 0.10 * distinctiveness,
            6,
        )

    eligible = [
        item
        for item in clusters
        if len(item.members) >= min_scene_frames
        and max(member.total_score for member in item.members) >= min_scene_quality
    ]
    # A very short input can have no repeated cluster.  Return its best visual
    # group instead of failing, but do not generally promote transient frames.
    if not eligible and clusters:
        eligible = [max(clusters, key=lambda item: item.importance_score)]
    important = sorted(
        eligible, key=lambda item: item.importance_score, reverse=True
    )[:resolved_max_scenes]
    important.sort(key=lambda item: min(member.timestamp_sec for member in item.members))
    return clusters, important, selection


def _select_scene_frames(
    cluster: SceneCluster,
    frames_per_scene: int,
    minimum_gap_sec: float,
    aux_similarity: float,
    aux_selection: str = "legacy",
    anchor_selection: str = "topk",
    max_aux_similarity: float = 0.95,
    aux_min_score: float = 0.15,
) -> list[Candidate]:
    """Select related main/aux views inside exactly one DINO scene cluster."""
    if not cluster.members:
        return []

    def centroid_similarity(item: Candidate) -> float:
        return _cosine(item.dino_embedding, cluster.centroid)

    main = max(
        cluster.members,
        key=lambda item: 0.82 * item.total_score + 0.18 * _clip01(centroid_similarity(item)),
    )
    main.selection_score = round(
        0.82 * main.total_score + 0.18 * _clip01(centroid_similarity(main)), 6
    )
    selected = [main]
    for item in cluster.members:
        item.dino_similarity_to_main = round(
            _cosine(item.dino_embedding, main.dino_embedding), 6
        )
        item.same_scene_as_main = item.scene_cluster_id == main.scene_cluster_id

    # Membership alone is insufficient for aux: require a direct main/aux
    # similarity as protection against chaining in agglomerative clustering.
    remaining = [
        item
        for item in cluster.members
        if item is not main and item.dino_similarity_to_main >= aux_similarity
    ]

    def selection_value(item: Candidate) -> float:
        time_distance = min(abs(item.timestamp_sec - old.timestamp_sec) for old in selected)
        time_diversity = _clip01(time_distance / max(1.0, minimum_gap_sec * 3.0))
        appearance_change = min(
            1.0 - _cosine(item.descriptor, old.descriptor) for old in selected
        )
        return (
            0.72 * item.total_score
            + 0.13 * _clip01(centroid_similarity(item))
            + 0.10 * time_diversity
            + 0.05 * _clip01(appearance_change * 8.0)
        )

    if anchor_selection == "coverage":
        # Coverage overrides the aux policy; --aux-selection only governs
        # the default topk mode.
        _select_aux_coverage(
            cluster,
            selected,
            remaining,
            frames_per_scene,
            minimum_gap_sec,
        )
    elif aux_selection == "marginal":
        _select_aux_marginal(
            cluster,
            selected,
            remaining,
            frames_per_scene,
            minimum_gap_sec,
            max_aux_similarity,
            aux_min_score,
            selection_value,
        )
    else:
        while remaining and len(selected) < min(frames_per_scene, len(cluster.members)):
            spaced = [
                item
                for item in remaining
                if all(
                    abs(item.timestamp_sec - old.timestamp_sec) >= minimum_gap_sec
                    for old in selected
                )
            ]
            pool = spaced or remaining
            chosen = max(pool, key=selection_value)
            chosen.selection_score = round(selection_value(chosen), 6)
            selected.append(chosen)
            remaining.remove(chosen)

    for rank, item in enumerate(selected, 1):
        item.selected_rank = rank
    cluster.selected = selected
    return selected


def _select_aux_marginal(
    cluster: SceneCluster,
    selected: list[Candidate],
    remaining: list[Candidate],
    frames_per_scene: int,
    minimum_gap_sec: float,
    max_aux_similarity: float,
    aux_min_score: float,
    selection_value: Any,
) -> None:
    """Greedy marginal-information aux selection (task E1).

    Same cluster membership and the same ``dino_similarity_to_main >=
    aux_similarity`` floor as legacy (``remaining`` is already gated), but
    each round scores a candidate against *all* already-selected anchors via
    pairwise DINO cosine: a complementarity reward ``0.25 * (1 -
    max_sim_to_selected)`` favours views adding new visual information, and a
    0.5 near-duplicate penalty demotes candidates whose
    ``max_sim_to_selected`` exceeds ``max_aux_similarity``.  Candidates below
    ``aux_min_score`` only fill a scene that would otherwise stay short, and
    are flagged with ``selection_note``.
    """

    def max_sim_to_selected(item: Candidate) -> float:
        return max(_cosine(item.dino_embedding, old.dino_embedding) for old in selected)

    def marginal_value(item: Candidate) -> float:
        max_sim = max_sim_to_selected(item)
        # ``selection_value`` closes over ``selected`` and stays current as the
        # list grows, so the legacy quality term is reused unchanged.
        complementarity = 0.25 * (1.0 - max_sim)
        near_duplicate_penalty = 0.5 if max_sim > max_aux_similarity else 0.0
        return selection_value(item) + complementarity - near_duplicate_penalty

    target = min(frames_per_scene, len(cluster.members))
    pools = [
        (
            [item for item in remaining if item.total_score >= aux_min_score],
            False,
        ),
        (
            [item for item in remaining if item.total_score < aux_min_score],
            True,
        ),
    ]
    for pool, below_min_score in pools:
        while pool and len(selected) < target:
            spaced = [
                item
                for item in pool
                if all(
                    abs(item.timestamp_sec - old.timestamp_sec) >= minimum_gap_sec
                    for old in selected
                )
            ]
            round_pool = spaced or pool
            chosen = max(round_pool, key=marginal_value)
            chosen.selection_score = round(marginal_value(chosen), 6)
            chosen.max_sim_to_selected = round(max_sim_to_selected(chosen), 6)
            if below_min_score:
                chosen.selection_note = (
                    f"aux_min_score_relaxed: total_score {chosen.total_score:.6f} "
                    f"< {aux_min_score}"
                )
            selected.append(chosen)
            pool.remove(chosen)


def _select_aux_coverage(
    cluster: SceneCluster,
    selected: list[Candidate],
    remaining: list[Candidate],
    frames_per_scene: int,
    minimum_gap_sec: float,
) -> None:
    """k-center greedy aux anchors for viewpoint/composition coverage.

    ``--anchor-selection coverage`` path only.  This is the standard
    farthest-point traversal, i.e. the greedy approximation of the
    submodular facility-location coverage objective over same-cluster
    DINOv3 cosine distances: after the unchanged quality-best main seed,
    every next anchor is the candidate whose *closest* already-selected
    anchor lies the farthest away, so each pick adds the largest amount of
    uncovered viewing geometry instead of the next-best near-duplicate.
    ``remaining`` arrives already gated by the aux-to-main similarity floor,
    and temporal spacing stays a soft preference (spaced candidates first,
    whole pool as fallback), so coverage never returns near-duplicate
    timestamps or off-scene frames.  Trade-off vs topk: aux slots ignore
    total_score, so a rare but blurrier view can beat a sharper duplicate;
    the main anchor still follows the unchanged quality rule.
    """

    def min_distance_to_selected(item: Candidate) -> float:
        return min(
            1.0 - _cosine(item.dino_embedding, old.dino_embedding) for old in selected
        )

    target = min(frames_per_scene, len(cluster.members))
    while remaining and len(selected) < target:
        spaced = [
            item
            for item in remaining
            if all(
                abs(item.timestamp_sec - old.timestamp_sec) >= minimum_gap_sec
                for old in selected
            )
        ]
        pool = spaced or remaining
        chosen = max(pool, key=min_distance_to_selected)
        distance = min_distance_to_selected(chosen)
        chosen.selection_score = round(distance, 6)
        chosen.max_sim_to_selected = round(1.0 - distance, 6)
        selected.append(chosen)
        remaining.remove(chosen)


def _debug_overlay(candidate: Candidate) -> np.ndarray:
    image = candidate.image_bgr.copy()
    for detection in candidate.detections:
        if detection.label == "person":
            colour = (30, 30, 240)
        elif detection.label in STRUCTURAL_CLASSES:
            colour = (240, 190, 30)
        else:
            continue
        x0, y0, x1, y1 = detection.bbox_xyxy
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, 3)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"t={candidate.timestamp_sec:.2f}s  score={candidate.total_score:.3f}  "
        f"person={candidate.person_area_ratio:.3f}  occ={candidate.structure_occlusion_ratio:.3f}",
        (14, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _contact_sheet(selected: Sequence[Candidate], output: Path) -> None:
    cell_width, image_height, label_height = 560, 315, 74
    cells: list[np.ndarray] = []
    for candidate in selected:
        thumb = cv2.resize(
            candidate.image_bgr, (cell_width, image_height), interpolation=cv2.INTER_AREA
        )
        canvas = np.zeros((image_height + label_height, cell_width, 3), dtype=np.uint8)
        canvas[:image_height] = thumb
        rank_name = "MAIN" if candidate.selected_rank == 1 else f"AUX {candidate.selected_rank - 1}"
        cv2.putText(
            canvas,
            f"{rank_name} | t={candidate.timestamp_sec:.2f}s | total={candidate.total_score:.3f}",
            (12, image_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"DINO scene={candidate.scene_cluster_id} sim_main={candidate.dino_similarity_to_main:.2f} "
            f"person={candidate.person_area_ratio:.3f} sharp={candidate.sharpness_score:.2f}",
            (12, image_height + 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (190, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(canvas)
    cv2.imwrite(str(output), np.hstack(cells))


def _scene_overview(clusters: Sequence[SceneCluster], output: Path) -> None:
    if not clusters:
        return
    cell_width, image_height, label_height = 448, 252, 72
    columns = min(4, len(clusters))
    rows = int(math.ceil(len(clusters) / columns))
    sheet = np.zeros(
        (rows * (image_height + label_height), columns * cell_width, 3), dtype=np.uint8
    )
    for index, cluster in enumerate(clusters):
        main = cluster.selected[0]
        row, column = divmod(index, columns)
        y0, x0 = row * (image_height + label_height), column * cell_width
        sheet[y0 : y0 + image_height, x0 : x0 + cell_width] = cv2.resize(
            main.image_bgr, (cell_width, image_height), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            sheet,
            f"SCENE {index + 1:03d}  t={main.timestamp_sec:.1f}s  score={main.total_score:.2f}",
            (x0 + 10, y0 + image_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"cluster={cluster.cluster_id}  n={len(cluster.members)}  importance={cluster.importance_score:.2f}",
            (x0 + 10, y0 + image_height + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (190, 220, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), sheet)


def _process_video(
    video_path: Path,
    output_root: Path,
    detector: YoloSceneDetector,
    embedder: Dinov3SceneEmbedder,
    args: argparse.Namespace,
    shot_boundaries: Sequence[float] | None = None,
) -> dict[str, Any]:
    started = time.time()
    output_dir = output_root / video_path.stem
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output already exists (pass --overwrite to replace it): {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, video_info = _sample_video(
        video_path,
        args.sample_interval,
        args.min_candidates,
        args.max_candidates,
        shot_boundaries=shot_boundaries,
    )
    # Keep tiled-YOLO memory bounded when a long video reaches hundreds of
    # samples.  DINO already batches internally.
    for chunk in _chunks(candidates, max(1, args.batch_size * 2)):
        detections = detector.detect([item.image_bgr for item in chunk])
        for candidate, items in zip(chunk, detections):
            candidate.detections = items
    _measure_candidates(candidates, DEFAULT_WEIGHTS)
    embeddings = embedder.embed(candidates)
    clusters, important, scene_selection = _cluster_scenes(
        candidates,
        embeddings,
        args.scene_similarity,
        args.min_scene_frames,
        args.min_scene_quality,
        args.max_scenes,
    )
    if not important:
        raise RuntimeError(f"no important scene cluster found for {video_path}")

    # Coverage anchor selection overrides the aux policy; topk leaves it alone.
    aux_strategy = "coverage" if args.anchor_selection == "coverage" else args.aux_selection

    scene_records: list[dict[str, Any]] = []
    for scene_number, cluster in enumerate(important, 1):
        selected = _select_scene_frames(
            cluster,
            args.frames_per_scene,
            args.minimum_gap_sec,
            args.aux_similarity,
            aux_selection=args.aux_selection,
            anchor_selection=args.anchor_selection,
            max_aux_similarity=args.max_aux_similarity,
            aux_min_score=args.aux_min_score,
        )
        if not selected:
            continue
        scene_name = f"scene_{scene_number:03d}"
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        main = selected[0]
        cv2.imwrite(str(scene_dir / "anchor_main.jpg"), main.image_bgr)
        cv2.imwrite(str(scene_dir / "person_mask_main.png"), main.person_mask)
        cv2.imwrite(str(scene_dir / "anchor_main_debug.jpg"), _debug_overlay(main))
        for aux_index, candidate in enumerate(selected[1:], 1):
            cv2.imwrite(
                str(scene_dir / f"anchor_aux_{aux_index:02d}.jpg"), candidate.image_bgr
            )
        _contact_sheet(selected, scene_dir / "topk_preview.jpg")

        selected_records = []
        for item in selected:
            record = {
                "rank": item.selected_rank,
                "asset": "anchor_main.jpg"
                if item.selected_rank == 1
                else f"anchor_aux_{item.selected_rank - 1:02d}.jpg",
                "frame_index": item.frame_index,
                "timestamp_sec": item.timestamp_sec,
                "scene_cluster_id": item.scene_cluster_id,
                "same_scene_as_main": item.same_scene_as_main,
                "dino_similarity_to_main": item.dino_similarity_to_main,
                "total_score": item.total_score,
                "selection_score": item.selection_score,
                "person_area_ratio": item.person_area_ratio,
                "structure_occlusion_ratio": item.structure_occlusion_ratio,
            }
            # E1/coverage: extra fields only in non-legacy modes so legacy
            # output stays byte-identical to the pre-E1 schema.
            if aux_strategy != "legacy":
                record["selection_strategy"] = aux_strategy
                record["max_sim_to_selected"] = item.max_sim_to_selected
                if item.selection_note:
                    record["selection_note"] = item.selection_note
            selected_records.append(record)
        scene_record = {
            "scene_name": scene_name,
            "scene_cluster_id": cluster.cluster_id,
            "importance_score": cluster.importance_score,
            "sample_count": len(cluster.members),
            "sample_timestamps_sec": [item.timestamp_sec for item in cluster.members],
            "returned_frame_count": len(selected),
            "relation_constraint_satisfied": all(
                item.same_scene_as_main
                and item.dino_similarity_to_main >= args.aux_similarity
                for item in selected[1:]
            ),
            "selected": selected_records,
        }
        _json_dump(
            scene_dir / "scene_meta.json",
            {
                "schema_version": "important-scene-assets-v3",
                "source_video": str(video_path.resolve()),
                **scene_record,
            },
        )
        scene_records.append(scene_record)

    if not scene_records:
        raise RuntimeError(f"scene clusters exist but no frames were selected for {video_path}")
    _scene_overview(important, output_dir / "all_scenes_preview.jpg")

    ordered = sorted(candidates, key=lambda item: item.total_score, reverse=True)
    score_report = {
        "schema_version": "important-scene-candidates-v3",
        "video_path": str(video_path.resolve()),
        "score_weights": DEFAULT_WEIGHTS,
        "candidate_count": len(candidates),
        "dino_cluster_count": len(clusters),
        "candidates": [item.to_json() for item in ordered],
    }
    _json_dump(output_dir / "candidate_scores.json", score_report)

    meta = {
        "schema_version": "important-scene-index-v3",
        "video_id": video_path.stem,
        "source_video": str(video_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_policy": {
            "llm_used": False,
            "vlm_used": False,
            "remote_api_used": False,
            "shot_detection_used": False,
            "scene_embedder": "local DINOv3 ViT-L/16",
            "dino_weights": str(embedder.model_path),
            "object_detector": "YOLOv8n COCO",
            "detector_weights": str(detector.model_path),
            "person_segmentation": "conservative_detection_bbox_union_proxy",
            "classical_cv_metrics": [
                "laplacian_sharpness",
                "luminance_exposure",
                "edge_distribution_composition",
                "background_colour_texture_representativeness",
            ],
        },
        "video": video_info,
        "sampling": {
            "sample_interval_sec": args.sample_interval,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "candidate_count": len(candidates),
            **(
                {
                    "shots_manifest": str(Path(args.shots_manifest).expanduser().resolve()),
                    "shot_boundary_count": video_info["shot_boundary_count"],
                    "snap_tolerance_sec": video_info["snap_tolerance_sec"],
                    "snapped_timestamp_count": video_info["snapped_timestamp_count"],
                }
                if args.shots_manifest
                else {}
            ),
        },
        "scoring": {
            "weights": DEFAULT_WEIGHTS,
            "no_detected_structure_clear_score": 0.65,
            "quality_gate": "0.35 + 0.65 * sqrt(sharpness_score * exposure_score)",
            "person_gate": "exp(-2 * max(0, person_area_ratio - 0.08))",
        },
        "detector": {
            "confidence": args.confidence,
            "structure_confidence": args.structure_confidence,
            "image_size": args.image_size,
            "tiled_small_person_detection": not args.no_tiled_people,
            "tile_overlap": args.tile_overlap,
            "structural_classes": sorted(STRUCTURAL_CLASSES),
        },
        "scene_discovery": {
            "method": "average-linkage cosine clustering over person-suppressed DINOv3 patch embeddings",
            "scene_similarity_threshold": args.scene_similarity,
            "all_cluster_count": len(clusters),
            "min_scene_frames": args.min_scene_frames,
            "min_scene_quality": args.min_scene_quality,
            "max_scenes": args.max_scenes,
            **(
                {"selection": scene_selection}
                if scene_selection is not None
                else {}
            ),
            "returned_scene_count": len(scene_records),
            "frames_per_scene": args.frames_per_scene,
            "minimum_gap_sec": args.minimum_gap_sec,
            "aux_similarity_threshold": args.aux_similarity,
            "relation_policy": "same DINO cluster and direct aux-to-main cosine threshold",
            **(
                {"anchor_selection": args.anchor_selection}
                if args.anchor_selection != "topk"
                else {}
            ),
            **(
                {
                    "aux_selection": args.aux_selection,
                    "max_aux_similarity": args.max_aux_similarity,
                    "aux_min_score": args.aux_min_score,
                }
                if args.aux_selection == "marginal"
                else {}
            ),
        },
        "scenes": scene_records,
        "limitations": [
            "person_mask_main.png is a conservative bbox-union proxy, not an instance-segmentation mask",
            "structural coverage is limited to classes available in the COCO detector",
            "parallax_potential_score is an appearance-change proxy, not camera calibration",
            "very short scenes below the sampling interval can be missed",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _json_dump(output_dir / "scene_index.json", meta)
    return meta


def _resolve_videos(args: argparse.Namespace) -> list[Path]:
    inputs: list[Path] = []
    inputs.extend(Path(value).expanduser() for value in args.input)
    if args.input_dir:
        directory = Path(args.input_dir).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(f"input directory does not exist: {directory}")
        inputs.extend(sorted(directory.glob(args.pattern)))
    videos: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen:
            videos.append(resolved)
            seen.add(resolved)
    if not videos:
        raise FileNotFoundError("no input videos found; pass VIDEO paths or --input-dir")
    if args.random_count is not None and args.random_count < len(videos):
        videos = sorted(random.Random(args.seed).sample(videos, args.random_count))
    return videos


def _parse_max_scenes(value: str) -> int | str:
    """argparse type: a positive integer or the literal ``auto``."""
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--max-scenes must be a positive integer or 'auto', got {value!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover important scenes using local DINOv3 + YOLOv8n; no LLM/VLM/cut detection."
    )
    parser.add_argument("input", nargs="*", help="input video path(s)")
    parser.add_argument("--input-dir", help="directory containing videos")
    parser.add_argument("--pattern", default="*.mp4", help="input-dir glob (default: *.mp4)")
    parser.add_argument("--output-dir", required=True, help="output root directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="local YOLO weights")
    parser.add_argument("--dino-model", default=str(DEFAULT_DINO_MODEL), help="local DINOv3 model directory")
    parser.add_argument("--device", default="0", help="model device, e.g. 0, cuda:0, or cpu")
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument(
        "--structure-confidence",
        type=float,
        default=0.30,
        help="higher threshold for noisier COCO structural classes",
    )
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--no-tiled-people", action="store_true", help="disable small-person tiled inference")
    parser.add_argument("--tile-overlap", type=float, default=0.18)
    parser.add_argument("--sample-interval", type=float, default=1.2)
    parser.add_argument("--min-candidates", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument(
        "--shots-manifest",
        default=None,
        help="optional shots.jsonl (one JSON object per line with "
        "time_range_sec=[t0,t1]); sampled timestamps snap to the nearest "
        "t0/t1 boundary within 0.6s and collapsed duplicates are merged",
    )
    parser.add_argument(
        "--scene-similarity",
        type=float,
        default=0.65,
        help="minimum average-link DINO similarity within a scene",
    )
    parser.add_argument(
        "--aux-similarity",
        type=float,
        default=0.78,
        help="hard minimum DINO cosine similarity from each aux to its main",
    )
    parser.add_argument(
        "--aux-selection",
        choices=("legacy", "marginal"),
        default="legacy",
        help="aux anchor policy: legacy keeps the pre-E1 behaviour; marginal "
        "greedily picks the aux adding the most new information vs all "
        "already-selected anchors",
    )
    parser.add_argument(
        "--max-aux-similarity",
        type=float,
        default=0.95,
        help="marginal mode only: DINO cosine above this to any selected "
        "anchor is penalised as a near-duplicate",
    )
    parser.add_argument(
        "--aux-min-score",
        type=float,
        default=0.15,
        help="marginal mode only: minimum total_score for an aux candidate "
        "(relaxed with a selection_note if a scene would otherwise stay short)",
    )
    parser.add_argument(
        "--anchor-selection",
        choices=("topk", "coverage"),
        default="topk",
        help="anchor policy: topk keeps the current score-ranked main/aux "
        "picks; coverage picks aux anchors by k-center greedy over "
        "same-cluster DINOv3 cosine distance for viewpoint diversity "
        "(overrides --aux-selection)",
    )
    parser.add_argument("--min-scene-frames", type=int, default=2)
    parser.add_argument(
        "--min-scene-quality",
        type=float,
        default=0.25,
        help="minimum best-frame usability score for an important scene",
    )
    parser.add_argument(
        "--max-scenes",
        type=_parse_max_scenes,
        default=12,
        help="maximum important scenes to return, or 'auto' to pick k on the "
        "2..12 grid by best mean silhouette",
    )
    parser.add_argument("--frames-per-scene", type=int, default=3)
    parser.add_argument("--minimum-gap-sec", type=float, default=2.0)
    parser.add_argument("--random-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be in (0, 1)")
    if not 0.0 < args.structure_confidence < 1.0:
        raise ValueError("--structure-confidence must be in (0, 1)")
    if not 0.0 <= args.tile_overlap < 0.9:
        raise ValueError("--tile-overlap must be in [0, 0.9)")
    if args.sample_interval <= 0 or args.minimum_gap_sec < 0:
        raise ValueError("sampling intervals must be positive")
    if args.min_candidates <= 0 or args.max_candidates < args.min_candidates:
        raise ValueError("require 0 < min-candidates <= max-candidates")
    if not 0.0 < args.scene_similarity < 1.0:
        raise ValueError("--scene-similarity must be in (0, 1)")
    if not 0.0 < args.aux_similarity <= 1.0:
        raise ValueError("--aux-similarity must be in (0, 1]")
    if not 0.0 < args.max_aux_similarity <= 1.0:
        raise ValueError("--max-aux-similarity must be in (0, 1]")
    if not 0.0 <= args.aux_min_score <= 1.0:
        raise ValueError("--aux-min-score must be in [0, 1]")
    if args.min_scene_frames <= 0 or (
        isinstance(args.max_scenes, int) and args.max_scenes <= 0
    ):
        raise ValueError("--min-scene-frames and --max-scenes must be positive")
    if not 0.0 <= args.min_scene_quality <= 1.0:
        raise ValueError("--min-scene-quality must be in [0, 1]")
    if args.frames_per_scene <= 0:
        raise ValueError("--frames-per-scene must be positive")
    if args.random_count is not None and args.random_count <= 0:
        raise ValueError("--random-count must be positive")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    videos = _resolve_videos(args)
    shot_boundaries: list[float] | None = None
    if args.shots_manifest:
        shot_boundaries = _load_shot_boundaries(Path(args.shots_manifest).expanduser())
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"selected {len(videos)} video(s): {[path.name for path in videos]}")
    print("model policy: local DINOv3 + YOLOv8n + OpenCV; LLM/VLM/API/cut detection disabled")
    detector = YoloSceneDetector(
        model_path=Path(args.model).expanduser(),
        device=args.device,
        confidence=args.confidence,
        structure_confidence=args.structure_confidence,
        image_size=args.image_size,
        tiled_people=not args.no_tiled_people,
        tile_overlap=args.tile_overlap,
        batch_size=args.batch_size,
    )
    embedder = Dinov3SceneEmbedder(
        model_path=Path(args.dino_model).expanduser(),
        device=args.device,
        batch_size=args.batch_size,
    )
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] processing {video.name}", flush=True)
        try:
            report = _process_video(
                video,
                output_root,
                detector,
                embedder,
                args,
                shot_boundaries=shot_boundaries,
            )
            reports.append(report)
            main_anchor = report["scenes"][0]["selected"][0]
            print(
                f"  scenes={len(report['scenes'])}; first main: "
                f"t={main_anchor['timestamp_sec']:.2f}s score={main_anchor['total_score']:.3f}",
                flush=True,
            )
        except Exception as exc:  # continue a directory run and report failures
            failures.append({"video": str(video), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    manifest = {
        "schema_version": "important-scene-run-v2",
        "model_policy": "local DINOv3 + YOLOv8n + OpenCV; no LLM/VLM/remote API/cut detection",
        "seed": args.seed,
        **(
            {"shots_manifest": str(Path(args.shots_manifest).expanduser().resolve())}
            if args.shots_manifest
            else {}
        ),
        "requested_videos": [str(path) for path in videos],
        "successful_count": len(reports),
        "failure_count": len(failures),
        "videos": [
            {
                "video_id": report["video_id"],
                "source_video": report["source_video"],
                "output_dir": report["output_dir"],
                "scene_count": len(report["scenes"]),
                **(
                    {
                        "shot_boundary_count": report["sampling"]["shot_boundary_count"],
                        "snapped_timestamp_count": report["sampling"][
                            "snapped_timestamp_count"
                        ],
                    }
                    if args.shots_manifest
                    else {}
                ),
                "scenes": report["scenes"],
            }
            for report in reports
        ],
        "failures": failures,
    }
    _json_dump(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
