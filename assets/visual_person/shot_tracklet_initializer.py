#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initialize shot-local Tracklets from standalone shot MP4 files.

This module implements the step immediately after physical shot extraction:
for every ``*.mp4`` shot, decode its first usable frame, call the stable
single-frame Observation pipeline from ``entity_asset_extractor.py``, and
initialize one ACTIVE Tracklet per Observation.

It intentionally does not propagate tracks, perform periodic re-detection,
associate later observations, or resolve cross-shot character identities.

Example:
    python shot_tracklet_initializer.py \
        --shots-dir workspace/cuts/demo_full/shots \
        --output workspace/tracklet_initialization
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import entity_asset_extractor as observation_backend


SCHEMA_VERSION = "shot-tracklet-initialization-v2"


@dataclass(frozen=True)
class ShotInput:
    shot_id: str
    video_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class FrameQuality:
    """Auditable, model-free quality measurements for one candidate frame."""

    frame_index: int
    timestamp_sec: float
    valid: bool
    quality_score: float
    sharpness_score: float
    contrast_score: float
    exposure_score: float
    dark_fraction: float
    bright_fraction: float
    mean_luma: float
    invalid_reason: Optional[str] = None


@dataclass
class Tracklet:
    """Minimal shot-local Tracklet initialized by exactly one Observation."""

    tracklet_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    state: str
    observation_ids: List[str]
    observations: List[Dict[str, Any]]
    last_bbox: Optional[List[int]]
    last_mask_path: Optional[str]
    missed_frames: int = 0
    age: int = 1
    face_assets: List[Dict[str, Any]] = field(default_factory=list)
    body_assets: List[Dict[str, Any]] = field(default_factory=list)
    face_embeddings: List[List[float]] = field(default_factory=list)
    body_embeddings: List[List[float]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    identity_id: Optional[str] = None


def canonical_shot_id(stem: str) -> str:
    """Convert a filename stem into a stable, filesystem-safe shot ID."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()
    if not cleaned:
        raise ValueError(f"Cannot derive shot_id from filename stem: {stem!r}")
    match = re.fullmatch(r"SHOT_?(\d+)", cleaned)
    if match:
        return f"SHOT{int(match.group(1)):04d}"
    if cleaned.startswith("SHOT"):
        return cleaned
    return f"SHOT_{cleaned}"


def discover_shots(
    shots_dir: Path | str,
    pattern: str = "*.mp4",
    recursive: bool = False,
) -> List[Path]:
    root = Path(shots_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = sorted(path.resolve() for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No shot videos matching {pattern!r} under {root}")
    ids: Dict[str, Path] = {}
    for path in paths:
        shot_id = canonical_shot_id(path.stem)
        if shot_id in ids:
            raise ValueError(
                f"Duplicate canonical shot_id {shot_id}: {ids[shot_id]} and {path}"
            )
        ids[shot_id] = path
    return paths


def inspect_shot(video_path: Path | str) -> ShotInput:
    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open shot video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid shot metadata: frames={frame_count}, fps={fps}, "
            f"size={width}x{height}, path={path}"
        )
    return ShotInput(
        shot_id=canonical_shot_id(path.stem),
        video_path=str(path),
        frame_count=frame_count,
        fps=round(fps, 6),
        width=width,
        height=height,
        duration_sec=round(frame_count / fps, 6),
    )


def read_first_usable_frame(
    video_path: Path | str,
    max_decode_attempts: int = 8,
) -> Tuple[int, Any]:
    """Decode the first usable frame, tolerating a few damaged leading frames."""
    if max_decode_attempts < 1:
        raise ValueError("max_decode_attempts must be at least 1")
    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open shot video: {path}")
    try:
        for frame_index in range(max_decode_attempts):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                return frame_index, frame
    finally:
        capture.release()
    raise RuntimeError(
        f"No usable frame among first {max_decode_attempts} frames: {path}"
    )


def evaluate_frame_quality(
    frame: Any,
    frame_index: int,
    fps: float,
) -> FrameQuality:
    """Score sharpness, contrast and exposure without loading a visual model."""
    if frame is None or not getattr(frame, "size", 0):
        return FrameQuality(
            frame_index=frame_index,
            timestamp_sec=round(frame_index / fps, 6),
            valid=False,
            quality_score=0.0,
            sharpness_score=0.0,
            contrast_score=0.0,
            exposure_score=0.0,
            dark_fraction=1.0,
            bright_fraction=0.0,
            mean_luma=0.0,
            invalid_reason="empty_frame",
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Downscaling makes quality comparison cheap and reduces compression noise.
    height, width = gray.shape
    if width > 640:
        scale = 640.0 / width
        gray = cv2.resize(
            gray,
            (640, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    mean_luma = float(gray.mean())
    contrast = float(gray.std())
    dark_fraction = float(np.mean(gray <= 16))
    bright_fraction = float(np.mean(gray >= 245))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sharpness_score = 1.0 - float(np.exp(-laplacian_variance / 300.0))
    contrast_score = min(contrast / 64.0, 1.0)
    exposure_score = max(0.0, 1.0 - abs(mean_luma - 127.5) / 127.5)
    clipping_penalty = min(1.0, dark_fraction + bright_fraction)
    score = (
        0.50 * sharpness_score
        + 0.20 * contrast_score
        + 0.30 * exposure_score
        - 0.35 * clipping_penalty
    )
    invalid_reason: Optional[str] = None
    if dark_fraction >= 0.85:
        invalid_reason = "black_or_severely_underexposed"
    elif bright_fraction >= 0.85:
        invalid_reason = "white_or_severely_overexposed"
    elif contrast < 3.0:
        invalid_reason = "near_uniform_frame"
    valid = invalid_reason is None
    return FrameQuality(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / fps, 6),
        valid=valid,
        quality_score=round(max(0.0, min(score, 1.0)), 6),
        sharpness_score=round(sharpness_score, 6),
        contrast_score=round(contrast_score, 6),
        exposure_score=round(exposure_score, 6),
        dark_fraction=round(dark_fraction, 6),
        bright_fraction=round(bright_fraction, 6),
        mean_luma=round(mean_luma, 3),
        invalid_reason=invalid_reason,
    )


def select_initialization_frame(
    video_path: Path | str,
    *,
    fps: float,
    frame_count: int,
    selection_window_sec: float = 1.5,
    candidate_frames: int = 12,
    min_quality_score: float = 0.12,
    max_decode_attempts: int = 8,
) -> Tuple[int, Any, Dict[str, Any], List[Dict[str, Any]]]:
    """Choose the best valid frame near shot start and return its audit trail.

    The selection window never exceeds the first half of the shot, so a short
    shot still leaves frames for forward tracking and re-detection.
    """
    if selection_window_sec <= 0:
        raise ValueError("selection_window_sec must be positive")
    if candidate_frames < 1:
        raise ValueError("candidate_frames must be at least 1")
    if not 0.0 <= min_quality_score <= 1.0:
        raise ValueError("min_quality_score must be in [0, 1]")
    path = Path(video_path).expanduser().resolve()
    # Clamp the window to the first half of the shot: an anchor near the end
    # of a short shot leaves the forward direction with no frames to track or
    # re-detect in, which starves the tracklet of detection evidence.
    duration = frame_count / max(float(fps), 1e-6)
    effective_window = min(selection_window_sec, 0.5 * duration)
    last_index = min(
        frame_count - 1,
        max(0, int(round(effective_window * fps)) - 1),
    )
    positions = sorted({
        int(round(value))
        for value in np.linspace(0, last_index, min(candidate_frames, last_index + 1))
    })
    position_set = set(positions)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open shot video: {path}")
    decoded: List[Tuple[FrameQuality, Any]] = []
    try:
        for frame_index in range(last_index + 1):
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                continue
            if frame_index not in position_set:
                continue
            quality = evaluate_frame_quality(frame, frame_index, fps)
            decoded.append((quality, frame.copy()))
    finally:
        capture.release()

    valid = [item for item in decoded if item[0].valid]
    expanded_search = False
    if not valid and last_index < frame_count - 1:
        # Some extracted shots begin with a long black leader or fade. Only in
        # that case expand sparsely over the remainder instead of accepting a
        # known-invalid opening frame.
        expanded_search = True
        expanded_positions = sorted({
            int(round(value))
            for value in np.linspace(
                last_index + 1,
                frame_count - 1,
                min(candidate_frames, frame_count - last_index - 1),
            )
        })
        expanded_set = set(expanded_positions)
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, last_index + 1)
                for frame_index in range(last_index + 1, frame_count):
                    ok, frame = capture.read()
                    if not ok or frame is None or not frame.size:
                        continue
                    if frame_index not in expanded_set:
                        continue
                    quality = evaluate_frame_quality(frame, frame_index, fps)
                    decoded.append((quality, frame.copy()))
            finally:
                capture.release()
        valid = [item for item in decoded if item[0].valid]
    eligible = [item for item in valid if item[0].quality_score >= min_quality_score]
    pool = eligible or valid
    if not pool:
        # A pathological all-black/undecodable opening still gets the V1 decode
        # fallback, but the manifest makes that downgrade explicit.
        frame_index, frame = read_first_usable_frame(path, max_decode_attempts)
        fallback = asdict(evaluate_frame_quality(frame, frame_index, fps))
        fallback["selection_reason"] = "fallback_first_decodable_no_valid_candidate"
        return frame_index, frame, fallback, [asdict(item[0]) for item in decoded]

    quality, frame = max(
        pool,
        key=lambda item: (item[0].quality_score, -item[0].frame_index),
    )
    selected = asdict(quality)
    if expanded_search:
        selected["selection_reason"] = "expanded_search_after_invalid_opening"
    elif eligible:
        selected["selection_reason"] = "highest_quality_valid_candidate"
    else:
        selected["selection_reason"] = "highest_quality_below_preferred_threshold"
    return quality.frame_index, frame, selected, [asdict(item[0]) for item in decoded]


def initialize_tracklets(
    shot: ShotInput,
    observations: Sequence[observation_backend.Observation],
) -> Tuple[List[Tracklet], List[List[str]]]:
    """Create Tracklets and all same-frame cannot-link pairs."""
    tracklets: List[Tracklet] = []
    for index, observation in enumerate(observations):
        observation_id = f"{shot.shot_id}_F{observation.frame_index:06d}_OBS{index:02d}"
        tracklet_id = f"{shot.shot_id}_T{index:03d}"
        record = asdict(observation)
        record.update({
            "observation_id": observation_id,
            "shot_id": shot.shot_id,
            "frame_tag": "shot_start",
            "identity_id": None,
        })
        body_bbox = record.get("groundingdino_bbox") or record.get("sa2va_bbox")
        face = record.get("face") or {}
        face_assets = []
        if face.get("detected") and face.get("crop_path"):
            face_assets.append({
                "observation_id": observation_id,
                "crop_path": face["crop_path"],
                "bbox": face.get("bbox"),
                "det_score": face.get("det_score"),
            })
        body_assets = []
        if record.get("body_crop_path"):
            body_assets.append({
                "observation_id": observation_id,
                "crop_path": record["body_crop_path"],
                "masked_crop_path": record.get("masked_crop_path"),
                "rgba_crop_path": record.get("rgba_crop_path"),
                "bbox": body_bbox,
                "grounding_score": record.get("grounding_score"),
            })
        tracklets.append(Tracklet(
            tracklet_id=tracklet_id,
            shot_id=shot.shot_id,
            start_frame=observation.frame_index,
            end_frame=observation.frame_index,
            start_time=round(observation.timestamp_sec, 6),
            end_time=round(observation.timestamp_sec, 6),
            state="ACTIVE",
            observation_ids=[observation_id],
            observations=[record],
            last_bbox=body_bbox,
            last_mask_path=record.get("mask_path"),
            face_assets=face_assets,
            body_assets=body_assets,
            quality={"initialization_observation_count": 1},
            identity_id=None,
        ))

    cannot_links = [
        [tracklets[left].tracklet_id, tracklets[right].tracklet_id]
        for left in range(len(tracklets))
        for right in range(left + 1, len(tracklets))
    ]
    return tracklets, cannot_links


def _write_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _load_completed_shot(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") == SCHEMA_VERSION and payload.get("status") == "ok":
            return payload
    except (OSError, ValueError, TypeError):
        return None
    return None


def process_shot(
    shot: ShotInput,
    output_root: Path,
    models: Dict[str, Any],
    *,
    use_sam: bool,
    crop_padding: float,
    max_decode_attempts: int,
    selection_window_sec: float,
    candidate_frames: int,
    min_quality_score: float,
) -> Dict[str, Any]:
    shot_root = output_root / "shots" / shot.shot_id
    frame_index, frame, selected_quality, quality_candidates = select_initialization_frame(
        shot.video_path,
        fps=shot.fps,
        frame_count=shot.frame_count,
        selection_window_sec=selection_window_sec,
        candidate_frames=candidate_frames,
        min_quality_score=min_quality_score,
        max_decode_attempts=max_decode_attempts,
    )
    frame_path = shot_root / "frames" / f"frame_{frame_index:06d}.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Failed to write initialization frame: {frame_path}")

    sample = observation_backend.FrameSample(
        tag="shot_start",
        percentile=0.0,
        frame_index=frame_index,
        timestamp_sec=frame_index / shot.fps,
        image_path=str(frame_path),
    )
    observations, debug_paths = observation_backend.extract_frame_observations(
        frame_idx=frame_index,
        sample=sample,
        image_bgr=frame,
        sa2va=models["sa2va"],
        gdino=models["gdino"],
        face_app=models["face_app"],
        sam=models["sam"],
        use_sam=use_sam,
        output_root=shot_root,
        crop_padding=crop_padding,
    )
    tracklets, cannot_links = initialize_tracklets(shot, observations)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "stage": "shot_start_tracklet_initialization",
        "shot": asdict(shot),
        "initialization_frame": {
            "frame_index": frame_index,
            "timestamp_sec": round(frame_index / shot.fps, 6),
            "image_path": str(frame_path),
            "quality": selected_quality,
        },
        "frame_quality_candidates": quality_candidates,
        "tracklets": [asdict(tracklet) for tracklet in tracklets],
        "cannot_links": cannot_links,
        "debug_paths": debug_paths,
        "identity_note": "Tracklet != identity; identity_id is null by design.",
    }
    _write_json_atomic(payload, shot_root / "manifest.json")
    return payload


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = discover_shots(args.shots_dir, args.pattern, args.recursive)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be at least 1")
        paths = paths[:args.limit]
    shots = [inspect_shot(path) for path in paths]
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    settings = {
        "pattern": args.pattern,
        "recursive": args.recursive,
        "use_sam": not args.disable_sam,
        "crop_padding": args.crop_padding,
        "max_decode_attempts": args.max_decode_attempts,
        "selection_window_sec": args.selection_window_sec,
        "candidate_frames": args.candidate_frames,
        "min_quality_score": args.min_quality_score,
    }
    if args.dry_run:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run",
            "stage": "shot_discovery",
            "shots": [asdict(shot) for shot in shots],
            "settings": settings,
        }
        _write_json_atomic(manifest, output_root / "manifest.json")
        return manifest

    print("Loading Observation models once for all shots...", flush=True)
    use_sam = not args.disable_sam
    models = {
        "sa2va": observation_backend.load_entity_model(),
        "gdino": observation_backend.load_grounding_dino(),
        "face_app": observation_backend.load_face_detector(),
        "sam": observation_backend.load_sam() if use_sam else None,
    }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for position, shot in enumerate(shots, start=1):
        shot_manifest = output_root / "shots" / shot.shot_id / "manifest.json"
        if args.resume:
            completed = _load_completed_shot(shot_manifest)
            if completed is not None:
                results.append(completed)
                print(f"[{position}/{len(shots)}] {shot.shot_id}: resumed", flush=True)
                continue
        print(f"[{position}/{len(shots)}] {shot.shot_id}: extracting OBS", flush=True)
        try:
            result = process_shot(
                shot,
                output_root,
                models,
                use_sam=use_sam,
                crop_padding=args.crop_padding,
                max_decode_attempts=args.max_decode_attempts,
                selection_window_sec=args.selection_window_sec,
                candidate_frames=args.candidate_frames,
                min_quality_score=args.min_quality_score,
            )
            results.append(result)
            print(
                f"  initialized {len(result['tracklets'])} Tracklets",
                flush=True,
            )
        except Exception as exc:
            error = {
                "shot_id": shot.shot_id,
                "video_path": shot.video_path,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                break

    tracklet_count = sum(len(result.get("tracklets", [])) for result in results)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if not errors else "partial",
        "stage": "shot_start_tracklet_initialization",
        "shots_requested": len(shots),
        "shots_completed": len(results),
        "tracklet_count": tracklet_count,
        "shot_manifests": [
            str(output_root / "shots" / result["shot"]["shot_id"] / "manifest.json")
            for result in results
        ],
        "errors": errors,
        "settings": settings,
        "next_stage": "short_term_propagation_and_periodic_redetection",
    }
    _write_json_atomic(manifest, output_root / "manifest.json")
    if errors and not args.continue_on_error:
        raise RuntimeError(
            f"Stopped after {len(results)} completed shots; see "
            f"{output_root / 'manifest.json'}"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Observations from each shot's first frame and initialize "
            "one shot-local Tracklet per Observation."
        )
    )
    parser.add_argument("--shots-dir", required=True, help="Directory of shot MP4 files.")
    parser.add_argument("--output", required=True, help="Output workspace directory.")
    parser.add_argument("--pattern", default="*.mp4", help="Shot filename glob.")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted shots.")
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--max-decode-attempts", type=int, default=8)
    parser.add_argument(
        "--selection-window-sec", type=float, default=1.5,
        help="Only select initialization frames from this opening window.",
    )
    parser.add_argument(
        "--candidate-frames", type=int, default=12,
        help="Number of evenly spaced quality candidates in the opening window.",
    )
    parser.add_argument(
        "--min-quality-score", type=float, default=0.12,
        help="Preferred minimum quality; falls back to the best valid candidate.",
    )
    parser.add_argument("--disable-sam", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse completed per-shot manifests (default: enabled).",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Record a failed shot and continue with later shots.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect shot files and write metadata without loading models.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(
        f"Finished: status={manifest['status']}, "
        f"shots={manifest.get('shots_completed', len(manifest.get('shots', [])))}, "
        f"tracklets={manifest.get('tracklet_count', 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
