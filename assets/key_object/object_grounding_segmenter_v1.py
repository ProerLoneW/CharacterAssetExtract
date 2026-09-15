#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground key-object candidates on real shot frames and segment instances.

This is the versioned Step 2--6 implementation from
``关键物体提取_Pipeline_开发文档_V1.md``:

    caption candidate -> expected shot -> sampled frames -> GroundingDINO
      -> conservative instance arbitration -> box-prompted SAM -> observation

Every GroundingDINO candidate is retained.  The current arbitration can safely
auto-select a single spatial cluster.  With multiple clusters it uses agreement
across independent grounding phrases and records an ambiguity flag; it does not
pretend that detector confidence alone resolves same-category instances.
``link_clusters_across_frames``/``select_cluster_track`` additionally let V4
callers arbitrate per-frame clusters as cross-frame instance tracks, ranking
positional continuity above single-frame confidence.

SAM3.1 is the target backend, but its Python runtime is not installed in the
current environment.  This executable therefore uses the repository's local
SAM2 base-plus checkpoint and records ``segmentation_backend=sam2_fallback`` in
every output row.  The output schema makes this substitution explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


# local_models 保留在 v2 仓库根目录（本文件在 assets/key_object/ 下，上三级即 v2 根）。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINO_DIR = ROOT / "local_models/AI-ModelScope/GroundingDINO"
DEFAULT_SAM2_REPO = Path(
    "/gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V/"
    "new_data_ppl_v1.0/others/sam2_groundingDINO/segment-anything-2"
)
DEFAULT_SAM2_CHECKPOINT = (
    ROOT / "local_models/segmentation/sam2/sam2_hiera_base_plus.pt"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_shot_id(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("SHOT_"):
        suffix = text[5:]
    else:
        suffix = text
    try:
        return f"SHOT_{int(suffix):03d}"
    except ValueError:
        return text


def unique_phrases(candidate: dict[str, Any]) -> list[str]:
    values = [candidate.get("canonical_name", "")]
    values.extend(candidate.get("grounding_phrases") or [])
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = " ".join(str(value).lower().strip().split())
        if phrase and phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return result


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = [float(v) for v in left]
    rx0, ry0, rx1, ry1 = [float(v) for v in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def clip_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return [
        max(0, min(width - 1, int(math.floor(x0)))),
        max(0, min(height - 1, int(math.floor(y0)))),
        max(1, min(width, int(math.ceil(x1)))),
        max(1, min(height, int(math.ceil(y1)))),
    ]


def cluster_detections(
    detections: Sequence[dict[str, Any]], iou_threshold: float = 0.75
) -> list[dict[str, Any]]:
    """Merge duplicate boxes emitted for different text phrases."""
    clusters: list[dict[str, Any]] = []
    for detection in sorted(
        detections, key=lambda item: float(item["score"]), reverse=True
    ):
        match = next(
            (
                cluster
                for cluster in clusters
                if box_iou(cluster["bbox_xyxy"], detection["bbox_xyxy"])
                >= iou_threshold
            ),
            None,
        )
        if match is None:
            clusters.append(
                {
                    "bbox_xyxy": list(detection["bbox_xyxy"]),
                    "max_score": float(detection["score"]),
                    "phrase_hits": [str(detection["query_phrase"])],
                    "model_phrases": [str(detection.get("model_phrase", ""))],
                    "members": [dict(detection)],
                }
            )
            continue
        match["members"].append(dict(detection))
        phrase = str(detection["query_phrase"])
        if phrase not in match["phrase_hits"]:
            match["phrase_hits"].append(phrase)
        model_phrase = str(detection.get("model_phrase", ""))
        if model_phrase and model_phrase not in match["model_phrases"]:
            match["model_phrases"].append(model_phrase)
        if float(detection["score"]) > float(match["max_score"]):
            match["max_score"] = float(detection["score"])
            match["bbox_xyxy"] = list(detection["bbox_xyxy"])
    for index, cluster in enumerate(clusters, 1):
        cluster["cluster_id"] = f"C{index:02d}"
        cluster["phrase_vote_count"] = len(cluster["phrase_hits"])
    return clusters


def select_cluster(clusters: Sequence[dict[str, Any]]) -> tuple[dict | None, str, bool]:
    """Select conservatively, reporting whether same-instance ambiguity remains."""
    if not clusters:
        return None, "no_detection", False
    if len(clusters) == 1:
        return dict(clusters[0]), "single_spatial_cluster", False
    ranked = sorted(
        clusters,
        key=lambda item: (
            int(item.get("phrase_vote_count", 0)),
            float(item.get("max_score", 0.0)),
        ),
        reverse=True,
    )
    top, runner_up = ranked[0], ranked[1]
    top_votes = int(top.get("phrase_vote_count", 0))
    runner_votes = int(runner_up.get("phrase_vote_count", 0))
    ambiguous = top_votes == runner_votes
    strategy = "phrase_agreement_then_confidence"
    if ambiguous:
        strategy += "_ambiguous"
    return dict(top), strategy, ambiguous


def link_clusters_across_frames(
    frame_clusters: Iterable[tuple[int, Sequence[dict[str, Any]]]],
    link_iou: float = 0.3,
) -> list[dict[str, Any]]:
    """Link per-frame clusters into cross-frame instance tracks by box continuity.

    ``frame_clusters`` holds ``(frame_index, clusters)`` pairs in any order.  A
    cluster joins the track whose latest member box has the highest IoU >=
    ``link_iou`` and starts a new track when nothing matches.  A track holds at
    most one cluster per frame, so same-frame duplicates never merge into one
    instance.  Single-frame behavior above is unchanged; this only aggregates.
    """
    tracks: list[dict[str, Any]] = []
    ordered = sorted(frame_clusters, key=lambda item: int(item[0]))
    for raw_frame_index, clusters in ordered:
        frame_index = int(raw_frame_index)
        ranked = sorted(
            clusters, key=lambda item: float(item.get("max_score", 0.0)), reverse=True
        )
        for cluster in ranked:
            best_track = None
            best_iou = float(link_iou)
            for track in tracks:
                if track["frame_indices"][-1] == frame_index:
                    continue
                iou = box_iou(
                    track["members"][-1][1]["bbox_xyxy"], cluster["bbox_xyxy"]
                )
                if iou >= best_iou:
                    best_track = track
                    best_iou = iou
            if best_track is None:
                tracks.append(
                    {
                        "members": [(frame_index, dict(cluster))],
                        "frame_indices": [frame_index],
                    }
                )
                continue
            best_track["members"].append((frame_index, dict(cluster)))
            best_track["frame_indices"].append(frame_index)
    for index, track in enumerate(tracks, 1):
        member_clusters = [cluster for _, cluster in track["members"]]
        track["track_id"] = f"T{index:02d}"
        track["support_frame_count"] = len(track["frame_indices"])
        track["mean_max_score"] = round(
            sum(float(cluster.get("max_score", 0.0)) for cluster in member_clusters)
            / len(member_clusters),
            6,
        )
        track["phrase_vote_total"] = int(
            sum(int(cluster.get("phrase_vote_count", 0)) for cluster in member_clusters)
        )
        track["member_by_frame"] = {
            frame_index: cluster for frame_index, cluster in track["members"]
        }
        track["representative_cluster"] = dict(
            max(
                member_clusters,
                key=lambda cluster: (
                    float(cluster.get("max_score", 0.0)),
                    int(cluster.get("phrase_vote_count", 0)),
                ),
            )
        )
    return tracks


def select_cluster_track(
    tracks: Sequence[dict[str, Any]],
    min_support_frames: int = 2,
    ambiguity_tolerance: float = 0.05,
) -> tuple[dict | None, str, bool]:
    """Select the instance track with the strongest cross-frame support.

    Positional continuity (support frames) ranks above single-frame confidence,
    so a lone high-score detection cannot beat a consistent track.  Low
    consistency stays ambiguous; callers must not force an asset from it.
    """
    if not tracks:
        return None, "no_detection", False
    ranked = sorted(
        tracks,
        key=lambda item: (
            int(item.get("support_frame_count", 0)),
            float(item.get("mean_max_score", 0.0)),
            int(item.get("phrase_vote_total", 0)),
        ),
        reverse=True,
    )
    top = ranked[0]
    strategy = "multi_frame_track_continuity"
    ambiguous = False
    if int(top.get("support_frame_count", 0)) < int(min_support_frames):
        ambiguous = True
        strategy += "_low_consistency"
    elif len(ranked) > 1:
        runner = ranked[1]
        same_support = int(runner.get("support_frame_count", 0)) == int(
            top.get("support_frame_count", 0)
        )
        close_score = abs(
            float(runner.get("mean_max_score", 0.0))
            - float(top.get("mean_max_score", 0.0))
        ) <= float(ambiguity_tolerance)
        if same_support and close_score:
            ambiguous = True
            strategy += "_tied_tracks"
    return dict(top), strategy, ambiguous


def sample_video_frames(video_path: Path, count: int) -> tuple[list[dict[str, Any]], dict]:
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
    wanted = sorted(
        set(int(round(value)) for value in np.linspace(0, frame_count - 1, count))
    )
    frames: list[dict[str, Any]] = []
    for frame_index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if ok and image is not None:
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / fps, 6),
                    "image_bgr": image,
                }
            )
    capture.release()
    return frames, {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


class GroundingDinoBackend:
    def __init__(self, model_dir: Path, box_threshold: float, text_threshold: float):
        self.model_dir = model_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        # GroundingDINO's BertModelWarper still reads the legacy Transformers
        # helper.  Transformers 5 removed it, although inference always passes
        # head_mask=None here.  Restore that default path locally instead of
        # modifying either installed package.
        from transformers import BertModel

        if not hasattr(BertModel, "get_head_mask"):
            def _get_head_mask(
                bert_self,
                head_mask,
                num_hidden_layers,
                is_attention_chunked=False,
            ):
                if head_mask is None:
                    return [None] * int(num_hidden_layers)
                if head_mask.ndim == 1:
                    head_mask = head_mask[None, None, :, None, None]
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.ndim == 2:
                    head_mask = head_mask[:, None, :, None, None]
                head_mask = head_mask.to(dtype=bert_self.dtype)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
                return head_mask

            BertModel.get_head_mask = _get_head_mask
        # GroundingDINO calls the pre-v5 signature
        # get_extended_attention_mask(mask, shape, device).  In v5 the third
        # argument became dtype, so a CUDA device is otherwise passed to
        # Tensor.to(dtype=...), which fails during the first forward pass.
        current_extended_mask = BertModel.get_extended_attention_mask
        if not getattr(current_extended_mask, "_groundingdino_v5_compat", False):
            def _get_extended_attention_mask(
                bert_self, attention_mask, input_shape, device_or_dtype=None
            ):
                dtype = (
                    device_or_dtype
                    if hasattr(device_or_dtype, "is_floating_point")
                    else bert_self.dtype
                )
                return current_extended_mask(
                    bert_self, attention_mask, input_shape, dtype=dtype
                )

            _get_extended_attention_mask._groundingdino_v5_compat = True
            BertModel.get_extended_attention_mask = _get_extended_attention_mask
        from groundingdino.util.inference import load_model

        source_config = (
            self.model_dir
            / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        config_text = source_config.read_text(encoding="utf-8").replace(
            "'bert-base-uncased'",
            repr(str(self.model_dir / "bert-base-uncased")),
        )
        config_path = Path(tempfile.gettempdir()) / "object_gdino_swint_local_bert.py"
        config_path.write_text(config_text, encoding="utf-8")
        self.model = load_model(
            str(config_path), str(self.model_dir / "groundingdino_swint_ogc.pth")
        ).eval().to("cuda:0")

    def detect(self, image_bgr: np.ndarray, query_phrase: str) -> list[dict[str, Any]]:
        from PIL import Image
        from groundingdino.datasets import transforms as transforms
        from groundingdino.util.inference import predict

        self.load()
        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        tensor, _ = transform(pil, None)
        boxes, logits, phrases = predict(
            model=self.model,
            image=tensor,
            caption=query_phrase,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device="cuda:0",
        )
        height, width = image_bgr.shape[:2]
        rows: list[dict[str, Any]] = []
        for box, score, model_phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(value) for value in box]
            xyxy = clip_box(
                [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
                width,
                height,
            )
            rows.append(
                {
                    "query_phrase": query_phrase,
                    "model_phrase": str(model_phrase),
                    "score": round(float(score), 6),
                    "bbox_xyxy": xyxy,
                }
            )
        return rows


class Sam2FallbackBackend:
    def __init__(self, repo_path: Path, checkpoint_path: Path):
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.predictor = None

    def load(self) -> None:
        if self.predictor is not None:
            return
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            "sam2_hiera_b+.yaml",
            str(self.checkpoint_path),
            device="cuda:0",
        )
        self.predictor = SAM2ImagePredictor(model)

    def segment(
        self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]
    ) -> tuple[np.ndarray, float]:
        self.load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(bbox_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        if len(masks) == 0:
            return np.zeros(image_bgr.shape[:2], dtype=bool), 0.0
        best = int(np.argmax(scores))
        return np.asarray(masks[best]).astype(bool), float(scores[best])


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_assets(
    output_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    box: Sequence[int],
    mask: np.ndarray,
) -> dict[str, str]:
    frame_dir = output_dir / "frames"
    mask_dir = output_dir / "masks"
    crop_dir = output_dir / "crops"
    rgba_dir = output_dir / "rgba"
    overlay_dir = output_dir / "overlays"
    for directory in (frame_dir, mask_dir, crop_dir, rgba_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    x0, y0, x1, y1 = clip_box(box, image_bgr.shape[1], image_bgr.shape[0])
    frame_path = frame_dir / f"{stem}.jpg"
    mask_path = mask_dir / f"{stem}.png"
    crop_path = crop_dir / f"{stem}.jpg"
    rgba_path = rgba_dir / f"{stem}.png"
    overlay_path = overlay_dir / f"{stem}.jpg"

    mask_u8 = mask.astype(np.uint8) * 255
    crop = image_bgr[y0:y1, x0:x1]
    crop_mask = mask_u8[y0:y1, x0:x1]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask
    overlay = image_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[mask] = cv2.addWeighted(overlay, 0.45, tint, 0.55, 0)[mask]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 4)

    cv2.imwrite(str(frame_path), image_bgr)
    cv2.imwrite(str(mask_path), mask_u8)
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(rgba_path), rgba)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "frame_path": str(frame_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "crop_path": str(crop_path.resolve()),
        "rgba_path": str(rgba_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-shot", type=int, default=5)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--cluster-iou", type=float, default=0.75)
    parser.add_argument("--dino-model-dir", type=Path, default=DEFAULT_DINO_DIR)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument(
        "--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT
    )
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.candidates,
        args.shots,
        args.dino_model_dir / "groundingdino_swint_ogc.pth",
        args.sam2_checkpoint,
        args.sam2_repo / "sam2/build_sam.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.samples_per_shot < 1:
        raise ValueError("--samples-per-shot must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(args.candidates)
    shot_rows = read_jsonl(args.shots)
    shots = {
        normalized_shot_id(row.get("shot_id")): row
        for row in shot_rows
    }
    dino = GroundingDinoBackend(
        args.dino_model_dir, args.box_threshold, args.text_threshold
    )
    sam = Sam2FallbackBackend(args.sam2_repo, args.sam2_checkpoint)
    observations: list[dict[str, Any]] = []
    run_started = time.time()

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "UNKNOWN"))
        phrases = unique_phrases(candidate)
        expected = candidate.get("expected_shots") or [candidate.get("shot_id")]
        for raw_shot_id in expected:
            shot_id = normalized_shot_id(raw_shot_id)
            shot = shots.get(shot_id)
            if shot is None:
                observations.append(
                    {
                        "schema_version": "object-observation-v1",
                        "candidate_id": candidate_id,
                        "shot_id": shot_id,
                        "status": "shot_missing",
                    }
                )
                continue
            video_path = Path(str(shot.get("clip_path") or shot.get("video_path")))
            frames, metadata = sample_video_frames(video_path, args.samples_per_shot)
            for frame in frames:
                raw_detections: list[dict[str, Any]] = []
                for phrase in phrases:
                    raw_detections.extend(dino.detect(frame["image_bgr"], phrase))
                clusters = cluster_detections(raw_detections, args.cluster_iou)
                selected, strategy, ambiguous = select_cluster(clusters)
                base = {
                    "schema_version": "object-observation-v1",
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
                    "grounding_phrases": phrases,
                    "all_grounding_detections": raw_detections,
                    "grounding_clusters": clusters,
                    "selection_strategy": strategy,
                    "ambiguous_instance": ambiguous,
                    "segmentation_backend": "sam2_fallback",
                }
                if selected is None:
                    base.update(
                        {
                            "status": "no_detection",
                            "selected_cluster_id": None,
                            "selected_bbox_xyxy": None,
                            "mask_bbox_xyxy": None,
                            "mask_area_pixels": 0,
                            "mask_area_ratio": 0.0,
                            "sam_score": None,
                            "assets": {},
                        }
                    )
                    observations.append(base)
                    continue
                mask, sam_score = sam.segment(
                    frame["image_bgr"], selected["bbox_xyxy"]
                )
                stem = (
                    f"{candidate_id}_{shot_id}_F{int(frame['frame_index']):06d}"
                )
                assets = save_assets(
                    args.output,
                    stem,
                    frame["image_bgr"],
                    selected["bbox_xyxy"],
                    mask,
                )
                area = int(mask.sum())
                base.update(
                    {
                        "status": "ok" if area else "no_mask",
                        "selected_cluster_id": selected["cluster_id"],
                        "selected_bbox_xyxy": selected["bbox_xyxy"],
                        "selected_grounding_score": selected["max_score"],
                        "selected_phrase_vote_count": selected["phrase_vote_count"],
                        "mask_bbox_xyxy": mask_bbox(mask),
                        "mask_area_pixels": area,
                        "mask_area_ratio": round(area / mask.size, 8),
                        "sam_score": round(sam_score, 6),
                        "assets": assets,
                    }
                )
                observations.append(base)

    write_jsonl(args.output / "object_observations.jsonl", observations)
    successful = [row for row in observations if row.get("status") == "ok"]
    summary = {
        "schema_version": "object-grounding-segmentation-run-v1",
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
        "grounding_backend": "GroundingDINO_SwinT_OGC",
        "segmentation_backend": "sam2_fallback",
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "cluster_iou": args.cluster_iou,
        "elapsed_seconds": round(time.time() - run_started, 3),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
