#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ground key objects with GroundingDINO and the real local SAM3.1 model.

V2 reuses V1's JSONL, frame sampling, DINO and arbitration helpers, but replaces
the temporary SAM2 fallback with the repository's SAM3.1 source and multiplex
checkpoint.  V1 outputs remain untouched so both backends can be compared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import object_grounding_segmenter_v1 as common


# local_models / third_party 保留在 v2 仓库根目录（本文件在 assets/key_object/ 下，上三级）。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM31_SOURCE = ROOT / "third_party/sam3_src"
DEFAULT_SAM31_CHECKPOINT = (
    ROOT / "local_models/segmentation/sam3_1/sam3.1_multiplex.pt"
)
DEFAULT_SAM31_BPE = (
    ROOT / "local_models/segmentation/sam3_1/bpe_simple_vocab_16e6.txt.gz"
)


class Sam31Backend:
    def __init__(self, source_path: Path, checkpoint_path: Path, bpe_path: Path):
        self.source_path = source_path
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.processor = None

    def load(self) -> None:
        if self.processor is not None:
            return
        if str(self.source_path) not in sys.path:
            sys.path.insert(0, str(self.source_path))
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = build_sam3_image_model(
            bpe_path=str(self.bpe_path),
            checkpoint_path=str(self.checkpoint_path),
            device="cuda",
        )
        self.processor = Sam3Processor(model, confidence_threshold=0.5)

    def segment(
        self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        self.load()
        # Reuse the already-tested project implementation.  It chooses the
        # SAM3.1 output box with maximum IoU to the DINO prompt box.
        # entity_asset_extractor 已迁到 assets/visual_person/，按兄弟目录解析。
        _vp_dir = str(Path(__file__).resolve().parent.parent / "visual_person")
        if _vp_dir not in sys.path:
            sys.path.insert(0, _vp_dir)
        from entity_asset_extractor import sam_segment

        mask, state = sam_segment(self.processor, image_bgr, bbox_xyxy, None)
        details: dict[str, Any] = {
            "returned_instance_count": 0,
            "matched_box_iou": None,
        }
        boxes = state.get("boxes") if isinstance(state, dict) else None
        scores = state.get("scores") if isinstance(state, dict) else None
        if boxes is None or len(boxes) == 0:
            return mask, 0.0, details
        boxes_np = boxes.float().detach().cpu().numpy()
        details["returned_instance_count"] = int(len(boxes_np))
        ious = [common.box_iou(bbox_xyxy, box) for box in boxes_np]
        best = int(np.argmax(ious))
        details["matched_box_iou"] = round(float(ious[best]), 6)
        score = 0.0
        if scores is not None and len(scores) > best:
            score = float(scores[best].detach().cpu())
        return mask, score, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-shot", type=int, default=5)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--cluster-iou", type=float, default=0.75)
    parser.add_argument(
        "--dino-model-dir", type=Path, default=common.DEFAULT_DINO_DIR
    )
    parser.add_argument("--sam31-source", type=Path, default=DEFAULT_SAM31_SOURCE)
    parser.add_argument(
        "--sam31-checkpoint", type=Path, default=DEFAULT_SAM31_CHECKPOINT
    )
    parser.add_argument("--sam31-bpe", type=Path, default=DEFAULT_SAM31_BPE)
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    required = (
        args.candidates,
        args.shots,
        args.dino_model_dir / "groundingdino_swint_ogc.pth",
        args.sam31_source / "sam3/model_builder.py",
        args.sam31_checkpoint,
        args.sam31_bpe,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.samples_per_shot < 1:
        raise ValueError("--samples-per-shot must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = common.read_jsonl(args.candidates)
    shots = {
        common.normalized_shot_id(row.get("shot_id")): row
        for row in common.read_jsonl(args.shots)
    }
    dino = common.GroundingDinoBackend(
        args.dino_model_dir, args.box_threshold, args.text_threshold
    )
    sam = Sam31Backend(
        args.sam31_source, args.sam31_checkpoint, args.sam31_bpe
    )
    observations: list[dict[str, Any]] = []
    run_started = time.time()

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "UNKNOWN"))
        phrases = common.unique_phrases(candidate)
        expected = candidate.get("expected_shots") or [candidate.get("shot_id")]
        for raw_shot_id in expected:
            shot_id = common.normalized_shot_id(raw_shot_id)
            shot = shots.get(shot_id)
            if shot is None:
                observations.append(
                    {
                        "schema_version": "object-observation-v2",
                        "candidate_id": candidate_id,
                        "shot_id": shot_id,
                        "status": "shot_missing",
                    }
                )
                continue
            video_path = Path(str(shot.get("clip_path") or shot.get("video_path")))
            frames, _ = common.sample_video_frames(video_path, args.samples_per_shot)
            for frame in frames:
                raw_detections: list[dict[str, Any]] = []
                for phrase in phrases:
                    raw_detections.extend(dino.detect(frame["image_bgr"], phrase))
                clusters = common.cluster_detections(
                    raw_detections, args.cluster_iou
                )
                selected, strategy, ambiguous = common.select_cluster(clusters)
                base = {
                    "schema_version": "object-observation-v2",
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
                    "segmentation_backend": "sam3.1_image_geometric_prompt",
                    "sam31_checkpoint": str(args.sam31_checkpoint.resolve()),
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
                            "sam_details": {},
                            "assets": {},
                        }
                    )
                    observations.append(base)
                    continue
                mask, sam_score, sam_details = sam.segment(
                    frame["image_bgr"], selected["bbox_xyxy"]
                )
                stem = (
                    f"{candidate_id}_{shot_id}_F{int(frame['frame_index']):06d}"
                )
                assets = common.save_assets(
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
                        "mask_bbox_xyxy": common.mask_bbox(mask),
                        "mask_area_pixels": area,
                        "mask_area_ratio": round(area / mask.size, 8),
                        "sam_score": round(sam_score, 6),
                        "sam_details": sam_details,
                        "assets": assets,
                    }
                )
                observations.append(base)

    common.write_jsonl(args.output / "object_observations.jsonl", observations)
    successful = [row for row in observations if row.get("status") == "ok"]
    summary = {
        "schema_version": "object-grounding-segmentation-run-v2",
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
        "segmentation_backend": "sam3.1_image_geometric_prompt",
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
