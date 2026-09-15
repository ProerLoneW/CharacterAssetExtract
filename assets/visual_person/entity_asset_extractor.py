#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
entity_asset_extractor.py

Single-frame subject observation pipeline (重构说明.md).

This stage answers ONE question only:

    "In this frame, which independent visual subjects did I observe?"

It deliberately does NOT answer "who is this person across the video".
Cross-frame identity association (face embeddings, body ReID, CHR ids)
is a later stage and is not called here.

Pipeline per sampled frame:

    frame
      -> Sa2VA "all the main characters" mask
      -> connected components -> OBS00/OBS01/... (frame-local)
      -> GroundingDINO("person") keeps ALL confident boxes
      -> difficult frames (low coverage / conflicting boxes / faceless
         boxes) get a bounded local zoom-in re-detection, projected back
         to full-frame coordinates (V4; --disable-local-redetect)
      -> each OBS binds its best-matching box (mask overlap + IoU)
      -> SAM 3.1 refines the matched box into an instance mask
      -> face detector + face-to-body binding -> at most ONE best face
      -> assets + debug overlays + manifest (identity_id = null)

Local models:
    discovery      : Sa2VA-Qwen3-VL-4B  (subject_grounding/sa2va)
    localization   : GroundingDINO SwinT (AI-ModelScope/GroundingDINO)
    face           : InsightFace buffalo_l det_10g (insightface/)
    segmentation   : SAM 3.1 multiplex   (segmentation/sam3_1)

Usage:
    python entity_asset_extractor.py \
        --video demo.mp4 --output workspace/obs_demo --num-samples 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MIN_FACELESS_PERSON_HEIGHT_WIDTH_RATIO = 0.80

# V4: difficult-frame local multi-scale re-detection.  All thresholds are
# conservative so the extra detector calls stay rare; rolling back simply
# means disabling the pass (extract_frame_observations(local_redetect=False)
# or --disable-local-redetect).
LOCAL_REDETECT_MAX_UPSCALE_SIDE = 1600    # cap zoomed-crop long side (compute)
LOCAL_REDETECT_MIN_FACE_SCORE = 0.60      # discard weak zoom-pass faces early
LOCAL_REDETECT_MERGE_IOU = 0.50          # zoom box duplicating a full-frame box
LOCAL_REDETECT_FACE_MERGE_IOU = 0.50     # zoom face duplicating a known face
LOCAL_REDETECT_ROI_EXPAND = 0.50         # context padding around evidence box
LOCAL_REDETECT_MIN_ROI_SIDE = 24         # degenerate ROIs are skipped
LOCAL_REDETECT_CONFLICT_IOU = 0.35
LOCAL_REDETECT_CONFLICT_CONTAINMENT = 0.60
LOCAL_REDETECT_HEIGHT_OVERLAP = 0.70     # shared vertical span + x overlap
LOCAL_REDETECT_MASK_RATIO = 0.10         # Sa2VA subject mass implying people
LOCAL_REDETECT_COVERED_FRACTION = 0.50   # boxes cover less of the mask than this


# ============================================================
# Data structures
# ============================================================

@dataclass
class FrameSample:
    tag: str
    percentile: float
    frame_index: int
    timestamp_sec: float
    image_path: str


@dataclass
class Observation:
    """One frame-local visual subject.  Observation != identity."""

    observation_id: str          # e.g. F0000_OBS00
    frame_tag: str
    frame_index: int
    timestamp_sec: float
    entity_type: str             # "human" only after independent person evidence

    sa2va_bbox: List[int]
    sa2va_mask_area: int

    groundingdino_bbox: Optional[List[int]]
    grounding_score: Optional[float]

    body_crop_path: Optional[str] = None
    mask_path: Optional[str] = None
    masked_crop_path: Optional[str] = None
    rgba_crop_path: Optional[str] = None
    overlay_path: Optional[str] = None

    face: Optional[Dict] = None
    notes: List[str] = None

    # current stage never resolves identity
    identity_id: Optional[str] = None


# ============================================================
# MODEL-SPECIFIC IMPLEMENTATIONS (local models)
# ============================================================

# 模型权重保留在 v2 仓库根目录 local_models/（本文件在 assets/visual_person/ 下，上三级即 v2 根）。
_MODEL_ROOT = str(Path(__file__).resolve().parents[3] / "local_models")


def load_entity_model() -> Any:
    """Sa2VA referring segmentation, used for main-character discovery."""
    import sys

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from global_roster_v3.sa2va import Sa2VASegmenter

    model = Sa2VASegmenter(
        os.path.join(_MODEL_ROOT, "subject_grounding", "sa2va",
                     "Sa2VA-Qwen3-VL-4B"),
        device="cuda:0",
    )
    model.load()
    return model


def load_grounding_dino() -> Any:
    # Same transformers-5 compatibility shims as
    # object_grounding_segmenter_v1.GroundingDinoBackend.load(): GroundingDINO's
    # BertModelWarper still reads BertModel.get_head_mask (removed in
    # transformers 5; inference passes head_mask=None anyway), and calls
    # get_extended_attention_mask with the pre-v5 positional signature.
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

    model_dir = os.path.join(_MODEL_ROOT, "AI-ModelScope", "GroundingDINO")
    src_cfg = os.path.join(model_dir, "groundingdino", "config",
                           "GroundingDINO_SwinT_OGC.py")
    with open(src_cfg, "r", encoding="utf-8") as handle:
        cfg_text = handle.read()
    # point the text encoder at the local bert copy (offline)
    cfg_text = cfg_text.replace(
        "'bert-base-uncased'",
        repr(os.path.join(model_dir, "bert-base-uncased")),
    )
    tmp_cfg = os.path.join(tempfile.gettempdir(),
                           "gdino_swint_local_bert.py")
    with open(tmp_cfg, "w", encoding="utf-8") as handle:
        handle.write(cfg_text)
    model = load_model(
        tmp_cfg,
        os.path.join(model_dir, "groundingdino_swint_ogc.pth"),
    )
    return model.eval().to("cuda:0")


def grounding_dino_detect(
    model: Any,
    image_bgr: np.ndarray,
    text_prompt: str,
    roi: Optional[Sequence[float]] = None,
    upscale: float = 1.0,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """All confident boxes for the prompt: [(bbox_xyxy, score), ...].

    With ``roi`` ((x1, y1, x2, y2) in full-frame coordinates) detection runs
    on the cropped region magnified by ``upscale`` and every box is
    projected back into full-frame coordinates; this backs the V4
    difficult-frame local re-detection pass.
    """
    from PIL import Image
    from groundingdino.datasets import transforms as T
    from groundingdino.util.inference import predict

    if roi is not None:
        work, offset, scale = _prepare_roi(image_bgr, roi, upscale)
    else:
        work, offset, scale = image_bgr, (0, 0), 1.0
    if work.size == 0:
        return []

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
    image_tensor, _ = transform(pil, None)

    boxes, logits, _phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=0.25,
        text_threshold=0.20,
        device="cuda:0",
    )

    h, w = work.shape[:2]
    out: List[Tuple[Tuple[float, float, float, float], float]] = []
    for box, score in zip(boxes, logits):
        cx, cy, bw, bh = [float(v) for v in box]
        work_box = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h)
        out.append((_project_roi_box(work_box, offset, scale), float(score)))
    return out


def load_face_detector() -> Any:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=os.path.join(_MODEL_ROOT, "insightface"),
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detect_faces(
    model: Any,
    image_bgr: np.ndarray,
    roi: Optional[Sequence[float]] = None,
    upscale: float = 1.0,
) -> List[Dict]:
    """All faces with bbox + det_score (full-frame coordinates).

    With ``roi`` ((x1, y1, x2, y2) in full-frame coordinates) the detector
    runs on the cropped region magnified by ``upscale`` and boxes are
    projected back into full-frame coordinates; this backs the V4
    difficult-frame local re-detection pass.
    """
    if roi is not None:
        work, offset, scale = _prepare_roi(image_bgr, roi, upscale)
    else:
        work, offset, scale = image_bgr, (0, 0), 1.0
    if work.size == 0:
        return []
    out: List[Dict] = []
    for face in model.get(work):
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        out.append({
            "bbox": _project_roi_box((x1, y1, x2, y2), offset, scale),
            "det_score": float(getattr(face, "det_score", 0.5)),
        })
    return out


def load_sam() -> Any:
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam_dir = os.path.join(_MODEL_ROOT, "segmentation", "sam3_1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(
        bpe_path=os.path.join(sam_dir, "bpe_simple_vocab_16e6.txt.gz"),
        checkpoint_path=os.path.join(sam_dir, "sam3.1_multiplex.pt"),
        device="cuda",
    )
    return Sam3Processor(model, confidence_threshold=0.5)


def sam_segment(
    model: Any,
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    state: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Segment one instance using the matched box as a SAM 3.1 prompt.

    Returns (full-frame bool mask, reusable state).  add_geometric_prompt
    returns ALL confident objects in the frame, so we must pick the one
    whose box best matches our prompt instead of taking index 0.
    """
    import torch
    from PIL import Image
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox

    h, w = image_bgr.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_xywh = torch.tensor([[x1, y1, x2 - x1, y2 - y1]])
    box_cxcywh = box_xywh_to_cxcywh(box_xywh)
    norm_box = normalize_bbox(box_cxcywh, w, h).flatten().tolist()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if state is None or "backbone_out" not in state:
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            state = model.set_image(pil, state)
        else:
            model.reset_all_prompts(state)
        state = model.add_geometric_prompt(state=state, box=norm_box,
                                           label=True)

    empty = np.zeros((h, w), dtype=bool)
    masks = state.get("masks")
    boxes = state.get("boxes")
    if masks is None or len(masks) == 0 or boxes is None or len(boxes) == 0:
        return empty, state

    boxes_np = boxes.float().cpu().numpy()
    best_i, best_iou = -1, 0.0
    for i in range(len(boxes_np)):
        iou = box_iou(bbox_xyxy, boxes_np[i])
        if iou > best_iou:
            best_iou, best_i = iou, i
    if best_i < 0 or best_iou < 0.3:
        # SAM did not return anything matching our prompt
        return empty, state

    mask = masks[best_i].squeeze(0).cpu().numpy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    return mask > 0, state


# ============================================================
# Generic utilities
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width - 1, int(math.floor(x1))))
    y1 = max(0, min(height - 1, int(math.floor(y1))))
    x2 = max(1, min(width, int(math.ceil(x2))))
    y2 = max(1, min(height, int(math.ceil(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_box(
    box: Sequence[int],
    width: int,
    height: int,
    ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    return clip_box(
        (x1 - bw * ratio, y1 - bh * ratio,
         x2 + bw * ratio, y2 + bh * ratio),
        width,
        height,
    )


def _prepare_roi(
    image_bgr: np.ndarray,
    roi: Sequence[float],
    upscale: float,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Crop an ROI and magnify it for a local detection pass.

    Returns (work_image, (offset_x, offset_y), scale) where ``scale`` is the
    magnification actually applied — capped so the long side stays within
    LOCAL_REDETECT_MAX_UPSCALE_SIDE, and never below 1 (no downscaling).
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clip_box(roi, w, h)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return crop, (x1, y1), 1.0
    long_side = max(x2 - x1, y2 - y1, 1)
    scale = min(float(upscale), LOCAL_REDETECT_MAX_UPSCALE_SIDE / long_side)
    if scale <= 1.0:
        return crop, (x1, y1), 1.0
    resized = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return resized, (x1, y1), scale


def _project_roi_box(
    box: Sequence[float],
    offset_xy: Tuple[int, int],
    scale: float,
) -> Tuple[float, float, float, float]:
    """Map a box from magnified-ROI coordinates back to full-frame ones."""
    ox, oy = offset_xy
    return (ox + float(box[0]) / scale, oy + float(box[1]) / scale,
            ox + float(box[2]) / scale, oy + float(box[3]) / scale)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def box_area(b: Sequence[float]) -> float:
    return max(float(b[2]) - float(b[0]), 0.0) * max(float(b[3]) - float(b[1]), 0.0)


def box_inside_fraction(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of `inner` area that lies inside `outer`."""
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(box_area(inner), 1.0)


def normalize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    return mask > 0


def make_masked_crop(image_bgr, mask, crop_box) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    out = np.zeros_like(crop)
    out[crop_mask] = crop[crop_mask]
    return out


def make_rgba_crop(image_bgr, mask, crop_box) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = crop_mask.astype(np.uint8) * 255
    return bgra


# ============================================================
# Video frame sampling
# ============================================================

DEFAULT_PERCENTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)


def get_video_metadata(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0:
        raise RuntimeError("Invalid frame count.")

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
    }


def percentile_to_index(percentile: float, frame_count: int) -> int:
    return max(0, min(frame_count - 1,
                      int(round(percentile * (frame_count - 1)))))


def sample_frames(
    video_path: Path,
    output_root: Path,
    percentiles: Sequence[Tuple[str, float]],
) -> Tuple[List[FrameSample], Dict[str, np.ndarray], Dict[str, float]]:
    metadata = get_video_metadata(video_path)
    frame_count = int(metadata["frame_count"])
    fps = float(metadata["fps"])

    frames_dir = ensure_dir(output_root / "frames")
    cap = cv2.VideoCapture(str(video_path))

    samples: List[FrameSample] = []
    images: Dict[str, np.ndarray] = {}

    try:
        for tag, p in percentiles:
            idx = percentile_to_index(p, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode frame {idx}")
            path = frames_dir / f"{tag}.jpg"
            cv2.imwrite(str(path), frame)
            samples.append(FrameSample(
                tag=tag,
                percentile=p,
                frame_index=idx,
                timestamp_sec=(idx / fps if fps > 0 else 0.0),
                image_path=str(path),
            ))
            images[tag] = frame
    finally:
        cap.release()

    return samples, images, metadata


# ============================================================
# Per-frame observation logic (no cross-frame identity)
# ============================================================

def components_from_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.005,
) -> List[Dict]:
    """Split one Sa2VA binary mask into candidate subject components."""
    out: List[Dict] = []
    if mask.size == 0:
        return out
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    h, w = mask.shape
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area_ratio * h * w:
            continue
        comp_mask = (labels == i)
        out.append({
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "area": int(area),
            "mask": comp_mask,
        })
    return out


def match_detection_to_observation(
    detections: Sequence[Tuple[Tuple[float, float, float, float], float]],
    comp_bbox: Sequence[int],
    comp_mask: np.ndarray,
) -> Optional[Tuple[Tuple[float, float, float, float], float, Dict]]:
    """Bind the best GroundingDINO box to one Sa2VA observation.

    All confident boxes arrive here (no top-1 before binding); matching
    uses mask coverage first, bbox IoU second.  Each detection may only
    be consumed once per frame (caller enforces).
    """
    comp_area = max(int(comp_mask.sum()), 1)
    best = None
    for det_box, score in detections:
        x0 = max(comp_bbox[0], int(det_box[0]))
        y0 = max(comp_bbox[1], int(det_box[1]))
        x1 = min(comp_bbox[2], int(det_box[2]))
        y1 = min(comp_bbox[3], int(det_box[3]))
        mask_coverage = 0.0
        if x1 > x0 and y1 > y0:
            mask_coverage = float(comp_mask[y0:y1, x0:x1].sum()) / comp_area
        iou = box_iou(comp_bbox, det_box)
        match_score = mask_coverage + 0.5 * iou
        if best is None or match_score > best[2]["match_score"]:
            best = (det_box, score, {
                "match_score": match_score,
                "mask_coverage": mask_coverage,
                "bbox_iou": iou,
            })
    if best is None:
        return None
    stats = best[2]
    if stats["mask_coverage"] < 0.3 and stats["bbox_iou"] < 0.2:
        return None
    return best


def bind_best_face(
    face_candidates: Sequence[Dict],
    body_bbox: Sequence[float],
    person_mask: Optional[np.ndarray],
) -> Optional[Dict]:
    """Face -> body ownership (重构说明 §10).  At most one face survives.

    Conditions: center inside body bbox, center inside refined mask (if
    available), center in the upper body half.  Ties resolved by
    det_score + mask containment + relative face area + upper position.
    """
    bx1, by1, bx2, by2 = [float(v) for v in body_bbox]
    bw = max(bx2 - bx1, 1.0)
    bh = max(by2 - by1, 1.0)
    body_area = bw * bh

    scored: List[Tuple[float, Dict]] = []
    for cand in face_candidates:
        fx1, fy1, fx2, fy2 = cand["bbox"]
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
            continue                       # condition 1: center in body bbox
        mask_containment = 0.0
        if person_mask is not None and person_mask.size:
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= iy < person_mask.shape[0] and \
                    0 <= ix < person_mask.shape[1]:
                mask_containment = float(person_mask[iy, ix])
            if mask_containment <= 0.0:
                continue                   # condition 2: center inside mask
        upper_limit = by1 + 0.55 * bh
        if cy > upper_limit:
            continue                       # condition 3: upper body only
        face_area = max((fx2 - fx1) * (fy2 - fy1), 0.0)
        score = (cand.get("det_score", 0.5) * 0.4
                 + mask_containment * 0.3
                 + min(face_area / body_area, 0.5) * 0.4
                 + max(0.0, 1.0 - (cy - by1) / bh) * 0.1)
        scored.append((score, cand))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])
    return scored[0][1]


def overlay_components(
    image_bgr: np.ndarray,
    comps: List[Dict],
    assigned: Optional[List] = None,
    rescued: Optional[List] = None,
) -> np.ndarray:
    """Debug overlay: component bboxes + OBS ids (+ matched GDINO boxes)."""
    out = image_bgr.copy()
    colors = [(0, 200, 255), (0, 255, 128), (255, 128, 0),
              (128, 0, 255), (0, 128, 255), (255, 0, 128)]
    for i, comp in enumerate(comps):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = comp["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"OBS{i:02d}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if assigned and i < len(assigned) and assigned[i] is not None:
            det_box, score, _ = assigned[i]
            dx1, dy1, dx2, dy2 = [int(v) for v in det_box]
            cv2.rectangle(out, (dx1, dy1), (dx2, dy2), (255, 255, 255), 2)
            cv2.putText(out, f"gdino {score:.2f}",
                        (dx1, max(40, dy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if rescued:
        for r, (det_box, score) in enumerate(rescued):
            obs_i = len(comps) + r
            rx1, ry1, rx2, ry2 = [int(v) for v in det_box]
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
            cv2.putText(out, f"OBS{obs_i:02d} rescued {score:.2f}",
                        (rx1, max(20, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def _box_has_face_evidence(
    box: Sequence[float],
    face_candidates: Sequence[Dict],
    mask: Optional[np.ndarray] = None,
    min_det_score: float = 0.75,
) -> bool:
    """High-confidence face center inside the box's upper region."""
    x1, y1, x2, y2 = box
    height = max(y2 - y1, 1)
    for face in face_candidates:
        if float(face.get("det_score", 0.0)) < min_det_score:
            continue
        fx1, fy1, fx2, fy2 = face["bbox"]
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        if not (x1 <= cx <= x2 and y1 <= cy <= y1 + 0.65 * height):
            continue
        if mask is not None:
            ix, iy = int(round(cx)), int(round(cy))
            if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]
                    and bool(mask[iy, ix])):
                continue
        return True
    return False


def _boxes_conflict(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two person boxes that likely describe entangled instances."""
    if box_iou(a, b) >= LOCAL_REDETECT_CONFLICT_IOU:
        return True
    if (box_inside_fraction(a, b) >= LOCAL_REDETECT_CONFLICT_CONTAINMENT
            or box_inside_fraction(b, a) >= LOCAL_REDETECT_CONFLICT_CONTAINMENT):
        return True
    # Side-by-side people sharing most of their vertical span and some
    # horizontal strip are often boxed as one merged or split instance.
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    min_height = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    return (y_overlap / min_height >= LOCAL_REDETECT_HEIGHT_OVERLAP
            and x_overlap > 0)


def _mask_coverage_by_boxes(
    mask_full: np.ndarray,
    boxes: Sequence[Sequence[float]],
) -> float:
    """Fraction of mask pixels covered by at least one detection box."""
    total = int(mask_full.sum())
    if total <= 0:
        return 1.0
    h, w = mask_full.shape
    covered = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, w, h)
        covered[y1:y2, x1:x2] = True
    return float(int((covered & mask_full).sum()) / total)


def plan_local_redetect_rois(
    detections: Sequence[Tuple[Tuple[float, float, float, float], float]],
    face_candidates: Sequence[Dict],
    comps: Sequence[Dict],
    mask_full: np.ndarray,
    frame_shape: Tuple[int, int],
    max_rois: int = 3,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Pick zoom-in ROIs for a difficult frame (V4 local re-detection).

    A frame is difficult when confident detections conflict (heavy overlap,
    nesting, or side-by-side boxes sharing most of their vertical span),
    when a confident person box has no face bound to it, or when the Sa2VA
    subject mask is large but detection boxes cover little of it.  Returns
    up to ``max_rois`` clipped ROIs in full-frame coordinates plus the
    triggering reason labels; an empty ROI list means the frame is fine and
    no extra detector calls should happen.
    """
    h, w = frame_shape
    candidates: List[Tuple[Tuple[int, int, int, int], str]] = []

    # 1. Conflicting detection pairs: zoom in to separate merged instances.
    conflicts: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            box_a, score_a = detections[i]
            box_b, score_b = detections[j]
            if not _boxes_conflict(box_a, box_b):
                continue
            union = (min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                     max(box_a[2], box_b[2]), max(box_a[3], box_b[3]))
            conflicts.append((max(score_a, score_b),
                              expand_box(clip_box(union, w, h), w, h,
                                         LOCAL_REDETECT_ROI_EXPAND)))
    for _score, roi in sorted(conflicts, key=lambda item: -item[0]):
        candidates.append((roi, "detection_conflict"))

    # 2. Confident person box without any bound face: zoom in for small or
    #    profile faces that the full-frame pass could not see.
    unbound = sorted(
        ((box, score) for box, score in detections
         if not _box_has_face_evidence(box, face_candidates)),
        key=lambda item: -item[1])
    for box, _score in unbound:
        candidates.append(
            (expand_box(clip_box(box, w, h), w, h, LOCAL_REDETECT_ROI_EXPAND),
             "unbound_face_box"))

    # 3. Low detection coverage: Sa2VA sees subject mass that GroundingDINO
    #    did not box (distant / occluded people).
    mask_ratio = float(mask_full.mean()) if mask_full.size else 0.0
    if mask_ratio >= LOCAL_REDETECT_MASK_RATIO:
        covered = _mask_coverage_by_boxes(mask_full,
                                          [box for box, _ in detections])
        if covered < LOCAL_REDETECT_COVERED_FRACTION:
            unmatched = sorted(
                (comp for comp in comps
                 if not any(box_iou(comp["bbox"], box) >= 0.30
                            or box_inside_fraction(comp["bbox"], box) >= 0.50
                            for box, _ in detections)),
                key=lambda comp: -comp["area"])
            for comp in unmatched:
                candidates.append(
                    (expand_box(clip_box(comp["bbox"], w, h), w, h,
                                LOCAL_REDETECT_ROI_EXPAND),
                     "low_detection_coverage"))

    # Conflict resolution first, then faceless boxes, then coverage gaps;
    # drop ROIs already covered by a chosen one and cap the per-frame budget.
    chosen: List[Tuple[Tuple[int, int, int, int], str]] = []
    for roi, kind in candidates:
        if len(chosen) >= max(0, int(max_rois)):
            break
        if (roi[2] - roi[0] < LOCAL_REDETECT_MIN_ROI_SIDE
                or roi[3] - roi[1] < LOCAL_REDETECT_MIN_ROI_SIDE):
            continue
        if any(box_inside_fraction(roi, kept) >= 0.80
               or box_iou(roi, kept) >= 0.60 for kept, _ in chosen):
            continue
        chosen.append((roi, kind))
    reasons = list(dict.fromkeys(kind for _roi, kind in chosen))
    return [roi for roi, _kind in chosen], reasons


def extract_frame_observations(
    frame_idx: int,
    sample: FrameSample,
    image_bgr: np.ndarray,
    sa2va: Any,
    gdino: Any,
    face_app: Any,
    sam: Any,
    use_sam: bool,
    output_root: Path,
    crop_padding: float = 0.05,
    local_redetect: bool = True,
    max_local_rois: int = 3,
    local_upscale: float = 2.0,
) -> Tuple[List[Observation], Dict[str, str]]:
    """All observations for ONE frame.  No cross-frame state.

    ``local_redetect`` enables the V4 difficult-frame pass: up to
    ``max_local_rois`` zoom-in ROIs are re-run through GroundingDINO /
    InsightFace at ``local_upscale`` magnification and projected back to
    full-frame coordinates.  Set it to False to roll back to the original
    single-scale behavior.
    """
    from PIL import Image

    h, w = image_bgr.shape[:2]
    frame_dir = ensure_dir(
        output_root / "observations" / f"frame_{frame_idx:04d}"
    )
    debug_dir = ensure_dir(frame_dir / "debug")
    debug_paths: Dict[str, str] = {}

    # -- Sa2VA discovery ------------------------------------------------
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask = sa2va.segment_frame(pil)
    mask_full = mask if mask.size else np.zeros((h, w), dtype=bool)
    mask_path = debug_dir / "sa2va_mask.png"
    cv2.imwrite(str(mask_path), mask_full.astype(np.uint8) * 255)

    comps = components_from_mask(mask_full)
    # An empty Sa2VA proposal set must not short-circuit person detection:
    # GroundingDINO can still rescue people that the subject segmenter missed.
    debug_paths["sa2va_mask"] = str(mask_path)

    # -- GroundingDINO: keep ALL confident person boxes ------------------
    detections = grounding_dino_detect(gdino, image_bgr, "person")

    # -- full-frame face detection once ----------------------------------
    face_candidates = detect_faces(face_app, image_bgr) if face_app else []

    # -- V4: local multi-scale re-detection on difficult frames only -----
    # Distant, small or occluded people are missed at full-frame scale, so
    # difficult frames (low detection coverage / conflicting boxes / person
    # boxes without a bound face) get a bounded number of zoom-in ROIs.
    # Every ROI detection is projected back to full-frame coordinates and
    # rejoins the normal binding/rescue/SAM flow below.
    local_boxes: List[Tuple[float, float, float, float]] = []
    local_face_boxes: List[Tuple[float, float, float, float]] = []
    redetect_reasons: List[str] = []
    if local_redetect and (gdino is not None or face_app is not None):
        rois, redetect_reasons = plan_local_redetect_rois(
            detections, face_candidates, comps, mask_full, (h, w),
            max_rois=max_local_rois)
        if rois:
            roi_overlay = image_bgr.copy()
            for roi_index, roi in enumerate(rois):
                cv2.rectangle(roi_overlay, (roi[0], roi[1]), (roi[2], roi[3]),
                              (255, 0, 255), 2)
                cv2.putText(roi_overlay, f"ROI{roi_index}",
                            (roi[0], max(20, roi[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                if gdino is not None:
                    for box, score in grounding_dino_detect(
                            gdino, image_bgr, "person", roi=roi,
                            upscale=local_upscale):
                        # Skip zoom boxes duplicating an existing detection.
                        if any(box_iou(box, kept_box) >= LOCAL_REDETECT_MERGE_IOU
                               for kept_box, _ in detections):
                            continue
                        detections.append((box, score))
                        local_boxes.append(box)
                        bx1, by1, bx2, by2 = [int(v) for v in box]
                        cv2.rectangle(roi_overlay, (bx1, by1), (bx2, by2),
                                      (255, 255, 0), 2)
                if face_app is not None:
                    for face in detect_faces(face_app, image_bgr, roi=roi,
                                             upscale=local_upscale):
                        if face["det_score"] < LOCAL_REDETECT_MIN_FACE_SCORE:
                            continue
                        if any(box_iou(face["bbox"], kept["bbox"])
                               >= LOCAL_REDETECT_FACE_MERGE_IOU
                               for kept in face_candidates):
                            continue
                        face_candidates.append(face)
                        local_face_boxes.append(face["bbox"])
            roi_overlay_path = debug_dir / "local_redetect_overlay.jpg"
            cv2.imwrite(str(roi_overlay_path), roi_overlay)
            debug_paths["local_redetect_overlay"] = str(roi_overlay_path)

    # -- bind each OBS to its best detection (consumed at most once) -----
    assigned: List[Optional[Tuple]] = [None] * len(comps)
    used = set()
    order = sorted(range(len(comps)), key=lambda i: -comps[i]["area"])
    for i in order:
        pool = [d for j, d in enumerate(detections) if j not in used]
        match = match_detection_to_observation(
            pool, comps[i]["bbox"], comps[i]["mask"]
        )
        if match is not None:
            assigned[i] = match
            used.add(detections.index(match[:2]))

    # -- rescue: unclaimed GDINO boxes become their own OBS --------------
    # Sa2VA components often merge touching people (dinner table!), so any
    # confident box nobody claimed recovers a missed/merged subject.
    # Boxes that mostly overlap a claimed box are nested duplicates.
    rescued: List[Tuple] = []
    for j, det in enumerate(detections):
        if j in used:
            continue
        det_box, score = det
        dup = any(box_iou(det_box, assigned[i][0]) > 0.5
                  for i in range(len(comps)) if assigned[i] is not None)
        if not dup:
            rescued.append(det)

    comp_overlay = overlay_components(image_bgr, comps, assigned, rescued)
    comp_overlay_path = frame_dir / "components_overlay.jpg"
    cv2.imwrite(str(comp_overlay_path), comp_overlay)
    debug_paths["components_overlay"] = str(comp_overlay_path)

    # Sa2VA is a class-agnostic subject proposer.  An unmatched component may
    # be a hand, towel, animal or any other salient/moving object, so it must
    # not be labelled as human merely because it can be segmented.  Keep it
    # only when a reliable face supplies independent human evidence.
    def box_has_face(box: Sequence[float], mask: Optional[np.ndarray] = None) -> bool:
        return _box_has_face_evidence(box, face_candidates, mask)

    def plausible_faceless_person(box: Sequence[float]) -> bool:
        x1, y1, x2, y2 = box
        width = max(float(x2) - float(x1), 1.0)
        height = max(float(y2) - float(y1), 1.0)
        # Face-backed close-ups bypass this gate.  For faceless detections,
        # reject the horizontal shapes produced by hands, towels and merged
        # foreground objects that open-set detectors may call "person".
        return height / width >= MIN_FACELESS_PERSON_HEIGHT_WIDTH_RATIO

    units: List[Dict] = []
    for i, comp in enumerate(comps):
        has_face = box_has_face(comp["bbox"], comp["mask"])
        if (assigned[i] is not None
                and (has_face or plausible_faceless_person(assigned[i][0]))):
            units.append({"comp": comp, "match": assigned[i],
                          "rescued": False,
                          "human_evidence": ("groundingdino_person+face" if has_face
                                             else "groundingdino_person+body_geometry")})
        elif has_face:
            units.append({"comp": comp, "match": None, "rescued": False,
                          "human_evidence": "high_confidence_face"})
    units += [
        {"comp": None, "match": (det_box, score, None), "rescued": True,
         "human_evidence": ("groundingdino_person+face"
                            if box_has_face(det_box)
                            else "groundingdino_person+body_geometry")}
        for det_box, score in rescued
        if box_has_face(det_box) or plausible_faceless_person(det_box)
    ]

    # -- phase 1: geometry ------------------------------------------------
    for unit in units:
        match = unit["match"]
        if match is not None:
            unit["det_score"] = match[1]
            unit["body_box"] = clip_box(match[0], w, h)
        else:
            unit["det_score"] = None
            unit["body_box"] = tuple(unit["comp"]["bbox"])
        unit["extra_notes"] = []

    # -- phase 2: SAM refinement (backbone encoded once per frame) --------
    sam_state = None
    for unit in units:
        unit["mask"] = None
        if use_sam and sam is not None:
            m, sam_state = sam_segment(
                sam, image_bgr, unit["body_box"], sam_state)
            unit["mask"] = m if m.any() else None

    # -- phase 3: subtract inner OBS masks from enclosing merged OBS ------
    # A merged observation (e.g. foreground back-view person + background
    # people) must not keep the background pixels that already belong to
    # other observations.
    for i, unit in enumerate(units):
        if unit["mask"] is None:
            continue
        bx = unit["body_box"]
        inner = [j for j in range(len(units))
                 if j != i and units[j]["mask"] is not None
                 and box_inside_fraction(units[j]["body_box"], bx) >= 0.6]
        if len(inner) < 2:
            continue
        sub = np.zeros((h, w), dtype=bool)
        for j in inner:
            sub |= units[j]["mask"]
        cleaned = unit["mask"] & ~sub
        if cleaned.any():
            unit["mask"] = cleaned
            unit["extra_notes"].append(
                "mask subtracted: enclosing merged observation")
        else:
            unit["mask"] = None

    # -- phase 4: exclusive face assignment (tighter boxes first) ---------
    remaining = list(face_candidates)
    face_by_unit: List[Optional[Dict]] = [None] * len(units)
    for i in sorted(range(len(units)),
                    key=lambda k: box_area(units[k]["body_box"])):
        best = bind_best_face(remaining, units[i]["body_box"],
                              units[i]["mask"])
        if best is not None:
            face_by_unit[i] = best
            remaining.remove(best)
    if remaining:
        for ci, cand in enumerate(remaining):
            fx1, fy1, fx2, fy2 = [int(v) for v in cand["bbox"]]
            cv2.imwrite(
                str(debug_dir / f"unassigned_face_cand_{ci:02d}.jpg"),
                image_bgr[max(0, fy1):fy2, max(0, fx1):fx2])

    # -- phase 5: assets ----------------------------------------------------
    observations: List[Observation] = []
    for i, unit in enumerate(units):
        comp = unit["comp"]
        match = unit["match"]
        det_score = unit["det_score"]
        body_box = unit["body_box"]
        person_mask = unit["mask"]
        best = face_by_unit[i]

        obs_id = f"F{frame_idx:04d}_OBS{i:02d}"
        obs_dir = ensure_dir(frame_dir / f"OBS{i:02d}")
        crop_box = expand_box(body_box, w, h, crop_padding)

        body_crop = image_bgr[crop_box[1]:crop_box[3],
                              crop_box[0]:crop_box[2]].copy()
        body_crop_path = obs_dir / "body_crop.jpg"
        cv2.imwrite(str(body_crop_path), body_crop)

        mask_out = None
        masked_path = None
        rgba_path = None
        if person_mask is not None:
            mask_out = obs_dir / "mask.png"
            cv2.imwrite(str(mask_out),
                        person_mask.astype(np.uint8) * 255)
            masked_path = obs_dir / "masked.jpg"
            cv2.imwrite(str(masked_path),
                        make_masked_crop(image_bgr, person_mask, crop_box))
            rgba_path = obs_dir / "rgba.png"
            cv2.imwrite(str(rgba_path),
                        make_rgba_crop(image_bgr, person_mask, crop_box))

        # -- overlay ------------------------------------------------------
        overlay = image_bgr.copy()
        color = (0, 200, 255) if match is not None else (0, 0, 255)
        cv2.rectangle(overlay,
                      (int(body_box[0]), int(body_box[1])),
                      (int(body_box[2]), int(body_box[3])), color, 2)
        label = obs_id + (f" gdino {det_score:.2f}"
                          if det_score is not None else " sa2va fallback")
        cv2.putText(overlay, label,
                    (int(body_box[0]), max(20, int(body_box[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        overlay_path = obs_dir / "overlay.jpg"

        # -- face asset (max ONE official face per OBS, owned exclusively) --
        if best is not None:
            fx1, fy1, fx2, fy2 = [int(v) for v in best["bbox"]]
            face_path = obs_dir / "face.jpg"
            cv2.imwrite(str(face_path),
                        image_bgr[max(0, fy1):fy2, max(0, fx1):fx2])
            face_record = {
                "detected": True,
                "bbox": [fx1, fy1, fx2, fy2],
                "det_score": round(best["det_score"], 3),
                "crop_path": str(face_path),
            }
            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        else:
            face_record = {"detected": False, "bbox": None,
                           "det_score": None, "crop_path": None}
        cv2.imwrite(str(overlay_path), overlay)

        notes: List[str] = []
        if unit["rescued"]:
            notes.append("rescued from unclaimed groundingdino box; "
                         "no Sa2VA component")
        elif match is None:
            notes.append("Sa2VA component retained by high-confidence face; "
                         "no groundingdino match")
        notes.append(f'human evidence: {unit["human_evidence"]}')
        notes.extend(unit["extra_notes"])
        # V4 provenance: record when a box/face came from the zoom-in pass.
        if (match is not None and local_boxes
                and any(box_iou(match[0], local_box) >= 0.98
                        for local_box in local_boxes)):
            notes.append("groundingdino box from local redetect "
                         f"({', '.join(redetect_reasons)})")
        if (best is not None and local_face_boxes
                and any(box_iou(best["bbox"], local_face) >= 0.90
                        for local_face in local_face_boxes)):
            notes.append("face from local redetect")
        if not face_record["detected"]:
            notes.append("no owned face; body/mask assets still valid")

        observations.append(Observation(
            observation_id=obs_id,
            frame_tag=sample.tag,
            frame_index=sample.frame_index,
            timestamp_sec=sample.timestamp_sec,
            entity_type="human",
            sa2va_bbox=([int(v) for v in comp["bbox"]]
                        if comp is not None else None),
            sa2va_mask_area=comp["area"] if comp is not None else 0,
            groundingdino_bbox=(
                [int(v) for v in match[0]] if match else None),
            grounding_score=(round(det_score, 3)
                             if det_score is not None else None),
            body_crop_path=str(body_crop_path),
            mask_path=str(mask_out) if mask_out else None,
            masked_crop_path=str(masked_path) if masked_path else None,
            rgba_crop_path=str(rgba_path) if rgba_path else None,
            overlay_path=str(overlay_path),
            face=face_record,
            notes=notes,
            identity_id=None,
        ))

    return observations, debug_paths


# ============================================================
# Main pipeline
# ============================================================

def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video).expanduser().resolve()
    output_root = ensure_dir(Path(args.output).expanduser().resolve())

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if getattr(args, "num_samples", 5) == 5:
        percentiles = DEFAULT_PERCENTILES
    else:
        n = max(1, int(args.num_samples))
        step = 1.0 / n
        percentiles = tuple(
            (f"s{i + 1:02d}", min(step * (i + 0.5), 0.999))
            for i in range(n)
        )

    # Step 1: sample frames -----------------------------------------------
    samples, images, metadata = sample_frames(
        video_path=video_path,
        output_root=output_root,
        percentiles=percentiles,
    )
    print("Sampled frames:")
    for s in samples:
        print(f"  {s.tag}: frame={s.frame_index}, "
              f"time={s.timestamp_sec:.3f}s")

    if args.dry_run:
        manifest = {
            "mode": "dry_run",
            "video": str(video_path),
            "video_metadata": metadata,
            "sampled_frames": [asdict(s) for s in samples],
        }
        with open(output_root / "manifest.json", "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"\nDry-run manifest: {output_root / 'manifest.json'}")
        return

    # Step 2: load local models -------------------------------------------
    sa2va = load_entity_model()
    gdino = load_grounding_dino()
    face_app = load_face_detector()
    use_sam = not args.disable_sam
    sam = load_sam() if use_sam else None

    # Step 3: per-frame observation extraction (no cross-frame state) -----
    all_observations: List[Observation] = []
    frame_debug: Dict[str, Dict[str, str]] = {}
    for frame_idx, sample in enumerate(samples):
        print(f"\n[{sample.tag}] frame={sample.frame_index}")
        obs, debug_paths = extract_frame_observations(
            frame_idx=frame_idx,
            sample=sample,
            image_bgr=images[sample.tag],
            sa2va=sa2va,
            gdino=gdino,
            face_app=face_app,
            sam=sam,
            use_sam=use_sam,
            output_root=output_root,
            crop_padding=args.crop_padding,
            local_redetect=not args.disable_local_redetect,
            max_local_rois=args.max_local_rois,
        )
        all_observations.extend(obs)
        frame_debug[sample.tag] = debug_paths
        for observation in obs:
            has_face = bool(observation.face and observation.face["detected"])
            print(f"  {observation.observation_id}: "
                  f"gdino={observation.grounding_score}, face={has_face}")

    # Step 4: manifest -----------------------------------------------------
    manifest = {
        "mode": "single_frame_observations",
        "pipeline": [
            "sample_frames",
            "sa2va_component_discovery",
            "groundingdino_box_binding",
            "sam31_instance_refinement",
            "insightface_face_binding",
        ],
        "video": str(video_path),
        "video_metadata": metadata,
        "sampled_frames": [asdict(s) for s in samples],
        "observations": [asdict(o) for o in all_observations],
        "frame_debug": frame_debug,
        "settings": {
            "num_samples": len(samples),
            "crop_padding": args.crop_padding,
            "use_sam": use_sam,
            "local_redetect": not args.disable_local_redetect,
            "max_local_rois": args.max_local_rois,
        },
        "identity_note": (
            "Observation != identity. identity_id is null by design; "
            "cross-frame identity association is a later stage."
        ),
    }
    with open(output_root / "manifest.json", "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\nFinished.")
    print(f"Manifest: {output_root / 'manifest.json'}")
    print(f"Observations: {len(all_observations)}")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-frame subject observation extraction: "
            "Sa2VA discovery -> GroundingDINO binding -> "
            "SAM 3.1 refinement -> face binding. "
            "No cross-frame identity association."
        )
    )
    parser.add_argument("--video", required=True,
                        help="Input video path.")
    parser.add_argument("--output", default="./entity_asset_output",
                        help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of evenly spaced frames to sample. "
                             "Default 5 keeps the p05..p95 tags.")
    parser.add_argument("--crop-padding", type=float, default=0.05,
                        help="Padding ratio around the bound box for "
                             "saved crops.")
    parser.add_argument("--disable-sam", action="store_true",
                        help="Skip SAM 3.1 refinement.")
    parser.add_argument("--disable-local-redetect", action="store_true",
                        help="Disable the V4 difficult-frame local zoom-in "
                             "re-detection pass (rollback to full-frame-only "
                             "detection).")
    parser.add_argument("--max-local-rois", type=int, default=3,
                        help="Maximum zoom-in ROIs re-detected per difficult "
                             "frame when local re-detection is enabled.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only sample frames and write a skeleton "
                             "manifest. No model inference.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
