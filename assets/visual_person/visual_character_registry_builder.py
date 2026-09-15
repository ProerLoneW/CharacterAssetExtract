#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: merge shot-local TrackletSignatures into global identities.

One file, one responsibility: consume shot-local TrackletSignature JSON files
and produce a face-first, constraint-aware, reversible global identity graph.
It implements ``第三阶段_跨Shot人物身份合并_TODO_V1.md`` while deliberately
keeping uncertain face-less observations unresolved.

The output is intentionally graph based: original tracklets and merge edges
remain intact, so later rollback/split work can rebuild clusters safely.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


SCHEMA_VERSION = "cross-shot-identity-v2"
DETECTION_COOCCURRENCE = "detection_cooccurrence"
LEGACY_INTERVAL_OVERLAP = "legacy_interval_overlap"
DECISION_MUST = "MUST_LINK"
DECISION_POSSIBLE = "POSSIBLE_LINK"
DECISION_CANNOT = "CANNOT_LINK"
DECISION_NONE = "NO_EVIDENCE"
# Face topk_mean below this value between two tracklets that both carry
# high-quality faces is treated as positive evidence *against* identity.
CONTRADICTION_FACE_MEAN = 0.25


@dataclass
class FaceFeature:
    asset_path: str
    observation_id: str
    det_score: float
    face_short_side: int
    sharpness: float
    quality_score: float
    quality_tier: str
    yaw: Optional[float]
    embedding: List[float] = field(repr=False)


@dataclass
class BodyFeature:
    asset_path: str
    rgba_path: Optional[str]
    observation_id: str
    grounding_score: float
    width: int
    height: int
    sharpness: float
    quality_score: float
    quality_tier: str
    embedding: List[float] = field(repr=False)


@dataclass
class TrackletSignature:
    tracklet_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    observation_ids: List[str]
    face_gallery: List[Dict[str, Any]]
    body_gallery: List[Dict[str, Any]]
    face_embeddings: List[List[float]]
    body_embeddings: List[List[float]]
    appearance_description: Optional[str]
    appearance_embedding: Optional[List[float]]
    cooccurring_tracklets: List[str]
    quality: Dict[str, Any]
    face_features: List[FaceFeature] = field(default_factory=list)
    body_features: List[BodyFeature] = field(default_factory=list)
    # v2 inputs carry per-pair co-occurrence provenance; v1 inputs leave this
    # empty and the builder falls back to conservative interval overlap.
    cannot_link_evidence: Dict[str, Any] = field(default_factory=dict)
    signature_schema: str = "shot-tracklets-v1"


@dataclass
class PairEvidence:
    edge_id: str
    tracklet_a: str
    tracklet_b: str
    decision: str
    source: str
    max_similarity: Optional[float] = None
    topk_mean: Optional[float] = None
    support_pair_count: int = 0
    quality_weighted_similarity: Optional[float] = None
    reason: str = ""


@dataclass
class CharacterCandidate:
    candidate_id: str
    tracklet_ids: List[str]
    face_gallery: List[Dict[str, Any]]
    body_gallery: List[Dict[str, Any]]
    appearance_states: List[Dict[str, Any]]
    merge_evidence: List[str]
    cannot_links: List[str]
    confidence: float
    status: str = "CANDIDATE"
    purity_dropped_faces: List[Dict[str, Any]] = field(default_factory=list)
    # "verified" when the gallery was large enough for the purity filter to
    # run, "insufficient_evidence" when it was kept unchecked.
    purity_status: str = "insufficient_evidence"


class IncrementalCharacterRegistry:
    """Small graph-backed API for online attach/create and reversible edges."""

    def __init__(self, cannot_links: Optional[Set[frozenset[str]]] = None):
        self.cannot_links = cannot_links or set()
        self.candidates: Dict[str, Set[str]] = {}
        self.tracklet_owner: Dict[str, str] = {}
        self.identity_edges: Dict[str, PairEvidence] = {}

    def create_candidate(self, signature: TrackletSignature) -> str:
        candidate_id = f"CAND_CHR{len(self.candidates)+1:03d}"
        self.candidates[candidate_id] = {signature.tracklet_id}
        self.tracklet_owner[signature.tracklet_id] = candidate_id
        return candidate_id

    def match(self, signature: TrackletSignature,
              evidence: Sequence[PairEvidence]) -> Dict[str, Any]:
        ranked = sorted((edge for edge in evidence
                         if signature.tracklet_id in (edge.tracklet_a, edge.tracklet_b)),
                        key=lambda edge: edge.topk_mean if edge.topk_mean is not None else -1,
                        reverse=True)
        best = ranked[0] if ranked else None
        return {"decision": best.decision if best else DECISION_NONE,
                "candidate_id": (self.tracklet_owner.get(
                    best.tracklet_b if best and best.tracklet_a == signature.tracklet_id
                    else best.tracklet_a) if best else None),
                "evidence": asdict(best) if best else None}

    def attach(self, tracklet_id: str, candidate_id: str, evidence: PairEvidence) -> None:
        members = self.candidates[candidate_id]
        if any(frozenset((tracklet_id, member)) in self.cannot_links for member in members):
            raise ValueError("attach would violate Cannot-Link")
        if evidence.decision != DECISION_MUST:
            raise ValueError("only MUST_LINK evidence can attach automatically")
        members.add(tracklet_id); self.tracklet_owner[tracklet_id] = candidate_id
        self.identity_edges[evidence.edge_id] = evidence

    def remove_identity_edge(self, edge_id: str) -> None:
        self.identity_edges.pop(edge_id, None)

    def rebuild_clusters(self, tracklets: Sequence[TrackletSignature],
                         compatibility_mean: float = .4) -> List[Set[str]]:
        evidence = {frozenset((edge.tracklet_a, edge.tracklet_b)): edge
                    for edge in self.identity_edges.values()}
        return conservative_clusters(tracklets, evidence, self.cannot_links,
                                     compatibility_mean)


class BodyFeatureExtractor:
    """DINOv3 global body descriptor using the repository-local checkpoint."""

    def __init__(self, model_path: Path, device_id: int = 0):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        # This local DINOv3 checkpoint produces non-finite CLS tokens under
        # FP16 in the project environment; identity evidence must remain FP32.
        dtype = torch.float32
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                               torch_dtype=dtype).to(self.device).eval()

    def embedding(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
        vector = output.last_hidden_state[:, 0].float().cpu().numpy()[0]
        if not np.isfinite(vector).all():
            raise RuntimeError("DINOv3 produced a non-finite body embedding")
        return _unit(vector)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector.astype(np.float32) / max(norm, 1e-12)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def discover_signature_files(inputs: Sequence[str]) -> List[Path]:
    found: Set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob("tracklet_signatures.json"))
    return sorted(found)


def load_tracklet_signatures(paths: Sequence[Path]) -> List[TrackletSignature]:
    by_id: Dict[str, TrackletSignature] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema_version") or "shot-tracklets-v1")
        for raw in payload.get("tracklet_signatures", []):
            tracklet_id = str(raw["tracklet_id"])
            if tracklet_id in by_id:
                raise ValueError(f"duplicate tracklet_id {tracklet_id}: {path}")
            appearance = raw.get("appearance_embedding")
            if appearance is None and isinstance(raw.get("appearance_state"), dict):
                appearance = raw["appearance_state"].get("embedding")
            by_id[tracklet_id] = TrackletSignature(
                tracklet_id=tracklet_id, shot_id=str(raw["shot_id"]),
                start_frame=int(raw.get("start_frame", 0)),
                end_frame=int(raw.get("end_frame", 0)),
                start_time=float(raw.get("start_time", 0.0)),
                end_time=float(raw.get("end_time", 0.0)),
                observation_ids=list(raw.get("observation_ids", [])),
                face_gallery=list(raw.get("face_gallery", [])),
                body_gallery=list(raw.get("body_gallery", [])),
                face_embeddings=list(raw.get("face_embeddings", [])),
                body_embeddings=list(raw.get("body_embeddings", [])),
                appearance_description=raw.get("appearance_description"),
                appearance_embedding=appearance,
                cooccurring_tracklets=list(raw.get("cooccurring_tracklets",
                                                    raw.get("cannot_links", []))),
                quality=dict(raw.get("quality", {})),
                cannot_link_evidence=dict(raw.get("cannot_link_evidence", {})),
                signature_schema=schema,
            )
    return sorted(by_id.values(), key=lambda item: item.tracklet_id)


class FaceFeatureExtractor:
    """Detect, align and embed gallery faces with the local InsightFace pack."""

    def __init__(self, model_path: Path, device_id: int = 0):
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # model_path points to .../insightface/models/<pack>/recognition.onnx.
        # FaceAnalysis supplies the missing five-point alignment step before
        # recognition and rejects crops that cannot be confirmed as faces.
        pack = model_path.parent.name
        root = model_path.parents[2]
        self.app = FaceAnalysis(name=pack, root=str(root),
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app.prepare(ctx_id=device_id, det_size=(320, 320))

    def embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        height, width = image.shape[:2]
        pad_y, pad_x = max(8, height // 4), max(8, width // 4)
        padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_REPLICATE)
        faces = self.app.get(padded)
        if not faces:
            return None
        center_x, center_y = padded.shape[1] / 2.0, padded.shape[0] / 2.0
        def rank(face: Any) -> Tuple[float, float]:
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            area = max((x2 - x1) * (y2 - y1), 0.0)
            distance = ((x1 + x2) / 2.0 - center_x) ** 2 + ((y1 + y2) / 2.0 - center_y) ** 2
            return area, -distance
        face = max(faces, key=rank)
        vector = getattr(face, "normed_embedding", None)
        if vector is None:
            return None
        return _unit(np.asarray(vector, dtype=np.float32))


def face_quality(asset: Dict[str, Any], image: np.ndarray,
                 min_det_score: float, min_face_side: int,
                 min_sharpness: float) -> Tuple[str, Dict[str, Any]]:
    height, width = image.shape[:2]
    short_side = min(height, width)
    laplacian = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    cv2.CV_64F).var())
    det_score = float(asset.get("det_score") or 0.0)
    yaw = asset.get("yaw")
    yaw_ok = yaw is None or abs(float(yaw)) <= 65.0
    high = (det_score >= min_det_score and short_side >= min_face_side
            and laplacian >= min_sharpness and yaw_ok)
    medium = (det_score >= max(0.55, min_det_score - 0.15)
              and short_side >= max(28, min_face_side // 2)
              and laplacian >= min_sharpness * 0.45 and yaw_ok)
    tier = "high" if high else "medium" if medium else "rejected"
    resolution = min(1.0, short_side / max(float(min_face_side * 2), 1.0))
    sharpness_score = 1.0 - math.exp(-laplacian / 250.0)
    quality = 0.40 * det_score + 0.30 * resolution + 0.30 * sharpness_score
    return tier, {"det_score": round(det_score, 6), "face_short_side": short_side,
                  "sharpness": round(laplacian, 6), "quality_score": round(quality, 6),
                  "yaw": None if yaw is None else float(yaw)}


def extract_face_features(tracklets: Sequence[TrackletSignature], extractor: FaceFeatureExtractor,
                          min_det_score: float, min_face_side: int,
                          min_sharpness: float) -> Dict[str, int]:
    stats = {"high": 0, "medium": 0, "rejected": 0, "missing": 0,
             "alignment_failed": 0}
    for tracklet in tracklets:
        for asset in tracklet.face_gallery:
            path = str(asset.get("crop_path") or asset.get("asset_path") or "")
            image = cv2.imread(path) if path else None
            if image is None or image.size == 0:
                stats["missing"] += 1
                continue
            tier, quality = face_quality(asset, image, min_det_score,
                                         min_face_side, min_sharpness)
            stats[tier] += 1
            if tier == "rejected":
                continue
            embedding = extractor.embedding(image)
            if embedding is None:
                stats["alignment_failed"] += 1
                continue
            feature = FaceFeature(
                asset_path=path, observation_id=str(asset.get("observation_id", "")),
                quality_tier=tier, embedding=embedding.tolist(), **quality)
            tracklet.face_features.append(feature)
        tracklet.face_embeddings = [item.embedding for item in tracklet.face_features]
    return stats


def extract_body_features(tracklets: Sequence[TrackletSignature], extractor: BodyFeatureExtractor,
                          min_side: int = 64, min_sharpness: float = 20.0,
                          min_score: float = 0.50) -> Dict[str, int]:
    stats = {"high": 0, "medium": 0, "rejected": 0, "missing": 0}
    for tracklet in tracklets:
        for asset in tracklet.body_gallery:
            path = str(asset.get("crop_path") or "")
            image = cv2.imread(path) if path else None
            if image is None or image.size == 0:
                stats["missing"] += 1; continue
            height, width = image.shape[:2]; short = min(height, width)
            sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                            cv2.CV_64F).var())
            score = float(asset.get("grounding_score") or 0.0)
            high = short >= min_side and sharpness >= min_sharpness and score >= min_score
            medium = short >= max(32, min_side // 2) and sharpness >= min_sharpness * .4
            tier = "high" if high else "medium" if medium else "rejected"
            stats[tier] += 1
            if tier == "rejected": continue
            quality = (.35 * min(1.0, short / 256.0)
                       + .35 * (1.0 - math.exp(-sharpness / 250.0)) + .30 * score)
            tracklet.body_features.append(BodyFeature(
                asset_path=path, rgba_path=asset.get("rgba_crop_path"),
                observation_id=str(asset.get("observation_id", "")),
                grounding_score=round(score, 6), width=width, height=height,
                sharpness=round(sharpness, 6), quality_score=round(quality, 6),
                quality_tier=tier, embedding=extractor.embedding(image).tolist()))
        tracklet.body_embeddings = [item.embedding for item in tracklet.body_features]
    return stats


def _shot_number(shot_id: str) -> Optional[int]:
    match = re.search(r"(\d+)$", shot_id)
    return int(match.group(1)) if match else None


def body_similarity(left: TrackletSignature, right: TrackletSignature,
                    top_pairs: int = 3) -> Optional[float]:
    lf = [item for item in left.body_features if item.quality_tier == "high"]
    rf = [item for item in right.body_features if item.quality_tier == "high"]
    if not lf or not rf: return None
    scores = [float(np.dot(a.embedding, b.embedding)) for a in lf for b in rf]
    scores = sorted((score for score in scores if math.isfinite(score)), reverse=True)
    if not scores: return None
    return round(float(np.mean(scores[:top_pairs])), 6)


def temporally_local(left: TrackletSignature, right: TrackletSignature,
                     max_shot_gap: int) -> bool:
    a, b = _shot_number(left.shot_id), _shot_number(right.shot_id)
    return a is not None and b is not None and abs(a - b) <= max_shot_gap


def build_appearance_states(candidate_id: str, members: Sequence[TrackletSignature],
                            threshold: float = .82) -> List[Dict[str, Any]]:
    """Group body-bearing members into changeable clothing/appearance states."""
    states: List[Dict[str, Any]] = []
    for member in sorted(members, key=lambda item: (_shot_number(item.shot_id) or 10**12,
                                                    item.tracklet_id)):
        features = [item for item in member.body_features if item.quality_tier == "high"]
        if not features: continue
        prototype = _unit(np.mean([item.embedding for item in features], axis=0))
        chosen = None
        for state in states:
            if float(np.dot(prototype, np.asarray(state["prototype"], np.float32))) >= threshold:
                chosen = state; break
        if chosen is None:
            chosen = {"state_id": f'{candidate_id}_APP{len(states)+1:03d}',
                      "tracklet_ids": [], "shots": [], "body_assets": [],
                      "prototype": prototype.tolist(), "description": None}
            states.append(chosen)
        chosen["tracklet_ids"].append(member.tracklet_id)
        chosen["shots"].append(member.shot_id)
        chosen["body_assets"].extend(item.asset_path for item in features)
        old = np.asarray(chosen["prototype"], np.float32)
        chosen["prototype"] = _unit(.7 * old + .3 * prototype).tolist()
    return states


def build_cannot_links(tracklets: Sequence[TrackletSignature]
                       ) -> Tuple[Set[frozenset[str]], List[Dict[str, Any]],
                                  Dict[frozenset[str], str]]:
    """Hard cannot-links must trace to detection-frame co-occurrence.

    v2 signatures list only detection-confirmed pairs in ``cannot_links`` and
    carry the frames in ``cannot_link_evidence``.  Same-shot frame ranges can
    also intersect purely through propagation tails or duplicate tracks; those
    pairs are reported as ``overlap_review_pairs`` for audit instead of being
    promoted to hard edges.  v1 signatures have no provenance, so pairs
    involving them keep the legacy conservative interval-overlap fallback.
    """
    ids = {item.tracklet_id for item in tracklets}
    by_id = {item.tracklet_id: item for item in tracklets}
    links: Set[frozenset[str]] = set()
    sources: Dict[frozenset[str], str] = {}
    review: List[Dict[str, Any]] = []
    for tracklet in tracklets:
        for other in tracklet.cooccurring_tracklets:
            if other in ids and other != tracklet.tracklet_id:
                pair = frozenset((tracklet.tracklet_id, other))
                links.add(pair)
                sources.setdefault(pair, DETECTION_COOCCURRENCE)
    for index, left in enumerate(tracklets):
        for right in tracklets[index + 1:]:
            if left.shot_id != right.shot_id:
                continue
            overlap_frames = min(left.end_frame, right.end_frame) \
                - max(left.start_frame, right.start_frame) + 1
            if overlap_frames <= 0:
                continue
            pair = frozenset((left.tracklet_id, right.tracklet_id))
            if pair in links:
                continue
            both_v2 = (left.signature_schema == "shot-tracklets-v2"
                       and right.signature_schema == "shot-tracklets-v2")
            if both_v2:
                review.append({
                    "tracklet_a": left.tracklet_id, "tracklet_b": right.tracklet_id,
                    "shot_id": left.shot_id,
                    "overlap_frames": int(overlap_frames),
                    "reason": "propagation_or_interval_overlap_without_detection_evidence",
                })
            else:
                links.add(pair)
                sources[pair] = LEGACY_INTERVAL_OVERLAP
    return links, review, sources


def _high_features(tracklet: TrackletSignature) -> List[FaceFeature]:
    return [item for item in tracklet.face_features if item.quality_tier == "high"]


def _usable_features(tracklet: TrackletSignature) -> List[FaceFeature]:
    return [item for item in tracklet.face_features if item.quality_tier != "rejected"]


def _quality_weighted_prototype(features: Sequence[FaceFeature]) -> Optional[np.ndarray]:
    if not features:
        return None
    weights = np.asarray([max(item.quality_score, 1e-3) for item in features], np.float32)
    matrix = np.asarray([item.embedding for item in features], np.float32)
    return _unit(np.average(matrix, axis=0, weights=weights))


def face_prototype(tracklet: TrackletSignature) -> Optional[np.ndarray]:
    return _quality_weighted_prototype(_high_features(tracklet))


def _medium_fallback_prototype(tracklet: TrackletSignature) -> Optional[np.ndarray]:
    """Prototype from high+medium faces for tracklets with no high-tier face."""
    return _quality_weighted_prototype(_usable_features(tracklet))


def candidate_pairs(tracklets: Sequence[TrackletSignature], top_k: int,
                    allow_medium: bool = False) -> Set[Tuple[str, str]]:
    prototypes = {item.tracklet_id: face_prototype(item) for item in tracklets}
    if allow_medium:
        # Medium-only tracklets get a fallback prototype so they can enter ANN
        # candidacy; tracklets with a high-tier prototype are untouched.
        for item in tracklets:
            if prototypes[item.tracklet_id] is None:
                prototypes[item.tracklet_id] = _medium_fallback_prototype(item)
    pairs: Set[Tuple[str, str]] = set()
    for left in tracklets:
        if prototypes[left.tracklet_id] is None:
            continue
        neighbors = []
        for right in tracklets:
            if right.tracklet_id == left.tracklet_id or right.shot_id == left.shot_id:
                continue
            prototype = prototypes[right.tracklet_id]
            if prototype is not None:
                neighbors.append((float(np.dot(prototypes[left.tracklet_id], prototype)),
                                  right.tracklet_id))
        for _score, right_id in sorted(neighbors, reverse=True)[:top_k]:
            pairs.add(tuple(sorted((left.tracklet_id, right_id))))
    return pairs


def gallery_similarity(left: TrackletSignature, right: TrackletSignature,
                       top_pairs: int = 3,
                       feature_selector=_high_features) -> Dict[str, Any]:
    lf, rf = feature_selector(left), feature_selector(right)
    if not lf or not rf:
        return {"max_similarity": None, "topk_mean": None, "support_pair_count": 0,
                "quality_weighted_similarity": None}
    scored: List[Tuple[float, float]] = []
    for a in lf:
        av = np.asarray(a.embedding, np.float32)
        for b in rf:
            similarity = float(np.dot(av, np.asarray(b.embedding, np.float32)))
            scored.append((similarity, math.sqrt(a.quality_score * b.quality_score)))
    scored.sort(reverse=True)
    selected = scored[:max(1, top_pairs)]
    weights = np.asarray([item[1] for item in selected], np.float32)
    values = np.asarray([item[0] for item in selected], np.float32)
    weighted = float(np.average(values, weights=weights)) if weights.sum() else float(values.mean())
    return {"max_similarity": round(scored[0][0], 6),
            "topk_mean": round(float(values.mean()), 6),
            "support_pair_count": sum(score >= 0.45 for score, _ in scored),
            "quality_weighted_similarity": round(weighted, 6)}


def classify_pair(left: TrackletSignature, right: TrackletSignature,
                  cannot_links: Set[frozenset[str]], must_max: float,
                  must_mean: float, possible_mean: float,
                  min_support: int,
                  cannot_sources: Optional[Dict[frozenset[str], str]] = None,
                  allow_medium: bool = False, medium_min_mean: float = 0.50,
                  medium_min_max: float = 0.55, medium_min_support: int = 2) -> PairEvidence:
    pair = frozenset((left.tracklet_id, right.tracklet_id))
    edge_id = "EDGE_" + "__".join(sorted(pair))
    if pair in cannot_links:
        source = (cannot_sources or {}).get(pair, LEGACY_INTERVAL_OVERLAP)
        return PairEvidence(edge_id, left.tracklet_id, right.tracklet_id,
                            DECISION_CANNOT, "hard_constraint", reason=source)
    metrics = gallery_similarity(left, right)
    mean = metrics["topk_mean"]
    if mean is None and allow_medium:
        # Either side lacks a high-tier face; retry on high+medium galleries.
        metrics = gallery_similarity(left, right, feature_selector=_usable_features)
        mean = metrics["topk_mean"]
        if mean is not None:
            if (metrics["max_similarity"] >= medium_min_max
                    and mean >= medium_min_mean
                    and metrics["support_pair_count"] >= medium_min_support):
                # Medium-tier evidence only opens a controlled POSSIBLE link;
                # promotion to MUST stays with the VLM confirmation path.
                return PairEvidence(edge_id, left.tracklet_id, right.tracklet_id,
                                    DECISION_POSSIBLE, "face_gallery_medium",
                                    reason="medium_face_multi_support", **metrics)
            return PairEvidence(edge_id, left.tracklet_id, right.tracklet_id,
                                DECISION_NONE, "face_gallery",
                                reason="medium_face_evidence_insufficient", **metrics)
    if mean is None:
        decision, reason = DECISION_NONE, "no_high_quality_face_pair"
    elif (metrics["max_similarity"] >= must_max and mean >= must_mean
          and metrics["support_pair_count"] >= min_support):
        decision, reason = DECISION_MUST, "strict_gallery_face_support"
    elif mean >= possible_mean:
        decision, reason = DECISION_POSSIBLE, "face_evidence_below_must_threshold"
    else:
        decision, reason = DECISION_NONE, "face_similarity_too_low"
    return PairEvidence(edge_id, left.tracklet_id, right.tracklet_id,
                        decision, "face_gallery", reason=reason, **metrics)


def violates_cannot(cluster_a: Set[str], cluster_b: Set[str],
                    cannot_links: Set[frozenset[str]]) -> bool:
    return any(frozenset((a, b)) in cannot_links for a in cluster_a for b in cluster_b)


def compatible_face_clusters(cluster_a: Set[str], cluster_b: Set[str],
                             evidence: Dict[frozenset[str], PairEvidence],
                             cannot_links: Set[frozenset[str]],
                             compatibility_mean: float,
                             tracklets: Optional[Dict[str, TrackletSignature]] = None) -> bool:
    if violates_cannot(cluster_a, cluster_b, cannot_links):
        return False
    supported = []
    semantic_supported = False
    contradicted = False
    for a in cluster_a:
        for b in cluster_b:
            edge = evidence.get(frozenset((a, b)))
            if (edge is None and tracklets is not None
                    and _high_features(tracklets[a]) and _high_features(tracklets[b])):
                # Never accept an unverified transitive bridge between two
                # face-bearing nodes merely because both match a third node.
                return False
            if edge is None:
                continue
            if edge.decision == DECISION_MUST and edge.source in (
                    "vlm_semantic_alignment", "vlm_verified_possible_link"):
                semantic_supported = True
            if edge.topk_mean is not None:
                supported.append(edge.topk_mean)
                if edge.topk_mean < CONTRADICTION_FACE_MEAN:
                    contradicted = True
    if semantic_supported and not contradicted:
        # High-confidence semantic alignment (Stage 1.5) overrides weak or
        # slightly-below-threshold face similarity — but only while no cross
        # pair carries face evidence that actively contradicts the merge.
        # A mislabeled tracklet (e.g. aligned to the primary subject by
        # overlay-video reading) must not drag a different person into the
        # cluster through this override.
        return True
    if not supported:
        return False
    return min(supported) >= compatibility_mean


def conservative_clusters(tracklets: Sequence[TrackletSignature],
                          evidence: Dict[frozenset[str], PairEvidence],
                          cannot_links: Set[frozenset[str]],
                          compatibility_mean: float) -> List[Set[str]]:
    clusters = [{item.tracklet_id} for item in tracklets]
    by_id = {item.tracklet_id: item for item in tracklets}
    must_edges = sorted((edge for edge in evidence.values() if edge.decision == DECISION_MUST),
                        key=lambda edge: edge.topk_mean or -1.0, reverse=True)
    for edge in must_edges:
        ia = next(i for i, cluster in enumerate(clusters) if edge.tracklet_a in cluster)
        ib = next(i for i, cluster in enumerate(clusters) if edge.tracklet_b in cluster)
        if ia == ib:
            continue
        if compatible_face_clusters(clusters[ia], clusters[ib], evidence,
                                    cannot_links, compatibility_mean, by_id):
            clusters[ia] |= clusters[ib]
            clusters.pop(ib)
    return sorted(clusters, key=lambda cluster: sorted(cluster)[0])


def attach_face_consensus_singletons(
        clusters: Sequence[Set[str]],
        tracklets: Dict[str, TrackletSignature],
        evidence: Dict[frozenset[str], PairEvidence],
        cannot_links: Set[frozenset[str]],
        min_anchor_tracklets: int = 2,
        min_face_mean: float = .45,
        min_face_max: float = .48,
        min_support_pairs: int = 2,
        ambiguity_margin: float = .03,
        blocked_edge_ids: Optional[Set[str]] = None) -> List[Set[str]]:
    """Attach a singleton to a mature face cluster using independent anchors.

    Complete-link clustering is intentionally strict, but it can split one
    person when profile/blurred views do not match every member of an already
    pure multi-shot cluster.  This second pass relaxes only that specific
    failure mode: a singleton must be supported by multiple *different*
    tracklets in one mature cluster, with multiple gallery pairs per anchor.
    Competing clusters within ``ambiguity_margin`` keep it unresolved and hard
    cannot-links always win.  Multi-member clusters are never joined here, so
    the rule cannot create broad transitive identity chains.
    """
    blocked_edge_ids = blocked_edge_ids or set()
    result = [set(cluster) for cluster in clusters]
    mature = [cluster for cluster in result if len(cluster) >= min_anchor_tracklets]
    singletons = [cluster for cluster in result if len(cluster) == 1]
    attachments: List[Tuple[str, Set[str], List[Tuple[str, Dict[str, Any]]], float]] = []

    for singleton_cluster in singletons:
        singleton = next(iter(singleton_cluster))
        if not _high_features(tracklets[singleton]):
            continue
        ranked: List[Tuple[float, Set[str], List[Tuple[str, Dict[str, Any]]]]] = []
        for target in mature:
            if violates_cannot(singleton_cluster, target, cannot_links):
                continue
            anchors: List[Tuple[str, Dict[str, Any]]] = []
            for anchor in sorted(target):
                edge_id = "EDGE_" + "__".join(sorted((singleton, anchor)))
                if edge_id in blocked_edge_ids:
                    continue
                metrics = gallery_similarity(tracklets[singleton], tracklets[anchor])
                if (metrics["topk_mean"] is not None
                        and metrics["topk_mean"] >= min_face_mean
                        and metrics["max_similarity"] >= min_face_max
                        and metrics["support_pair_count"] >= min_support_pairs):
                    anchors.append((anchor, metrics))
            if len(anchors) < min_anchor_tracklets:
                continue
            score = float(np.mean([metrics["topk_mean"] for _, metrics in anchors]))
            ranked.append((score, target, anchors))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < ambiguity_margin:
            continue
        score, target, anchors = ranked[0]
        attachments.append((singleton, target, anchors, score))

    for singleton, target, anchors, _score in sorted(
            attachments, key=lambda item: item[3], reverse=True):
        singleton_cluster = next((cluster for cluster in result
                                  if singleton in cluster), None)
        target_cluster = next((cluster for cluster in result
                               if cluster is target or cluster == target), None)
        if (singleton_cluster is None or target_cluster is None
                or singleton_cluster is target_cluster
                or len(singleton_cluster) != 1
                or violates_cannot(singleton_cluster, target_cluster, cannot_links)):
            continue
        for anchor, metrics in anchors:
            pair = frozenset((singleton, anchor))
            evidence[pair] = PairEvidence(
                edge_id="EDGE_" + "__".join(sorted(pair)),
                tracklet_a=min(singleton, anchor), tracklet_b=max(singleton, anchor),
                decision=DECISION_MUST, source="face_cluster_consensus",
                reason="independent_multi_tracklet_face_support", **metrics)
        target_cluster.update(singleton_cluster)
        result.remove(singleton_cluster)
    return sorted(result, key=lambda cluster: sorted(cluster)[0])


def filter_gallery_purity(faces: List[Dict[str, Any]],
                          min_similarity: float = 0.30,
                          min_gallery: int = 4,
                          gap_ratio: float = 0.15,
                          gap_min_keep: float = 0.50
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Split off face crops that disagree with the cluster's face prototype.

    A single tracklet can contain identity-switch observations (tracker drift,
    e.g. visitor leaning over a hospital bed and continuing as the child), so
    even a visually coherent cluster may inherit foreign face crops through one
    contaminated member gallery.  Each crop is compared against a
    leave-one-out prototype -- the quality-weighted mean of every *other*
    crop -- so a crop is never vindicated by its own contribution.  Crops are
    dropped when either

    - their cosine similarity to the reference falls below ``min_similarity``
      (absolute floor), or
    - a similarity gap opens between them and the bulk of the gallery: sorting
      similarities descending, the largest gap (relative to the higher side)
      separates a well-matching majority from outliers whose own similarity is
      below ``gap_min_keep``.

    Galleries smaller than ``min_gallery`` cannot support a stable prototype;
    they are returned untouched with status ``insufficient_evidence`` so the
    candidate is auditable as unverified instead of silently "pure".  Dropped
    crops are returned for audit, not deleted from disk.
    """
    if len(faces) < min_gallery:
        return faces, [], "insufficient_evidence"
    try:
        embs = np.asarray([f["embedding"] for f in faces], np.float32)
    except (KeyError, ValueError):
        return faces, [], "insufficient_evidence"
    weights = np.asarray([max(float(f.get("quality_score", 0.0)), 1e-3)
                          for f in faces], np.float32)

    sims = np.zeros(len(faces), np.float32)
    for i in range(len(faces)):
        others = weights.copy()
        others[i] = 0.0
        proto = (embs * others[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(proto))
        # A face whose exclusion empties the reference cannot be checked; keep
        # it rather than punish it for the gallery's lack of evidence.
        sims[i] = float(embs[i] @ (proto / norm)) if norm > 0 else 1.0

    cutoff = min_similarity
    order = sorted(range(len(faces)), key=lambda i: -float(sims[i]))
    sorted_sims = [float(sims[i]) for i in order]
    best_gap, best_pos = 0.0, None
    for pos in range(1, len(sorted_sims)):
        hi, lo = sorted_sims[pos - 1], sorted_sims[pos]
        gap = hi - lo
        if gap > best_gap:
            best_gap, best_pos = gap, pos
    if (best_pos is not None and best_gap >= gap_ratio
            and sorted_sims[best_pos] < gap_min_keep):
        cutoff = max(cutoff, sorted_sims[best_pos] + 1e-6)

    keep, dropped = [], []
    for feature, similarity in zip(faces, sims):
        if float(similarity) >= cutoff:
            keep.append(feature)
        else:
            feature = dict(feature)
            feature["purity_similarity"] = round(float(similarity), 6)
            dropped.append(feature)
    return keep, dropped, "verified"


def build_candidates(clusters: Sequence[Set[str]], tracklets: Dict[str, TrackletSignature],
                     evidence: Dict[frozenset[str], PairEvidence],
                     cannot_links: Set[frozenset[str]]) -> List[CharacterCandidate]:
    candidates = []
    for number, cluster in enumerate(clusters, 1):
        ids = sorted(cluster); faces, bodies = [], []
        for tracklet_id in ids:
            item = tracklets[tracklet_id]
            faces.extend(asdict(feature) for feature in item.face_features
                         if feature.quality_tier != "rejected")
            bodies.extend(item.body_gallery)
        faces, purity_dropped, purity_status = filter_gallery_purity(faces)
        internal_edges = [edge for pair, edge in evidence.items()
                          if pair.issubset(cluster) and edge.decision == DECISION_MUST]
        external_cannot = sorted({other for pair in cannot_links for other in pair
                                  if other not in cluster and bool(pair & cluster)})
        # Candidate promotion answers "is this a stable visible character?"
        # and is therefore driven by the best verified face asset.  Identity
        # edge similarities answer a different question (whether two
        # tracklets may merge) and have already passed the strict MUST_LINK
        # plus cluster-compatibility gates.  Reusing the weakest identity edge
        # here made a larger, correctly merged cluster less likely to be
        # promoted than any of its singleton members.
        confidence = max((f["quality_score"] for f in faces), default=0.0)
        candidates.append(CharacterCandidate(
            candidate_id=f"CAND_CHR{number:03d}", tracklet_ids=ids,
            face_gallery=sorted(faces, key=lambda x: -x["quality_score"]),
            body_gallery=sorted(bodies, key=lambda x: -float(x.get("quality_score", 0.0))),
            appearance_states=[], merge_evidence=sorted(edge.edge_id for edge in internal_edges),
            cannot_links=external_cannot, confidence=round(float(confidence), 6),
            purity_dropped_faces=purity_dropped, purity_status=purity_status))
    return candidates


def add_body_bridge_evidence(candidates: Sequence[CharacterCandidate],
                             tracklets: Dict[str, TrackletSignature],
                             cannot_links: Set[frozenset[str]], max_shot_gap: int,
                             must_threshold: float, possible_threshold: float,
                             allow_must: bool) -> List[PairEvidence]:
    """Generate context-gated body edges; hard merge remains opt-in."""
    result: List[PairEvidence] = []
    owner = {tid: candidate for candidate in candidates for tid in candidate.tracklet_ids}
    ids = sorted(tracklets)
    for index, left_id in enumerate(ids):
        left = tracklets[left_id]
        for right_id in ids[index + 1:]:
            right = tracklets[right_id]
            if owner[left_id].candidate_id == owner[right_id].candidate_id: continue
            pair = frozenset((left_id, right_id))
            if pair in cannot_links or not temporally_local(left, right, max_shot_gap): continue
            # Body is used only when at least one side lacks a high-quality face.
            if _high_features(left) and _high_features(right): continue
            similarity = body_similarity(left, right)
            if similarity is None or similarity < possible_threshold: continue
            decision = (DECISION_MUST if allow_must and similarity >= must_threshold
                        else DECISION_POSSIBLE)
            result.append(PairEvidence(
                edge_id="BODY_" + "__".join(sorted(pair)), tracklet_a=left_id,
                tracklet_b=right_id, decision=decision, source="body_bridge",
                topk_mean=similarity, max_similarity=similarity,
                support_pair_count=1, quality_weighted_similarity=similarity,
                reason="temporally_local_body_support"))
    return result


def apply_vlm_decisions(evidence: Dict[frozenset[str], PairEvidence],
                        by_id: Dict[str, TrackletSignature],
                        path: Optional[str],
                        cannot_links: Set[frozenset[str]],
                        min_confidence: float = 0.90,
                        allow_edge_creation: bool = False,
                        creation_face_floor: float = 0.30) -> Dict[str, int]:
    """Apply cached verifier JSON to POSSIBLE links and create new MUST links.

    Hard cannot-link constraints always win. By default the verifier may only
    *upgrade* an existing face-based POSSIBLE_LINK to MUST_LINK — this is the
    contract documented by the Stage 1.5 exporter ("Stage 3 may promote only
    existing POSSIBLE_LINK edges").  Creating brand-new MUST edges for pairs
    that have no face evidence at all lets a mislabeled semantic alignment
    merge visually distinct people into one cluster, so edge creation is
    restricted to pairs whose face gallery simply has no comparable crops
    (``no_high_quality_face_pair``), and can be disabled entirely.
    """
    stats = {"upgraded": 0, "created": 0, "skipped_cannot_link": 0,
             "skipped_low_confidence": 0, "skipped_unknown_tracklet": 0,
             "skipped_face_contradiction": 0, "total_decisions": 0}
    if not path:
        return stats
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    decisions = payload.get("decisions", payload if isinstance(payload, list) else [])
    stats["total_decisions"] = len(decisions)
    for item in decisions:
        a = item.get("tracklet_a"); b = item.get("tracklet_b")
        if not a or not b or a == b:
            continue
        if a not in by_id or b not in by_id:
            stats["skipped_unknown_tracklet"] += 1
            continue
        pair = frozenset((a, b))
        if pair in cannot_links:
            stats["skipped_cannot_link"] += 1
            continue
        if item.get("same_identity") is not True:
            continue
        confidence = float(item.get("confidence", 0.0))
        if confidence < min_confidence:
            stats["skipped_low_confidence"] += 1
            continue
        existing = evidence.get(pair)
        if existing is None:
            if not allow_edge_creation:
                continue
            # The pair was never a face-based ANN candidate.  Check the
            # galleries on demand: a semantic label may create an edge only
            # when face evidence is missing, never when existing faces
            # actively contradict the claimed identity.  The contradiction
            # check falls back to medium-tier crops so that a tracklet whose
            # faces are merely dim (not absent) still vetoes a wrong merge.
            metrics = gallery_similarity(by_id[a], by_id[b])
            if metrics["topk_mean"] is None:
                metrics = gallery_similarity(by_id[a], by_id[b],
                                             feature_selector=_usable_features)
            if metrics["topk_mean"] is not None and metrics["topk_mean"] < creation_face_floor:
                stats["skipped_face_contradiction"] += 1
                continue
            left, right = sorted(pair)
            edge = PairEvidence(
                edge_id="EDGE_" + "__".join(sorted(pair)),
                tracklet_a=left, tracklet_b=right,
                decision=DECISION_MUST,
                source="vlm_semantic_alignment",
                reason=f"semantic_same_identity_conf_{confidence:.2f}",
                **{key: value for key, value in metrics.items()
                   if key in {"topk_mean", "max_similarity", "support_pair_count",
                              "quality_weighted_similarity"}})
            evidence[pair] = edge
            stats["created"] += 1
        elif existing.decision == DECISION_POSSIBLE:
            existing.decision = DECISION_MUST
            existing.source = "vlm_verified_possible_link"
            existing.reason = f"cached_vlm_confirmation_conf_{confidence:.2f}"
            stats["upgraded"] += 1
    return stats


def assign_asset_slots(candidate: CharacterCandidate, output: Path,
                       face_top_k: int = 5, body_top_k: int = 5) -> Dict[str, Any]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "face/frontal": [], "face/three_quarter": [], "face/profile": [],
        "body/front": [], "body/side": [], "body/back": [],
        "body/full_body": [], "body/upper_body": []}
    for asset in candidate.face_gallery:
        yaw = asset.get("yaw")
        slot = ("face/frontal" if yaw is None or abs(float(yaw)) < 20 else
                "face/three_quarter" if abs(float(yaw)) < 50 else "face/profile")
        slots[slot].append(asset)
    for asset in candidate.body_gallery:
        image = cv2.imread(str(asset.get("crop_path") or ""))
        if image is None: continue
        ratio = image.shape[0] / max(image.shape[1], 1)
        slots["body/full_body" if ratio >= 1.7 else "body/upper_body"].append(asset)
    limits = {key: (face_top_k if key.startswith("face/") else body_top_k) for key in slots}
    exported: Dict[str, List[str]] = {}
    for slot, assets in slots.items():
        ranked = sorted(assets, key=lambda item: -float(item.get("quality_score", 0.0)))[:limits[slot]]
        exported[slot] = []
        target = output / slot
        for number, asset in enumerate(ranked, 1):
            source = Path(str(asset.get("asset_path") or asset.get("crop_path") or ""))
            if not source.is_file(): continue
            destination = target / f"{number:03d}_{source.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            exported[slot].append(str(destination))
    return exported


def promote_candidates(candidates: Sequence[CharacterCandidate], output: Path,
                       min_confidence: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    characters, pending = [], []
    number = 1
    for candidate in candidates:
        high_faces = [item for item in candidate.face_gallery if item.get("quality_tier") == "high"]
        medium_faces = [item for item in candidate.face_gallery if item.get("quality_tier") == "medium"]
        multi_tracklet = len(candidate.tracklet_ids) >= 2
        # A high-tier face is the normal bar for a usable identity reference.
        # But a recurring character confirmed across several shots by semantic
        # alignment (e.g. a dimly-lit sound engineer whose best crop is only
        # medium quality) is still a real character, so accept medium faces
        # when the cluster spans multiple tracklets. Single-tracklet candidates
        # must still clear the high-tier bar to avoid promoting background noise.
        stable = candidate.confidence >= min_confidence and (
            bool(high_faces) or (bool(medium_faces) and multi_tracklet))
        if not stable:
            pending.append(asdict(candidate)); continue
        character_id = f"CHR{number:03d}"; number += 1
        root = output / "characters" / character_id
        slots = assign_asset_slots(candidate, root)
        record = {"character_id": character_id, "source_candidate_id": candidate.candidate_id,
                  "tracklet_ids": candidate.tracklet_ids,
                  "face_gallery": {key.split('/', 1)[1]: value for key, value in slots.items()
                                   if key.startswith("face/")},
                  "body_gallery": {key.split('/', 1)[1]: value for key, value in slots.items()
                                   if key.startswith("body/")},
                  "appearance_states": candidate.appearance_states,
                  "shots": sorted({re.sub(r"_T\d+$", "", tid) for tid in candidate.tracklet_ids}),
                  "identity_confidence": candidate.confidence,
                  "voice_id": None, "audio_assets": []}
        _write_json(root / "character.json", record)
        _write_json(root / "tracklets.json", {"tracklet_ids": candidate.tracklet_ids,
                                               "merge_evidence": candidate.merge_evidence})
        characters.append(record)
    return characters, pending


def evaluate_registry(candidates: Sequence[CharacterCandidate], ground_truth_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not ground_truth_path: return None
    truth_payload = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    truth = truth_payload.get("tracklet_to_character", truth_payload)
    predicted = {tid: candidate.candidate_id for candidate in candidates for tid in candidate.tracklet_ids}
    ids = sorted(set(truth) & set(predicted)); tp = fp = fn = 0
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            same_gt = truth[left] == truth[right]; same_pred = predicted[left] == predicted[right]
            tp += same_gt and same_pred; fp += not same_gt and same_pred; fn += same_gt and not same_pred
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    purities = []
    for candidate in candidates:
        labels = [truth[tid] for tid in candidate.tracklet_ids if tid in truth]
        if labels: purities.append(max(labels.count(label) for label in set(labels)) / len(labels))
    fragmentation = {label: len({predicted[tid] for tid in ids if truth[tid] == label})
                     for label in set(truth[tid] for tid in ids)}
    return {"labeled_tracklets": len(ids), "pairwise_precision": round(precision, 6),
            "pairwise_recall": round(recall, 6),
            "cluster_purity": round(float(np.mean(purities or [1.])), 6),
            "fragmentation": fragmentation}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    files = discover_signature_files(args.inputs)
    if not files:
        raise FileNotFoundError("no tracklet_signatures.json found")
    tracklets = load_tracklet_signatures(files)
    model_path = Path(args.face_model).expanduser().resolve()
    extractor = FaceFeatureExtractor(model_path, args.device_id)
    quality_stats = extract_face_features(tracklets, extractor, args.min_face_det_score,
                                          args.min_face_side, args.min_face_sharpness)
    body_stats = {"high": 0, "medium": 0, "rejected": 0, "missing": 0}
    if not args.disable_body_features:
        body_extractor = BodyFeatureExtractor(Path(args.body_model).expanduser().resolve(),
                                              args.device_id)
        body_stats = extract_body_features(tracklets, body_extractor,
                                           args.min_body_side, args.min_body_sharpness,
                                           args.min_body_score)
    cannot_links, overlap_review_pairs, cannot_sources = build_cannot_links(tracklets)
    by_id = {item.tracklet_id: item for item in tracklets}
    pairs = candidate_pairs(tracklets, args.candidate_top_k,
                            allow_medium=args.allow_medium_face_linking)
    evidence: Dict[frozenset[str], PairEvidence] = {}
    # Hard constraints are always exported, even when they are not ANN candidates.
    for pair in cannot_links:
        left, right = sorted(pair)
        edge = classify_pair(by_id[left], by_id[right], cannot_links,
                             args.must_face_max, args.must_face_mean,
                             args.possible_face_mean, args.min_support_pairs,
                             cannot_sources,
                             allow_medium=args.allow_medium_face_linking,
                             medium_min_mean=args.medium_link_min_mean,
                             medium_min_max=args.medium_link_min_max,
                             medium_min_support=args.medium_link_min_support)
        evidence[pair] = edge
    for left, right in sorted(pairs):
        pair = frozenset((left, right))
        evidence[pair] = classify_pair(by_id[left], by_id[right], cannot_links,
                                       args.must_face_max, args.must_face_mean,
                                       args.possible_face_mean, args.min_support_pairs,
                                       cannot_sources,
                                       allow_medium=args.allow_medium_face_linking,
                                       medium_min_mean=args.medium_link_min_mean,
                                       medium_min_max=args.medium_link_min_max,
                                       medium_min_support=args.medium_link_min_support)
    removed_edges = set(args.remove_edge or [])
    clusters = conservative_clusters(tracklets, evidence, cannot_links,
                                     args.cluster_compatibility_mean)
    if not args.disable_face_consensus_attach:
        clusters = attach_face_consensus_singletons(
            clusters, by_id, evidence, cannot_links,
            args.consensus_min_anchor_tracklets, args.consensus_face_mean,
            args.consensus_face_max, args.consensus_min_support_pairs,
            args.consensus_ambiguity_margin, removed_edges)
    candidates = build_candidates(clusters, by_id, evidence, cannot_links)
    body_edges = add_body_bridge_evidence(
        candidates, by_id, cannot_links, args.max_body_shot_gap,
        args.must_body_mean, args.possible_body_mean, args.allow_body_must_link)
    for edge in body_edges:
        evidence[frozenset((edge.tracklet_a, edge.tracklet_b))] = edge
    vlm_stats = apply_vlm_decisions(evidence, by_id, args.vlm_decisions, cannot_links,
                                    allow_edge_creation=args.allow_vlm_edge_creation,
                                    creation_face_floor=args.vlm_creation_face_floor)
    evidence = {pair: edge for pair, edge in evidence.items() if edge.edge_id not in removed_edges}
    clusters = conservative_clusters(tracklets, evidence, cannot_links,
                                     args.cluster_compatibility_mean)
    if not args.disable_face_consensus_attach:
        clusters = attach_face_consensus_singletons(
            clusters, by_id, evidence, cannot_links,
            args.consensus_min_anchor_tracklets, args.consensus_face_mean,
            args.consensus_face_max, args.consensus_min_support_pairs,
            args.consensus_ambiguity_margin, removed_edges)
    candidates = build_candidates(clusters, by_id, evidence, cannot_links)
    for candidate in candidates:
        candidate.appearance_states = build_appearance_states(
            candidate.candidate_id, [by_id[tid] for tid in candidate.tracklet_ids],
            args.appearance_state_threshold)
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    serial_tracklets = []
    for item in tracklets:
        raw = asdict(item)
        serial_tracklets.append(raw)
    edge_list = [asdict(edge) for edge in sorted(evidence.values(), key=lambda x: x.edge_id)]
    candidate_list = [asdict(item) for item in candidates]
    graph = {"schema_version": SCHEMA_VERSION, "nodes": [item.tracklet_id for item in tracklets],
             "edges": edge_list}
    registry = {
        "schema_version": SCHEMA_VERSION, "status": "ok",
        "mode": "stage3_cross_shot_identity_association",
        "source_files": [str(path) for path in files],
        "models": {"face_embedding": str(model_path),
                   "body_embedding": None if args.disable_body_features else str(Path(args.body_model).resolve()),
                   "vlm_verifier": args.vlm_decisions},
        "settings": {key: getattr(args, key) for key in (
            "min_face_det_score", "min_face_side", "min_face_sharpness",
            "candidate_top_k", "must_face_max", "must_face_mean",
            "possible_face_mean", "min_support_pairs", "cluster_compatibility_mean",
            "disable_face_consensus_attach", "consensus_min_anchor_tracklets",
            "consensus_face_mean", "consensus_face_max",
            "consensus_min_support_pairs", "consensus_ambiguity_margin",
            "allow_medium_face_linking", "medium_link_min_mean",
            "medium_link_min_max", "medium_link_min_support")},
        "quality_gate_stats": {"face": quality_stats, "body": body_stats},
        "vlm_decision_stats": vlm_stats,
        "tracklet_count": len(tracklets), "candidate_count": len(candidates),
        "must_link_count": sum(edge.decision == DECISION_MUST for edge in evidence.values()),
        "possible_link_count": sum(edge.decision == DECISION_POSSIBLE for edge in evidence.values()),
        "cannot_link_count": len(cannot_links),
        "cannot_link_source_counts": {
            source: sum(1 for item in cannot_sources.values() if item == source)
            for source in sorted(set(cannot_sources.values()))},
        "overlap_review_pair_count": len(overlap_review_pairs),
        "cannot_link_violation_count": 0,
        "candidates": candidate_list,
        "pending_links": [edge for edge in edge_list if edge["decision"] == DECISION_POSSIBLE],
        "removed_identity_edges": sorted(removed_edges),
    }
    registry["cannot_link_violation_count"] = sum(
        any(pair.issubset(cluster) for pair in cannot_links) for cluster in clusters)
    characters, pending = promote_candidates(candidates, output, args.chr_min_confidence)
    registry["characters"] = characters
    registry["pending_candidates"] = pending
    registry["character_count"] = len(characters)
    registry["evaluation"] = evaluate_registry(candidates, args.ground_truth)
    _write_json(output / "tracklet_features.json", {"schema_version": SCHEMA_VERSION,
                                                     "tracklets": serial_tracklets})
    _write_json(output / "cannot_links.json", {"schema_version": SCHEMA_VERSION,
                                                "cannot_links": [sorted(pair) for pair in sorted(cannot_links, key=lambda x: sorted(x))]})
    _write_json(output / "overlap_review_pairs.json", {
        "schema_version": SCHEMA_VERSION,
        "note": "same-shot interval overlaps without detection-frame co-occurrence "
                "evidence; reviewed here, never hard cannot-links",
        "overlap_review_pairs": overlap_review_pairs,
    })
    _write_json(output / "identity_graph.json", graph)
    _write_json(output / "possible_links.json", {
        "schema_version": SCHEMA_VERSION,
        "possible_links": [edge for edge in edge_list
                           if edge["decision"] == DECISION_POSSIBLE],
    })
    # JSONL stays diff/stream friendly and is the canonical audit ledger.
    ledger_path = output / "merge_evidence.jsonl"
    ledger_tmp = ledger_path.with_suffix(".jsonl.tmp")
    ledger_tmp.write_text("".join(json.dumps(edge, ensure_ascii=False) + "\n"
                                  for edge in edge_list), encoding="utf-8")
    ledger_tmp.replace(ledger_path)
    for candidate in candidate_list:
        _write_json(output / "candidates" / f'{candidate["candidate_id"]}.json', candidate)
    _write_json(output / "registry.json", registry)
    return registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build face-first visual character candidates.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Tracklet signature files or directories containing them")
    parser.add_argument("--output", required=True)
    parser.add_argument("--face-model", default="local_models/insightface/models/buffalo_l/w600k_r50.onnx")
    parser.add_argument("--body-model", default="local_models/dinov3-vitl16")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--min-face-det-score", type=float, default=0.75)
    parser.add_argument("--min-face-side", type=int, default=48)
    parser.add_argument("--min-face-sharpness", type=float, default=30.0)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--must-face-max", type=float, default=0.55)
    parser.add_argument("--must-face-mean", type=float, default=0.50)
    parser.add_argument("--possible-face-mean", type=float, default=0.35)
    parser.add_argument("--min-support-pairs", type=int, default=1)
    parser.add_argument("--cluster-compatibility-mean", type=float, default=0.40)
    parser.add_argument("--disable-face-consensus-attach", action="store_true",
                        help="Disable conservative multi-anchor singleton attachment")
    parser.add_argument("--consensus-min-anchor-tracklets", type=int, default=2)
    parser.add_argument("--consensus-face-mean", type=float, default=0.45)
    parser.add_argument("--consensus-face-max", type=float, default=0.48)
    parser.add_argument("--consensus-min-support-pairs", type=int, default=2)
    parser.add_argument("--consensus-ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--allow-medium-face-linking", action="store_true",
                        help="Allow tracklets with no high-tier face but multiple consistent "
                             "medium-tier faces to enter controlled POSSIBLE_LINK candidacy; "
                             "disabled by default")
    parser.add_argument("--medium-link-min-mean", type=float, default=0.50,
                        help="Minimum usable-face topk_mean required for medium-tier linking")
    parser.add_argument("--medium-link-min-max", type=float, default=0.55,
                        help="Minimum usable-face max similarity required for medium-tier linking")
    parser.add_argument("--medium-link-min-support", type=int, default=2,
                        help="Minimum supporting usable-face gallery pairs for medium-tier linking")
    parser.add_argument("--disable-body-features", action="store_true")
    parser.add_argument("--min-body-side", type=int, default=64)
    parser.add_argument("--min-body-sharpness", type=float, default=20.0)
    parser.add_argument("--min-body-score", type=float, default=0.50)
    parser.add_argument("--max-body-shot-gap", type=int, default=2)
    parser.add_argument("--must-body-mean", type=float, default=0.90)
    parser.add_argument("--possible-body-mean", type=float, default=0.78)
    parser.add_argument("--allow-body-must-link", action="store_true",
                        help="Opt in to automatic body-only merges; disabled by default")
    parser.add_argument("--appearance-state-threshold", type=float, default=0.82)
    parser.add_argument("--vlm-decisions", help="Cached verifier JSON; only POSSIBLE links are eligible")
    parser.add_argument("--allow-vlm-edge-creation", action="store_true",
                        help="Opt in to creating new MUST edges from semantic decisions for "
                             "pairs without face evidence; contradicting faces still veto")
    parser.add_argument("--vlm-creation-face-floor", type=float, default=0.35,
                        help="Semantic edge creation is vetoed when both tracklets have "
                             "high-quality faces whose topk_mean falls below this floor")
    parser.add_argument("--chr-min-confidence", type=float, default=0.60)
    parser.add_argument("--remove-edge", action="append",
                        help="Rollback an identity edge by ID and rebuild clusters")
    parser.add_argument("--ground-truth", help="Optional Tracklet-to-character JSON benchmark")
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in (
        "status", "tracklet_count", "candidate_count", "must_link_count",
        "possible_link_count", "cannot_link_count", "overlap_review_pair_count",
        "cannot_link_violation_count")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
