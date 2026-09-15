#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4: maintain a compact, traceable visual character asset bank.

This module consumes the Stage-3 registry and tracklet feature cache.  It never
changes identity membership: only promoted CHR records are admitted.  Every
asset goes through quality, purity, classification, deduplication and bounded
slot competition before it is copied into the persistent bank.

Face purity is checked against a leave-one-out reference built from at least
two high-tier faces spanning two tracklets; when that evidence is missing the
asset is admitted as ``identity_unverified`` instead of being rejected or
silently trusted.  Deduplication keys on near-identical image evidence
(perceptual hash); identity-embedding similarity only assists a borderline
hash match, so distinct poses/expressions of the same person survive.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCHEMA_VERSION = "character-asset-bank-v2"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _number(value: Any, fallback: float = 0.0) -> float:
    """Parse optional upstream scores without letting JSON null poison a run."""
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _illumination(image: np.ndarray) -> float:
    mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    return max(0.0, 1.0 - abs(mean - 128.0) / 128.0)


def _phash(image: np.ndarray) -> str:
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32))
    low = cv2.dct(gray.astype(np.float32))[:8, :8]
    bits = (low > np.median(low[1:])).flatten()
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


@dataclass
class SourceRef:
    shot_id: str
    tracklet_id: str
    frame_index: Optional[int]
    observation_id: str
    source_path: str


@dataclass
class AssetRecord:
    asset_id: str
    kind: str
    view: str
    crop_type: Optional[str]
    expression: Optional[str]
    appearance_state_id: Optional[str]
    quality_score: float
    quality_components: Dict[str, float]
    source: SourceRef
    embedding: Optional[List[float]] = field(default=None, repr=False)
    perceptual_hash: str = ""
    registry_path: Optional[str] = None
    admission_reason: str = "accepted"

    def public_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("embedding", None)
        return value


@dataclass
class CharacterRecord:
    character_id: str
    source_candidate_id: str
    tracklet_ids: List[str]
    shot_ids: List[str]
    identity_confidence: float
    face_gallery: Dict[str, List[Dict[str, Any]]]
    body_gallery: Dict[str, Dict[str, List[Dict[str, Any]]]]
    appearance_states: List[Dict[str, Any]]
    descriptions: Dict[str, Any]
    representative_set: Dict[str, Optional[str]]
    coverage: Dict[str, Any]
    speaker_ids: List[str] = field(default_factory=list)
    voice_embedding: Optional[List[float]] = None
    audio_assets: List[Dict[str, Any]] = field(default_factory=list)
    speaking_turns: List[Dict[str, Any]] = field(default_factory=list)


def face_view(yaw: Optional[float]) -> str:
    if yaw is None: return "other"
    angle = abs(float(yaw))
    if angle <= 20: return "frontal"
    if angle <= 50: return "three_quarter"
    if angle <= 75: return "profile"
    return "other"


def face_score(raw: Dict[str, Any], image: np.ndarray) -> Tuple[float, Dict[str, float]]:
    height, width = image.shape[:2]
    detector = _number(raw.get("det_score"), _number(raw.get("quality_score")))
    resolution = min(min(height, width) / 160.0, 1.0)
    sharp = min(_sharpness(image) / 250.0, 1.0)
    yaw = raw.get("yaw")
    frontal = 0.55 if yaw is None else max(0.0, 1.0 - abs(float(yaw)) / 90.0)
    illumination = _illumination(image)
    components = {"detector": detector, "resolution": resolution,
                  "sharpness": sharp, "frontal": frontal,
                  "illumination": illumination, "occlusion": 1.0}
    score = .25*detector + .20*resolution + .20*sharp + .20*frontal + .10*illumination + .05
    return round(float(score), 6), {key: round(value, 6) for key, value in components.items()}


def body_classification(image: np.ndarray, raw: Dict[str, Any]) -> Tuple[str, str, float]:
    height, width = image.shape[:2]
    aspect = height / max(width, 1)
    crop_type = "full_body" if aspect >= 1.75 else "upper_body" if aspect >= 1.05 else "partial"
    observation = str(raw.get("observation_id", ""))
    view = str(raw.get("body_view", "unknown"))
    if view not in {"front", "side", "back", "unknown"}: view = "unknown"
    # A matched face supports front/side, but absence of a face never proves back.
    if view == "unknown" and raw.get("has_matching_face"):
        yaw = raw.get("matching_face_yaw")
        view = "front" if yaw is None or abs(float(yaw)) <= 40 else "side"
    completeness = {"full_body": 1.0, "upper_body": .72, "partial": .42}[crop_type]
    return crop_type, view, completeness


def body_score(raw: Dict[str, Any], image: np.ndarray,
               completeness: float) -> Tuple[float, Dict[str, float]]:
    height, width = image.shape[:2]
    resolution = min(math.sqrt(height*width) / 600.0, 1.0)
    sharp = min(_sharpness(image) / 300.0, 1.0)
    detector = _number(raw.get("grounding_score"), _number(raw.get("quality_score")))
    mask = cv2.imread(str(raw.get("rgba_path") or raw.get("rgba_crop_path") or ""), cv2.IMREAD_UNCHANGED)
    mask_score = 1.0
    if mask is not None and mask.ndim == 3 and mask.shape[2] == 4:
        alpha = mask[:, :, 3] > 16
        mask_score = min(float(alpha.mean()) / .55, 1.0)
    components = {"detector": detector, "resolution": resolution,
                  "sharpness": sharp, "completeness": completeness,
                  "mask": mask_score, "illumination": _illumination(image)}
    score = (.20*detector + .15*resolution + .15*sharp + .25*completeness
             + .15*mask_score + .10*components["illumination"])
    return round(float(score), 6), {key: round(value, 6) for key, value in components.items()}


class AssetSelector:
    """Quality/diversity gate with bounded slot competition."""

    def __init__(self, face_top_k: int, body_top_k: int, min_face_quality: float,
                 min_body_quality: float, phash_distance: int, embedding_duplicate: float):
        self.face_top_k = face_top_k; self.body_top_k = body_top_k
        self.min_face_quality = min_face_quality; self.min_body_quality = min_body_quality
        self.phash_distance = phash_distance; self.embedding_duplicate = embedding_duplicate
        self.slots: Dict[Tuple[str, ...], List[AssetRecord]] = {}
        self.rejected: List[Dict[str, Any]] = []

    def _slot(self, asset: AssetRecord) -> Tuple[str, ...]:
        if asset.kind == "face": return ("face", asset.view)
        return ("body", asset.view, asset.crop_type or "partial", asset.appearance_state_id or "APP_UNKNOWN")

    def _duplicate(self, left: AssetRecord, right: AssetRecord) -> bool:
        # Near-duplicate image evidence (perceptual hash) is the primary rule.
        # Identity-embedding similarity alone only shows the same person, which
        # must not squeeze a different pose or expression out of the slot.
        if not (left.perceptual_hash and right.perceptual_hash):
            return False
        distance = _hamming(left.perceptual_hash, right.perceptual_hash)
        if distance <= self.phash_distance:
            return True
        if distance <= 2 * self.phash_distance and left.embedding and right.embedding:
            return float(np.dot(_unit(left.embedding), _unit(right.embedding))) >= self.embedding_duplicate
        return False

    def try_add(self, character_id: str, asset: AssetRecord) -> bool:
        threshold = self.min_face_quality if asset.kind == "face" else self.min_body_quality
        if asset.quality_score < threshold:
            self.rejected.append({"asset_id": asset.asset_id, "reason": "quality_gate",
                                  "quality_score": asset.quality_score}); return False
        key = self._slot(asset); existing = self.slots.setdefault(key, [])
        duplicates = [item for item in existing if self._duplicate(asset, item)]
        if duplicates:
            best = max(duplicates, key=lambda item: item.quality_score)
            if asset.quality_score <= best.quality_score:
                self.rejected.append({"asset_id": asset.asset_id, "reason": "duplicate_weaker"}); return False
            existing.remove(best)
            self.rejected.append({"asset_id": best.asset_id, "reason": "replaced_by_better_duplicate"})
        limit = self.face_top_k if asset.kind == "face" else self.body_top_k
        existing.append(asset); existing.sort(key=lambda item: (-item.quality_score, item.asset_id))
        if len(existing) > limit:
            loser = existing.pop()
            self.rejected.append({"asset_id": loser.asset_id, "reason": "slot_competition"})
            return loser.asset_id != asset.asset_id
        return True

    def assets(self) -> List[AssetRecord]:
        return sorted((item for values in self.slots.values() for item in values),
                      key=lambda item: (item.kind, item.view, item.crop_type or "", -item.quality_score))


def _observation_owner(tracklets: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {observation: item["tracklet_id"] for item in tracklets
            for observation in item.get("observation_ids", [])}


def _source(raw: Dict[str, Any], owner: Dict[str, str], members: Sequence[str]) -> SourceRef:
    observation = str(raw.get("observation_id", ""))
    tracklet = owner.get(observation, "")
    if not tracklet:
        shot = observation.split("_F", 1)[0]
        matching = [item for item in members if item.startswith(shot + "_T")]
        tracklet = matching[0] if len(matching) == 1 else "UNKNOWN"
    shot = re.sub(r"_T\d+$", "", tracklet) if tracklet != "UNKNOWN" else observation.split("_F", 1)[0]
    frame = raw.get("frame_index")
    if frame is None:
        match = re.search(r"_F(\d+)", observation); frame = int(match.group(1)) if match else None
    path = str(raw.get("asset_path") or raw.get("crop_path") or "")
    return SourceRef(shot, tracklet, int(frame) if frame is not None else None, observation, path)


def _appearance_for(tracklet_id: str, states: Sequence[Dict[str, Any]]) -> Optional[str]:
    for state in states:
        if tracklet_id in state.get("tracklet_ids", []): return str(state.get("state_id"))
    return None


def _face_support(faces: Sequence[Dict[str, Any]],
                  owner: Dict[str, str]) -> List[Dict[str, Any]]:
    """High-tier faces forming the identity reference, with tracklet provenance."""
    support = []
    for index, item in enumerate(faces):
        if item.get("embedding") and item.get("quality_tier") == "high":
            support.append({"index": index,
                            "observation_id": str(item.get("observation_id") or ""),
                            "embedding": _unit(item["embedding"]),
                            "tracklet_id": owner.get(str(item.get("observation_id") or ""), "")})
    return support


def _face_identity_ok(raw: Dict[str, Any], support: Sequence[Dict[str, Any]],
                      threshold: float) -> Tuple[bool, str]:
    """Check one face against the identity reference, excluding the face itself.

    A reference built from fewer than two high-tier faces, or from faces of a
    single tracklet, is too weak to prove or disprove identity: the asset is
    admitted but flagged ``identity_unverified`` rather than silently treated
    as purity-checked.
    """
    embedding = raw.get("embedding")
    if not embedding:
        return True, "identity_unverified"
    distinct_tracklets = {item["tracklet_id"] for item in support}
    if len(support) < 2 or len(distinct_tracklets) < 2:
        return True, "identity_unverified"
    observation_id = str(raw.get("observation_id") or "")
    reference = [item["embedding"] for item in support
                 if item["observation_id"] != observation_id]
    if not reference:
        return True, "identity_unverified"
    prototype = _unit(np.mean(reference, axis=0))
    ok = float(np.dot(_unit(embedding), prototype)) >= threshold
    return ok, "identity_verified" if ok else "face_identity_purity"


def _description(character: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    supplied = override.get(character["character_id"], {})
    return {"gender_presentation": supplied.get("gender_presentation", "unknown"),
            "approx_age": supplied.get("approx_age", "unknown"),
            "hair": supplied.get("hair", "unknown"),
            "stable_face_traits": supplied.get("stable_face_traits", "unknown"),
            "body_traits": supplied.get("body_traits", "unknown"),
            "source": "provided" if supplied else "not_inferred_without_evidence"}


def _representatives(assets: Sequence[AssetRecord]) -> Dict[str, Optional[str]]:
    def best(predicate: Any) -> Optional[str]:
        found = [item for item in assets if predicate(item)]
        return max(found, key=lambda item: item.quality_score).registry_path if found else None
    return {"primary_face": (best(lambda a: a.kind == "face" and a.view == "frontal")
                             or best(lambda a: a.kind == "face")),
            "primary_full_body": best(lambda a: a.kind == "body" and a.crop_type == "full_body"),
            "primary_profile": best(lambda a: a.kind == "face" and a.view == "profile"),
            "primary_back_body": best(lambda a: a.kind == "body" and a.view == "back"),
            "primary_appearance": best(lambda a: a.kind == "body")}


def _coverage(assets: Sequence[AssetRecord], appearance_count: int) -> Dict[str, Any]:
    checks = {"frontal_face": any(a.kind == "face" and a.view == "frontal" for a in assets),
              "profile_face": any(a.kind == "face" and a.view == "profile" for a in assets),
              "full_body": any(a.kind == "body" and a.crop_type == "full_body" for a in assets),
              "upper_body": any(a.kind == "body" and a.crop_type == "upper_body" for a in assets),
              "back_body": any(a.kind == "body" and a.view == "back" for a in assets),
              "multiple_appearances": appearance_count > 1}
    return {**checks, "covered": sum(checks.values()), "total": len(checks),
            "ratio": round(sum(checks.values()) / len(checks), 6)}


def _copy_assets(root: Path, character_id: str, assets: Sequence[AssetRecord]) -> None:
    for asset in assets:
        source = Path(asset.source.source_path)
        if not source.is_file(): continue
        parts = [asset.kind, asset.view]
        if asset.kind == "body": parts += [asset.crop_type or "partial", asset.appearance_state_id or "APP_UNKNOWN"]
        destination = root.joinpath(*parts, f"{_safe_name(asset.asset_id)}{source.suffix.lower() or '.jpg'}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination); asset.registry_path = str(destination)


def build_character(character: Dict[str, Any], candidate: Dict[str, Any], tracklets: Sequence[Dict[str, Any]],
                    output: Path, args: argparse.Namespace, descriptions: Dict[str, Any]) -> Tuple[CharacterRecord, Dict[str, Any]]:
    members = character.get("tracklet_ids", []); owner = _observation_owner(tracklets)
    states = candidate.get("appearance_states", character.get("appearance_states", []))
    states = [{**state, "source_state_id": state.get("state_id"),
               "state_id": re.sub(r"^CAND_CHR\d+", character["character_id"],
                                  str(state.get("state_id", "APP_UNKNOWN")))} for state in states]
    selector = AssetSelector(args.face_top_k, args.body_top_k, args.min_face_quality,
                             args.min_body_quality, args.phash_distance, args.embedding_duplicate)
    face_raw = candidate.get("face_gallery", [])
    face_support = _face_support(face_raw, owner)
    face_by_obs = {item.get("observation_id"): item for item in face_raw}
    body_features = {item.get("observation_id"): item for tracklet in tracklets
                     if tracklet.get("tracklet_id") in members
                     for item in tracklet.get("body_features", [])}
    serial = 0
    for raw in face_raw:
        path = Path(str(raw.get("asset_path") or "")); image = cv2.imread(str(path))
        if image is None: selector.rejected.append({"source_path": str(path), "reason": "unreadable"}); continue
        identity_ok, identity_verdict = _face_identity_ok(raw, face_support,
                                                          args.face_purity_threshold)
        if not identity_ok:
            selector.rejected.append({"source_path": str(path), "reason": "face_identity_purity"}); continue
        serial += 1; source = _source(raw, owner, members); score, components = face_score(raw, image)
        asset = AssetRecord(f"{character['character_id']}_FACE_{serial:04d}", "face", face_view(raw.get("yaw")),
                            None, str(raw.get("expression", "other")), _appearance_for(source.tracklet_id, states),
                            score, components, source, raw.get("embedding"), _phash(image),
                            admission_reason=identity_verdict)
        selector.try_add(character["character_id"], asset)
    for raw_original in candidate.get("body_gallery", []):
        raw = dict(raw_original); face = face_by_obs.get(raw.get("observation_id"))
        feature = body_features.get(raw.get("observation_id"), {})
        raw["embedding"] = feature.get("embedding")
        raw["has_matching_face"] = face is not None
        raw["matching_face_yaw"] = face.get("yaw") if face else None
        path = Path(str(raw.get("crop_path") or "")); image = cv2.imread(str(path))
        if image is None: selector.rejected.append({"source_path": str(path), "reason": "unreadable"}); continue
        crop_type, view, completeness = body_classification(image, raw)
        score, components = body_score(raw, image, completeness); serial += 1
        source = _source(raw, owner, members)
        asset = AssetRecord(f"{character['character_id']}_BODY_{serial:04d}", "body", view, crop_type,
                            None, _appearance_for(source.tracklet_id, states), score, components, source,
                            raw.get("embedding"), _phash(image))
        selector.try_add(character["character_id"], asset)
    assets = selector.assets(); root = output / "characters" / character["character_id"]
    _copy_assets(root, character["character_id"], assets)
    face_gallery: Dict[str, List[Dict[str, Any]]] = {}
    body_gallery: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for asset in assets:
        if asset.kind == "face": face_gallery.setdefault(asset.view, []).append(asset.public_dict())
        else: body_gallery.setdefault(asset.view, {}).setdefault(asset.crop_type or "partial", []).append(asset.public_dict())
    representatives = _representatives(assets)
    representative_dir = root / "representative"; representative_dir.mkdir(parents=True, exist_ok=True)
    for role, source_name in list(representatives.items()):
        if not source_name: continue
        source_path = Path(source_name); destination = representative_dir / f"{role}{source_path.suffix}"
        shutil.copy2(source_path, destination); representatives[role] = str(destination)
    record = CharacterRecord(character["character_id"], str(character.get("source_candidate_id", "")),
                             list(members), sorted({re.sub(r"_T\d+$", "", item) for item in members}),
                             float(character.get("identity_confidence", 0.0)), face_gallery, body_gallery,
                             states, _description(character, descriptions), representatives,
                             _coverage(assets, len(states)))
    _write(root / "character.json", asdict(record))
    _write(root / "assets.json", {"schema_version": SCHEMA_VERSION,
                                    "assets": [item.public_dict() for item in assets],
                                    "rejected_assets": selector.rejected})
    _write(root / "tracklets.json", {"tracklet_ids": members})
    _write(root / "appearance_states.json", {"appearance_states": states})
    reasons: Dict[str, int] = {}
    for item in selector.rejected: reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return record, {"accepted": len(assets), "rejected": len(selector.rejected),
                    "rejection_reasons": reasons,
                    "identity_unverified_count": sum(
                        asset.admission_reason == "identity_unverified" for asset in assets),
                    "average_quality": round(float(np.mean([a.quality_score for a in assets] or [0.])), 6),
                    "estimated_asset_purity": round(1.0 - reasons.get("face_identity_purity", 0)
                                                    / max(len(assets) + len(selector.rejected), 1), 6)}


class CharacterAssetRegistry:
    """Read-only downstream API over an exported bank."""
    def __init__(self, root: str | Path):
        self.root = Path(root); payload = _read(self.root / "registry.json")
        self.characters = {item["character_id"]: item for item in payload.get("characters", [])}
    def get_character(self, character_id: str) -> Dict[str, Any]: return self.characters[character_id]
    def get_face_assets(self, character_id: str, view: Optional[str] = None) -> List[Dict[str, Any]]:
        gallery = self.get_character(character_id)["face_gallery"]
        return list(gallery.get(view, [])) if view else [a for values in gallery.values() for a in values]
    def get_body_assets(self, character_id: str, view: Optional[str] = None,
                        crop_type: Optional[str] = None) -> List[Dict[str, Any]]:
        gallery = self.get_character(character_id)["body_gallery"]; result = []
        for current_view, types in gallery.items():
            if view and current_view != view: continue
            for current_type, assets in types.items():
                if not crop_type or current_type == crop_type: result.extend(assets)
        return result
    def get_primary_assets(self, character_id: str) -> Dict[str, Optional[str]]:
        return self.get_character(character_id)["representative_set"]
    def get_appearance_states(self, character_id: str) -> List[Dict[str, Any]]:
        return self.get_character(character_id)["appearance_states"]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    identity_root = Path(args.identity).expanduser().resolve()
    registry = _read(identity_root / "registry.json")
    features = _read(identity_root / "tracklet_features.json").get("tracklets", [])
    candidates = {item["candidate_id"]: item for item in registry.get("candidates", [])}
    descriptions = _read(Path(args.descriptions).expanduser().resolve()) if args.descriptions else {}
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    # Character ids are renumbered on every run, so any leftover folder from a
    # previous clustering would inject another person's crops into a gallery.
    characters_dir = output / "characters"
    if characters_dir.is_dir():
        shutil.rmtree(characters_dir)
    records, stats = [], {}
    for character in registry.get("characters", []):
        candidate_id = character.get("source_candidate_id")
        if candidate_id not in candidates: raise ValueError(f"missing candidate {candidate_id}")
        record, character_stats = build_character(character, candidates[candidate_id], features,
                                                   output, args, descriptions)
        records.append(asdict(record)); stats[record.character_id] = character_stats
    accepted = sum(item["accepted"] for item in stats.values()); rejected = sum(item["rejected"] for item in stats.values())
    duplicate_rejections = sum(item["rejection_reasons"].get("duplicate_weaker", 0)
                               + item["rejection_reasons"].get("replaced_by_better_duplicate", 0)
                               for item in stats.values())
    coverage = [item["coverage"]["ratio"] for item in records]
    summary = {"schema_version": SCHEMA_VERSION, "status": "ok", "source_identity": str(identity_root),
               "character_count": len(records), "asset_count": accepted, "rejected_asset_count": rejected,
               "duplicate_rate": round(duplicate_rejections / max(accepted + rejected, 1), 6),
               "average_coverage": round(float(np.mean(coverage or [0.])), 6),
               "characters": records, "character_stats": stats}
    _write(output / "registry.json", summary); return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 visual character asset-bank maintenance")
    parser.add_argument("--identity", required=True, help="Stage-3 identity output directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--descriptions", help="Optional reviewed stable-description JSON keyed by CHR ID")
    parser.add_argument("--face-top-k", type=int, default=5)
    parser.add_argument("--body-top-k", type=int, default=5)
    parser.add_argument("--min-face-quality", type=float, default=.45)
    parser.add_argument("--min-body-quality", type=float, default=.45)
    parser.add_argument("--phash-distance", type=int, default=5)
    parser.add_argument("--embedding-duplicate", type=float, default=.97)
    parser.add_argument("--face-purity-threshold", type=float, default=.35)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in ("status", "character_count", "asset_count",
                                                    "rejected_asset_count", "duplicate_rate")}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
