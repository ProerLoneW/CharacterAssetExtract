"""Pipeline router: detect chain type and provide a uniform API over both
the new ``global_align_pipeline`` chain and the legacy stage1.5→3→4→5 chain.

All public functions return plain ``dict`` payloads that the HTTP handlers
in :mod:`asset_viewer.server` can serialise directly.  All paths in those
payloads are **relative to the workspace root** so the media proxy can
authenticate them against the workspace boundary.

The router deliberately carries no in-process state beyond the workspace
path itself; server-side caching is handled in ``server.py``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from . import parsers as P

LOGGER = logging.getLogger("asset_viewer.router")

CHAIN_NEW = "new"          # global_align_pipeline (global_tracklet_aligner)
CHAIN_LEGACY = "legacy"    # stage1.5 → 3 → 4 → 5
CHAIN_HYBRID = "hybrid"    # both chains present
CHAIN_NONE = "none"        # no recognisable layout

STAGE_TRACKLETS = "tracklets"
STAGE_ALIGNMENT = "alignment"
STAGE_MATERIALIZATION = "materialization"
STAGE_ASSET_BANK = "asset_bank"
STAGE_CHARACTERS = "characters"
STAGE_VOICE = "voice"
STAGE_GLOBAL_CLIPS = "global_clips"
STAGE_SHOTS = "shots"
STAGE_CAPTIONS_SHOT = "captions_shot"
STAGE_CAPTIONS_GLOBAL = "captions_global"
STAGE_CAPTIONS_SEGMENT = "captions_segment"
STAGE_SEGMENTS = "segments"
STAGE_SCENE = "scene"
STAGE_SCENE_OPT = "scene_opt"
STAGE_KEY_OBJECT = "key_object"
STAGE_I2I_BUNDLES = "i2i_bundles"
STAGE_I2I_TEST = "i2i_test"
STAGE_TTS_TEST = "tts_test"
STAGE_ARTIFACTS = "artifacts"

STAGE_LABELS = {
    STAGE_TRACKLETS: "Stage 2 · Tracklets (per-shot)",
    STAGE_ALIGNMENT: "Stage 1.5 · Global Alignment",
    STAGE_MATERIALIZATION: "Stage 1.5 · Materialization",
    STAGE_ASSET_BANK: "Stage 4 · Asset Bank (CHR)",
    STAGE_CHARACTERS: "Stage 5 · Characters (scope/S#)",
    STAGE_VOICE: "Voice Assets",
    STAGE_GLOBAL_CLIPS: "全局拼接片段 (global_clips)",
    STAGE_SHOTS: "Shot 切片与索引",
    STAGE_CAPTIONS_SHOT: "Shot Caption",
    STAGE_CAPTIONS_GLOBAL: "Global Caption",
    STAGE_CAPTIONS_SEGMENT: "Segment Caption 规划",
    STAGE_SEGMENTS: "Segment 成片",
    STAGE_SCENE: "场景切分索引 (scene)",
    STAGE_SCENE_OPT: "场景精排索引 (scene_opt)",
    STAGE_KEY_OBJECT: "关键物体观测 (key_object)",
    STAGE_I2I_BUNDLES: "Image-to-Image 任务包 (i2i_bundles)",
    STAGE_I2I_TEST: "I2I 模型效果对比",
    STAGE_TTS_TEST: "TTS 模型效果对比",
    STAGE_ARTIFACTS: "全部产物文件",
}


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect_chain(workspace: Path) -> str:
    """Identify which pipeline chain produced *workspace*."""
    has_new = (workspace / "global_alignment" / "global_tracklet_alignment.jsonl").is_file()
    has_legacy = (workspace / "assets" / "visual_person_v2" / "asset_bank" / "registry.json").is_file()
    if has_new and has_legacy:
        return CHAIN_HYBRID
    if has_new:
        return CHAIN_NEW
    if has_legacy:
        return CHAIN_LEGACY
    return CHAIN_NONE


@dataclass
class WorkspaceSummary:
    name: str                 # basename, e.g. "movie_full_pipeline_20260915_global_align_test"
    path: str                 # absolute path on disk
    chain: str                # new | legacy | hybrid | none
    stages: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)  # global_0040 .. global_0042
    has_tracklets: bool = False


def list_workspaces(root: Path) -> list[WorkspaceSummary]:
    """Discover ``movie_full_pipeline_*`` workspaces under *root*.

    The viewer ignores workspaces without a recognisable chain so empty /
    in-progress runs do not clutter the picker.
    """
    if not root.is_dir():
        return []
    out: list[WorkspaceSummary] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("movie_full_pipeline_"):
            continue
        chain = detect_chain(child)
        if chain == CHAIN_NONE:
            continue
        summary = WorkspaceSummary(name=child.name, path=str(child.resolve()), chain=chain)
        summary.stages = list_stages(child)
        summary.scopes = list_scopes(child)
        summary.has_tracklets = (child / "assets" / "visual_person_v2" / "tracklets").is_dir() \
            or (child / "tracklets").is_dir()
        out.append(summary)
    return out


def list_stages(workspace: Path) -> list[str]:
    """Return the subset of canonical stage names whose artifacts exist."""
    stages: list[str] = []
    if _has_global_clips_dir(workspace):
        stages.append(STAGE_GLOBAL_CLIPS)
    if (workspace / "middle" / "shots.jsonl").is_file():
        stages.append(STAGE_SHOTS)
    if (workspace / "captions" / "shot" / "shot_captions.jsonl").is_file():
        stages.append(STAGE_CAPTIONS_SHOT)
    if (workspace / "captions" / "global" / "global_captions.jsonl").is_file():
        stages.append(STAGE_CAPTIONS_GLOBAL)
    if (workspace / "captions" / "segment" / "segment_captions.jsonl").is_file():
        stages.append(STAGE_CAPTIONS_SEGMENT)
    if (workspace / "segments" / "segments.jsonl").is_file():
        stages.append(STAGE_SEGMENTS)
    has_tracklets_dir = any(
        (workspace / base / "tracklets").is_dir()
        for base in ("assets/visual_person_v2", "assets/visual_person", "")
    )
    if has_tracklets_dir:
        stages.append(STAGE_TRACKLETS)
    if (workspace / "global_alignment" / "global_tracklet_alignment.jsonl").is_file():
        stages.append(STAGE_ALIGNMENT)
    if (workspace / "materialization_summary.json").is_file():
        stages.append(STAGE_MATERIALIZATION)
    if (workspace / "assets" / "visual_person_v2" / "asset_bank" / "registry.json").is_file():
        stages.append(STAGE_ASSET_BANK)
    if _has_characters_dir(workspace):
        stages.append(STAGE_CHARACTERS)
    if _has_voice_dir(workspace):
        stages.append(STAGE_VOICE)
    if _has_scene_dir(workspace, prefer_scene_opt=False):
        stages.append(STAGE_SCENE)
    if _has_scene_dir(workspace, prefer_scene_opt=True):
        stages.append(STAGE_SCENE_OPT)
    if _has_key_object_dir(workspace):
        stages.append(STAGE_KEY_OBJECT)
    if _has_i2i_bundles_dir(workspace):
        stages.append(STAGE_I2I_BUNDLES)
    if (workspace / "i2i_test").is_dir():
        stages.append(STAGE_I2I_TEST)
    if (workspace / "tts_test" / "outputs").is_dir():
        stages.append(STAGE_TTS_TEST)
    # The catch-all browser is intentionally last: purpose-built views above
    # are easier to read, while this guarantees no pipeline output is hidden.
    stages.append(STAGE_ARTIFACTS)
    return stages


def list_scopes(workspace: Path) -> list[str]:
    """Return ``global_NNNN`` scopes present in any characters layout."""
    out: set[str] = set()
    candidates: Iterable[Path] = (
        workspace / "characters",                                # new chain
        workspace / "characters" / "characters",                 # legacy chain
    )
    for base in candidates:
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("global_"):
                out.add(child.name)
    return sorted(out)


# --------------------------------------------------------------------------- #
# per-stage entity listing
# --------------------------------------------------------------------------- #
def list_entities(workspace: Path, stage: str) -> list[dict[str, Any]]:
    """Return the entities available in *stage* for the picker."""
    if stage == STAGE_SHOTS:
        return _list_shot_entities(workspace)

    if stage == STAGE_CAPTIONS_SHOT:
        return _list_caption_entities(workspace, "shot")

    if stage == STAGE_CAPTIONS_GLOBAL:
        return _list_caption_entities(workspace, "global")

    if stage == STAGE_CAPTIONS_SEGMENT:
        return _list_caption_entities(workspace, "segment")

    if stage == STAGE_SEGMENTS:
        return _list_segment_entities(workspace)

    if stage == STAGE_TRACKLETS:
        out: list[dict[str, Any]] = []
        shots_root = _shots_root(workspace)
        if shots_root and shots_root.is_dir():
            for child in sorted(shots_root.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "tracklet_signatures.json").is_file():
                    continue
                try:
                    payload = P._safe_read_json(child / "tracklet_signatures.json") or {}
                    n = len(payload.get("tracklet_signatures", []) or [])
                except Exception:
                    n = 0
                out.append({
                    "kind": "shot",
                    "id": child.name,
                    "label": child.name,
                    "tracklet_count": n,
                })
        return out

    if stage == STAGE_ALIGNMENT:
        out = []
        for row in P.parse_global_alignment(workspace):
            out.append({
                "kind": "global",
                "id": row.scope_id,
                "label": row.scope_id,
                "tracklet_count": row.tracklet_count,
                "assigned": row.assigned_tracklet_count,
                "status": row.status,
            })
        return out

    if stage == STAGE_MATERIALIZATION:
        return [
            {
                "kind": "character",
                "id": r.scope_subject,
                "label": r.scope_subject,
                "tracklet_count": r.tracklet_count,
                "face_copied": r.face_copied,
                "body_copied": r.body_copied,
                "voice_copied": r.voice_copied,
            }
            for r in P.parse_materialization(workspace)
        ]

    if stage == STAGE_ASSET_BANK:
        bank_chars, _ = P.parse_asset_bank_registry(workspace)
        return [
            {
                "kind": "chr",
                "id": c.character_id,
                "label": c.character_id,
                "tracklet_count": len(c.tracklet_ids),
                "identity_confidence": c.identity_confidence,
                "shot_count": len(c.shot_ids),
            }
            for c in bank_chars
        ]

    if stage == STAGE_CHARACTERS:
        return _list_character_entities(workspace)

    if stage == STAGE_VOICE:
        return _list_voice_entities(workspace)

    if stage == STAGE_GLOBAL_CLIPS:
        return _list_global_clip_entities(workspace)

    if stage == STAGE_SCENE:
        return _list_scene_entities(workspace, scene_opt=False)

    if stage == STAGE_SCENE_OPT:
        return _list_scene_entities(workspace, scene_opt=True)

    if stage == STAGE_KEY_OBJECT:
        return _list_key_object_entities(workspace)

    if stage == STAGE_I2I_BUNDLES:
        return _list_i2i_bundle_entities(workspace)

    if stage == STAGE_I2I_TEST:
        return _list_i2i_test_entities(workspace)

    if stage == STAGE_TTS_TEST:
        return _list_tts_test_entities(workspace)

    if stage == STAGE_ARTIFACTS:
        return list_artifact_groups(workspace)

    return []


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _shot_scope_map(workspace: Path) -> dict[int, str]:
    """Map shot numbers to their owning global scope."""
    out: dict[int, str] = {}
    for row in P._load_jsonl(workspace / "middle" / "global_clips.jsonl"):
        try:
            scope = f"global_{int(row.get('global_id')):04d}"
        except (TypeError, ValueError):
            continue
        for shot_id in row.get("shot_ids", []) or []:
            try:
                out[int(shot_id)] = scope
            except (TypeError, ValueError):
                continue
    return out


def _list_shot_entities(workspace: Path) -> list[dict[str, Any]]:
    scope_map = _shot_scope_map(workspace)
    captions: dict[int, dict[str, Any]] = {}
    caption_dir = workspace / "captions" / "shot" / "shot_captions"
    if caption_dir.is_dir():
        for path in caption_dir.glob("shot_*.json"):
            payload = P._safe_read_json(path) or {}
            try:
                captions[int(payload.get("shot_id"))] = payload
            except (TypeError, ValueError):
                continue
    out: list[dict[str, Any]] = []
    for row in P._load_jsonl(workspace / "middle" / "shots.jsonl"):
        try:
            shot_num = int(row.get("shot_id"))
        except (TypeError, ValueError):
            continue
        shot_id = f"shot_{shot_num:05d}"
        media = workspace / "middle" / "shot_clips" / f"SHOT_{shot_num:05d}.mp4"
        caption = captions.get(shot_num, {}).get("shot_caption", {}) or {}
        summary = caption.get("shot_caption", {}) or {}
        out.append({
            "kind": "shot",
            "id": shot_id,
            "label": shot_id,
            "scope": scope_map.get(shot_num, ""),
            "duration_sec": row.get("duration_sec"),
            "time_range_sec": row.get("time_range_sec", []),
            "boundary_type": row.get("boundary_type", ""),
            "description": summary.get("shot_description", ""),
            "scene_description": summary.get("shot_scene_description", ""),
            "visible_count": caption.get("visible_count", 0),
            "media_path": _rel(media, workspace) if media.exists() else "",
            "size": _safe_size(media),
        })
    return out


def _caption_file(workspace: Path, level: str, item_id: str) -> Path:
    digits = "".join(ch for ch in item_id if ch.isdigit())
    number = int(digits) if digits else -1
    if level == "shot":
        return workspace / "captions" / "shot" / "shot_captions" / f"shot_{number:05d}.json"
    if level == "global":
        return workspace / "captions" / "global" / "global_captions" / f"global_{number:05d}.json"
    return workspace / "captions" / "segment" / "segments" / f"global_{number:05d}.json"


def _normalise_caption(record: dict[str, Any], level: str) -> dict[str, Any]:
    if level == "shot":
        model = record.get("shot_caption", {}) or {}
        summary = model.get("shot_caption", {}) or {}
        subjects = model.get("subjects", []) or []
        dialogue = model.get("dialogue", []) or []
        events = model.get("visual_events", []) or []
        segments: list[dict[str, Any]] = []
        description = summary.get("shot_description", "")
        scene_description = summary.get("shot_scene_description", "")
    elif level == "global":
        model = record.get("global_caption", {}) or {}
        summary = model.get("global_caption", {}) or {}
        subjects = model.get("subject_registry", []) or []
        dialogue = model.get("asr_speaker_attributions", []) or []
        events = model.get("global_event_timeline", []) or []
        segments = []
        description = summary.get("global_description", "")
        scene_description = summary.get("global_scene_description", "")
    else:
        model = record.get("segment_caption", {}) or {}
        summary = model.get("global_caption", {}) or {}
        # Segment planning embeds the complete global-caption record rather
        # than only its prose summary; peel those wrappers when present.
        for _ in range(3):
            if not isinstance(summary, dict):
                summary = {}
                break
            candidate = summary.get("global_caption")
            if not isinstance(candidate, dict):
                break
            summary = candidate
        global_model = record.get("global_caption", {}) or {}
        subjects = global_model.get("subject_registry", []) or model.get("subject_registry", []) or []
        segments = model.get("segments", []) or []
        dialogue = []
        for segment in segments:
            dialogue.extend(segment.get("dialogue", []) or [])
        events = segments
        description = summary.get("global_description", "")
        scene_description = summary.get("global_scene_description", "")
    return {
        "description": description,
        "scene_description": scene_description,
        "overall_soundscape": summary.get("overall_soundscape", ""),
        "non_diegetic_music": summary.get("non_diegetic_music", ""),
        "annotation_confidence": model.get("annotation_confidence", ""),
        "subjects": subjects,
        "dialogue": dialogue,
        "events": events,
        "segments": segments,
    }


def _list_caption_entities(workspace: Path, level: str) -> list[dict[str, Any]]:
    if level == "shot":
        root = workspace / "captions" / "shot" / "shot_captions"
        paths = sorted(root.glob("shot_*.json")) if root.is_dir() else []
    elif level == "global":
        root = workspace / "captions" / "global" / "global_captions"
        paths = sorted(root.glob("global_*.json")) if root.is_dir() else []
    else:
        root = workspace / "captions" / "segment" / "segments"
        paths = sorted(root.glob("global_*.json")) if root.is_dir() else []
    out: list[dict[str, Any]] = []
    scope_map = _shot_scope_map(workspace)
    for path in paths:
        record = P._safe_read_json(path) or {}
        normal = _normalise_caption(record, level)
        if level == "shot":
            number = int(record.get("shot_id", -1))
            item_id = f"shot_{number:05d}"
            scope = scope_map.get(number, "")
        else:
            number = int(record.get("global_id", -1))
            item_id = f"global_{number:04d}"
            scope = item_id
        out.append({
            "kind": f"caption_{level}",
            "id": item_id,
            "label": item_id,
            "scope": scope,
            "description": normal["description"],
            "scene_description": normal["scene_description"],
            "duration_sec": record.get("duration_sec"),
            "subject_count": len(normal["subjects"]),
            "dialogue_count": len(normal["dialogue"]),
            "segment_count": len(normal["segments"]),
            "confidence": normal["annotation_confidence"],
            "success": record.get("success", False),
        })
    return out


def _list_segment_entities(workspace: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in P._load_jsonl(workspace / "segments" / "segments.jsonl"):
        try:
            global_id = int(row.get("global_id"))
            segment_id = int(row.get("segment_id"))
        except (TypeError, ValueError):
            continue
        item_id = f"global_{global_id:04d}_segment_{segment_id:03d}"
        media = workspace / "segments" / "clips" / f"1080P_{item_id}.mp4"
        out.append({
            "kind": "segment",
            "id": item_id,
            "label": item_id,
            "scope": f"global_{global_id:04d}",
            "segment_id": segment_id,
            "description": row.get("description", ""),
            "scene_description": row.get("scene_description", ""),
            "duration_sec": row.get("duration_sec"),
            "shot_count": len(row.get("shot_ids", []) or []),
            "dialogue_count": len(row.get("dialogue", []) or []),
            "event_phase": row.get("event_phase", ""),
            "media_path": _rel(media, workspace) if media.is_file() else "",
            "size": _safe_size(media),
        })
    return out


def _list_voice_entities(workspace: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = workspace / "voice_by_role_src"
    for status in ("valid", "invalid"):
        manifest = root / f"{status}_manifest.jsonl"
        for row in P._load_jsonl(manifest):
            global_id = row.get("global_id")
            try:
                scope = f"global_{int(global_id):04d}"
            except (TypeError, ValueError):
                scope = "global_unknown"
            speaker = str(row.get("speaker_id") or "unknown")
            asr_id = str(row.get("asr_id") or "unknown")
            rel = str(row.get("review_audio_relative_path") or "")
            media = root / rel if rel else root / status / scope / speaker / f"{asr_id}.wav"
            out.append({
                "kind": "voice_asset",
                "id": f"{status}:{scope}:{speaker}:{asr_id}",
                "label": asr_id,
                "status": status,
                "scope": scope,
                "speaker_id": speaker,
                "asr_id": asr_id,
                "text": row.get("text", ""),
                "time_range_sec": row.get("time_range_sec", []),
                "confidence": row.get("attribution_confidence", ""),
                "evidence": row.get("attribution_evidence", ""),
                "rejection_reasons": row.get("audio_asset_rejection_reasons", []) or [],
                "media_path": _rel(media, workspace) if media.exists() else "",
                "size": _safe_size(media),
                "signal_stats": row.get("signal_stats", {}) or {},
            })
    return out


def list_artifact_groups(workspace: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for dirpath, _, filenames in os.walk(workspace, followlinks=False):
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            try:
                rel = path.relative_to(workspace)
            except ValueError:
                continue
            group = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            item = groups.setdefault(group, {
                "kind": "artifact_group", "id": group, "label": group,
                "file_count": 0, "size": 0,
            })
            item["file_count"] += 1
            item["size"] += _safe_size(path)
    return sorted(groups.values(), key=lambda item: item["id"])


def list_artifacts(workspace: Path, prefix: str = "") -> dict[str, Any]:
    clean = Path(prefix.strip("/")) if prefix else Path()
    if ".." in clean.parts:
        return {"ok": False, "error": "invalid artifact prefix"}
    root = workspace / clean
    if not root.exists():
        return {"ok": False, "error": f"artifact prefix not found: {prefix}"}
    paths: list[Path] = []
    if root.is_file():
        paths = [root]
    else:
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            directory = Path(dirpath)
            paths.extend(directory / name for name in filenames)
    items: list[dict[str, Any]] = []
    for path in sorted(paths):
        rel = path.relative_to(workspace).as_posix()
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            kind = "image"
        elif suffix in (".mp4", ".webm", ".mov"):
            kind = "video"
        elif suffix in (".wav", ".mp3", ".m4a", ".flac"):
            kind = "audio"
        elif suffix in (".json", ".jsonl", ".txt", ".log", ".md", ".csv", ".yaml", ".yml"):
            kind = "text"
        else:
            kind = "file"
        items.append({
            "path": rel,
            "name": path.name,
            "group": rel.split("/", 1)[0] if "/" in rel else "(root)",
            "suffix": suffix or "(none)",
            "kind": kind,
            "size": _safe_size(path),
        })
    return {"ok": True, "prefix": prefix, "items": items, "count": len(items)}


def get_artifact_payload(workspace: Path, rel_path: str) -> dict[str, Any]:
    rel = Path(rel_path.strip("/"))
    if not rel.parts or ".." in rel.parts:
        return {"ok": False, "error": "invalid artifact path"}
    path = workspace / rel
    if not path.is_file():
        return {"ok": False, "error": f"artifact not found: {rel_path}"}
    suffix = path.suffix.lower()
    result: dict[str, Any] = {
        "ok": True, "path": rel.as_posix(), "filename": path.name,
        "size": _safe_size(path), "suffix": suffix,
    }
    if suffix == ".json":
        try:
            result["payload"] = json.loads(path.read_text(encoding="utf-8"))
            result["kind"] = "json"
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}
    elif suffix == ".jsonl":
        rows = P._load_jsonl(path)
        result.update({"kind": "jsonl", "rows": rows[:100],
                       "row_count": len(rows), "truncated": len(rows) > 100})
    elif suffix in (".txt", ".log", ".md", ".csv", ".yaml", ".yml"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        result.update({"kind": "text", "text": text[:1_000_000],
                       "truncated": len(text) > 1_000_000})
    else:
        result["kind"] = "media"
    return result


def get_stage_detail(workspace: Path, stage: str, item_id: str) -> dict[str, Any]:
    if stage == STAGE_I2I_TEST:
        rel = Path(item_id)
        if len(rel.parts) != 2 or ".." in rel.parts:
            return {"ok": False, "error": "invalid I2I test id"}
        image = workspace / "i2i_test" / rel
        metadata_path = Path(f"{image}.json")
        if not image.is_file() or not metadata_path.is_file():
            return {"ok": False, "error": f"I2I test result not found: {item_id}"}
        metadata = P._safe_read_json(metadata_path)
        if metadata is None:
            return {"ok": False, "error": f"invalid I2I metadata: {item_id}"}
        references = [
            {"filename": path.name, "path": _rel(path, workspace), "size": _safe_size(path)}
            for path in sorted((workspace / "i2i_test").glob("*"))
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        return {
            "ok": True, "stage": stage, "id": item_id,
            "model": image.parent.name,
            "image_path": _rel(image, workspace),
            "json_path": _rel(metadata_path, workspace),
            "references": references,
            "raw": metadata,
        }
    if stage == STAGE_TTS_TEST:
        rel = Path(item_id)
        if len(rel.parts) != 2 or ".." in rel.parts:
            return {"ok": False, "error": "invalid TTS test id"}
        run_dir = workspace / "tts_test" / "outputs" / rel
        metadata_path = run_dir / "meta.json"
        metadata = P._safe_read_json(metadata_path)
        if metadata is None:
            return {"ok": False, "error": f"TTS test result not found: {item_id}"}
        reference = run_dir / "reference_merged.wav"
        outputs = [path for path in sorted(run_dir.glob("*.wav")) if path.name != reference.name]
        return {
            "ok": True, "stage": stage, "id": item_id,
            "model": rel.parts[0],
            "reference_path": _rel(reference, workspace) if reference.is_file() else "",
            "outputs": [
                {"filename": path.name, "path": _rel(path, workspace), "size": _safe_size(path)}
                for path in outputs
            ],
            "json_path": _rel(metadata_path, workspace),
            "raw": metadata,
        }
    if stage == STAGE_SHOTS:
        for row in P._load_jsonl(workspace / "middle" / "shots.jsonl"):
            try:
                current_id = f"shot_{int(row.get('shot_id')):05d}"
            except (TypeError, ValueError):
                continue
            if current_id == item_id:
                number = int(row["shot_id"])
                media = workspace / "middle" / "shot_clips" / f"SHOT_{number:05d}.mp4"
                caption_path = _caption_file(workspace, "shot", item_id)
                caption = P._safe_read_json(caption_path) or {}
                return {"ok": True, "stage": stage, "id": item_id,
                        "media_path": _rel(media, workspace), "record": row,
                        "caption": _normalise_caption(caption, "shot"), "raw": caption}
    elif stage == STAGE_SEGMENTS:
        for row in P._load_jsonl(workspace / "segments" / "segments.jsonl"):
            try:
                current_id = f"global_{int(row.get('global_id')):04d}_segment_{int(row.get('segment_id')):03d}"
            except (TypeError, ValueError):
                continue
            if current_id == item_id:
                media = workspace / "segments" / "clips" / f"1080P_{current_id}.mp4"
                return {"ok": True, "stage": stage, "id": item_id,
                        "media_path": _rel(media, workspace), "record": row}
    else:
        level_by_stage = {
            STAGE_CAPTIONS_SHOT: "shot",
            STAGE_CAPTIONS_GLOBAL: "global",
            STAGE_CAPTIONS_SEGMENT: "segment",
        }
        level = level_by_stage.get(stage)
        if level:
            path = _caption_file(workspace, level, item_id)
            record = P._safe_read_json(path)
            if record is not None:
                digits = "".join(ch for ch in item_id if ch.isdigit())
                number = int(digits) if digits else -1
                if level == "shot":
                    media = workspace / "middle" / "shot_clips" / f"SHOT_{number:05d}.mp4"
                else:
                    media = workspace / "global_clips" / f"1080P_global_{number:04d}.mp4"
                return {"ok": True, "stage": stage, "level": level, "id": item_id,
                        "media_path": _rel(media, workspace),
                        "caption": _normalise_caption(record, level), "raw": record}
    return {"ok": False, "error": f"result not found: {stage}/{item_id}"}


def _list_character_entities(workspace: Path) -> list[dict[str, Any]]:
    """Walk both ``characters/<scope>/<S>`` and
    ``characters/characters/<scope>/<S>`` to enumerate (scope, S) pairs.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for layout_base in (workspace / "characters", workspace / "characters" / "characters"):
        if not layout_base.is_dir():
            continue
        for scope_dir in sorted(layout_base.iterdir()):
            if not scope_dir.is_dir() or not scope_dir.name.startswith("global_"):
                continue
            for subject_dir in sorted(scope_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                key = f"{scope_dir.name}:{subject_dir.name}"
                if key in seen:
                    continue
                seen.add(key)
                role = subject_dir / "role_character.json"
                payload = P._safe_read_json(role) if role.is_file() else None
                tracklet_count = 0
                if payload:
                    try:
                        tracklet_count = len(payload.get("tracklet_ids", []) or [])
                    except TypeError:
                        tracklet_count = 0
                face = _count_media_recursive(subject_dir / "visual" / "face", (".jpg", ".png"))
                body = _count_media_recursive(subject_dir / "visual" / "body", (".jpg", ".png"))
                voice = _count_media_recursive(subject_dir / "voice" / "valid", (".wav",))
                out.append({
                    "kind": "character",
                    "id": key,
                    "label": key,
                    "tracklet_count": tracklet_count,
                    "face_copied": face,
                    "body_copied": body,
                    "voice_copied": voice,
                    "schema_version": (payload or {}).get("schema_version", ""),
                })
    return out


# --------------------------------------------------------------------------- #
# character payload
# --------------------------------------------------------------------------- #
def get_character_payload(workspace: Path, scope_id: str,
                          subject_id: str) -> dict[str, Any]:
    """Return the full payload for a (scope, S#) character."""
    base = _character_dir_for(workspace, scope_id, subject_id)
    if base is None:
        return {"ok": False, "error": f"character not found: {scope_id}:{subject_id}"}

    role_path = base / "role_character.json"
    role = P._safe_read_json(role_path) or {}
    visual_dir = base / "visual"
    voice_dir = base / "voice"

    face_assets = _list_media_recursive(visual_dir / "face", (".jpg", ".png"))
    body_assets = _list_media_recursive(visual_dir / "body", (".jpg", ".png"))
    rgba_assets = _list_media_recursive(visual_dir / "rgba", (".png", ".jpg"))
    voice_assets = _list_media_recursive(voice_dir / "valid", (".wav",)) + \
        _list_media_recursive(voice_dir / "invalid", (".wav",))

    return {
        "ok": True,
        "scope_id": scope_id,
        "subject_id": subject_id,
        "role_character": role,
        "role_character_path": _rel(role_path, workspace),
        "faces": face_assets,
        "bodies": body_assets,
        "rgbas": rgba_assets,
        "voices": voice_assets,
        "character_dir": _rel(base, workspace),
    }


# --------------------------------------------------------------------------- #
# global / alignment payload
# --------------------------------------------------------------------------- #
def get_global_payload(workspace: Path, scope_id: str) -> dict[str, Any]:
    """Return one global clip's alignment row plus a stitched-video URL."""
    rows = P.parse_global_alignment(workspace)
    for row in rows:
        if row.scope_id == scope_id:
            stitched_rel, stitched_abs = _resolve_stitched_video(workspace, row.stitched_video)
            return {
                "ok": True,
                "scope_id": scope_id,
                "status": row.status,
                "tracklet_count": row.tracklet_count,
                "assigned_tracklet_count": row.assigned_tracklet_count,
                "stitched_video": stitched_rel,
                "stitched_video_exists": bool(stitched_abs and stitched_abs.is_file()),
                "tracklet_alignments": row.alignments,
                "unmatched_subjects": row.unmatched_subjects,
                "quality_issues": row.quality_issues,
                "summary": P.parse_global_alignment_summary(workspace),
            }
    return {"ok": False, "error": f"global scope not found: {scope_id}"}


def _resolve_stitched_video(workspace: Path, value: str | None) -> tuple[str, Path | None]:
    """Resolve the stitched video path that the upstream pipeline stores as
    a bare filename (e.g. ``global_0041_tracklets.mp4``) inside the
    ``global_alignment/stitched/`` directory.

    Returns ``(relative_path_for_media_proxy, absolute_path_on_disk)``.
    The relative path is always workspace-rooted so the media proxy can
    authenticate it via :func:`server._safe_within_workspace`.
    """
    if not value:
        return "", None
    candidate = Path(value)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        # Bare filename — try global_alignment/stitched/<value> first,
        # then the value as a workspace-relative path.
        candidates.append(workspace / "global_alignment" / "stitched" / value)
        candidates.append(workspace / value)
    for abs_path in candidates:
        if abs_path.is_file():
            try:
                return str(abs_path.resolve().relative_to(workspace.resolve())), abs_path.resolve()
            except ValueError:
                return str(abs_path), abs_path
    # Not found on disk — return the best-effort relative path so the UI
    # can still surface the filename.
    fallback = candidates[0] if candidates else workspace / value
    try:
        return str(fallback.resolve().relative_to(workspace.resolve())), None
    except ValueError:
        return value, None


# --------------------------------------------------------------------------- #
# tracklet payload
# --------------------------------------------------------------------------- #
def get_tracklet_payload(workspace: Path, shot_id: str,
                         tracklet_id: str) -> dict[str, Any]:
    """Return a single tracklet's gallery paths."""
    rows = P.parse_tracklet_signatures(workspace, shot_id)
    for row in rows:
        if row.tracklet_id == tracklet_id:
            shot_number = P._shot_number(shot_id) or 0
            shot_dir_name = f"SHOT_1080P_SHOT_{shot_number:05d}"
            shot_root = _shots_root(workspace)
            if shot_root is None:
                return {"ok": False, "error": "shots directory not found"}
            shot_abs = shot_root / shot_dir_name
            face_dir = shot_abs / "tracklet_assets" / tracklet_id / "face"
            body_dir = shot_abs / "tracklet_assets" / tracklet_id / "body"
            rgba_dir = shot_abs / "tracklet_assets" / tracklet_id / "rgba"
            return {
                "ok": True,
                "shot_id": shot_id,
                "shot_dir": _rel(shot_abs, workspace),
                "tracklet_id": tracklet_id,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "observation_count": row.observation_count,
                "face_count": row.face_count,
                "body_count": row.body_count,
                "best_face_quality": row.best_face_quality,
                "best_body_quality": row.best_body_quality,
                "faces": _list_media(face_dir, (".jpg", ".png")) if face_dir.is_dir() else [],
                "bodies": _list_media(body_dir, (".jpg", ".png")) if body_dir.is_dir() else [],
                "rgbas": _list_media(rgba_dir, (".png", ".jpg")) if rgba_dir.is_dir() else [],
                "overlay_video_rel": _rel(shot_abs / "tracklet_overlay.mp4", workspace),
                "overlay_video_exists": (shot_abs / "tracklet_overlay.mp4").is_file(),
            }
    return {"ok": False, "error": f"tracklet not found: {tracklet_id}"}


def list_shot_tracklets(workspace: Path, shot_id: str) -> list[dict[str, Any]]:
    """Lightweight per-shot tracklet summary for the global alignment page."""
    out: list[dict[str, Any]] = []
    rows = P.parse_tracklet_signatures(workspace, shot_id)
    for r in rows:
        out.append({
            "tracklet_id": r.tracklet_id,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "observation_count": r.observation_count,
            "face_count": r.face_count,
            "body_count": r.body_count,
            "best_face_quality": r.best_face_quality,
            "best_body_quality": r.best_body_quality,
        })
    return out


def list_global_tracklets(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    """Bulk: scope_id -> list of per-tracklet summaries for the alignment table."""
    out: dict[str, list[dict[str, Any]]] = {}
    rows = P.parse_global_alignment(workspace)
    for row in rows:
        entries: list[dict[str, Any]] = []
        for entry in row.alignments:
            tid = str(entry.get("tracklet_id", ""))
            shot_id = _extract_shot_id(tid)
            if shot_id is None:
                continue
            entries.append({
                "tracklet_id": tid,
                "shot_id": shot_id,
                "subject_id": str(entry.get("subject_id", "unknown")),
                "importance_tier": str(entry.get("importance_tier", "unknown")),
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
                "visual_cues": list(entry.get("visual_cues", []) or []),
                "reason": str(entry.get("reason", "")),
            })
        out[row.scope_id] = entries
    return out


# --------------------------------------------------------------------------- #
# new stage entity listing
# --------------------------------------------------------------------------- #
def _list_global_clip_entities(workspace: Path) -> list[dict[str, Any]]:
    """Enumerate ``*.mp4`` files in ``global_clips/``."""
    out: list[dict[str, Any]] = []
    root = workspace / "global_clips"
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".mp4":
            continue
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        out.append({
            "kind": "global_clip",
            "id": entry.stem,
            "label": entry.name,
            "filename": entry.name,
            "path": _rel(entry, workspace),
            "size": size,
        })
    return out


def _list_scene_entities(workspace: Path, scene_opt: bool) -> list[dict[str, Any]]:
    """Enumerate per-global entries in ``scene/`` or ``scene_opt_marginal/``."""
    out: list[dict[str, Any]] = []
    root = _scene_root(workspace, scene_opt=scene_opt)
    if root is None or not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        idx_path = entry / "scene_index.json"
        cand_path = entry / "candidate_scores.json"
        out.append({
            "kind": "scene",
            "id": entry.name,
            "label": entry.name,
            "scope_id": _scope_from_dir_name(entry.name),
            "scene_dir": _rel(entry, workspace),
            "scene_index_exists": idx_path.is_file(),
            "candidate_scores_exists": cand_path.is_file(),
        })
    return out


def _list_key_object_entities(workspace: Path) -> list[dict[str, Any]]:
    """Return the first 100 object observations from
    ``key_object/grounding/object_observations.jsonl`` with id + bbox + scope.
    """
    out: list[dict[str, Any]] = []
    obs_path = workspace / "key_object" / "grounding" / "object_observations.jsonl"
    if not obs_path.is_file():
        # fallback: read any jsonl inside key_object/grounding
        grounding = workspace / "key_object" / "grounding"
        if grounding.is_dir():
            for child in sorted(grounding.glob("*.jsonl")):
                obs_path = child
                break
    if not obs_path.is_file():
        return out
    rows = P._load_jsonl(obs_path)
    for idx, row in enumerate(rows):
        if idx >= 100:
            break
        cid = str(row.get("candidate_id") or row.get("id") or f"obs_{idx:04d}")
        scope = str(row.get("semantic_scope_id") or row.get("scope_id") or "")
        bbox = row.get("selected_bbox_xyxy") or row.get("mask_bbox_xyxy") or row.get("bbox") or {}
        canonical = str(row.get("canonical_name") or row.get("name") or "")
        out.append({
            "kind": "object_observation",
            "id": cid,
            "label": f"{cid} · {canonical}" if canonical else cid,
            "scope_id": scope,
            "canonical_name": canonical,
            "shot_id": str(row.get("shot_id") or ""),
            "bbox": bbox,
        })
    return out


def _list_i2i_bundle_entities(workspace: Path) -> list[dict[str, Any]]:
    """Enumerate ``*.json`` files under ``i2i_bundles/`` recursively."""
    out: list[dict[str, Any]] = []
    root = workspace / "i2i_bundles"
    if not root.is_dir():
        return out
    for entry in sorted(root.rglob("*.json")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(root)
        bundle_id = str(rel.with_suffix("")).replace(os.sep, "/")
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        # peek at task_type for quick display
        task_type = ""
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                task_type = str(payload.get("task_type", ""))
        except (OSError, json.JSONDecodeError):
            payload = None
        out.append({
            "kind": "i2i_bundle",
            "id": bundle_id,
            "label": bundle_id,
            "task_type": task_type,
            "filename": entry.name,
            "path": _rel(entry, workspace),
            "size": size,
        })
    return out


def _list_i2i_test_entities(workspace: Path) -> list[dict[str, Any]]:
    """Enumerate the copied I2I comparison references and generated PNGs."""
    root = workspace / "i2i_test"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            out.append({
                "kind": "i2i_test_reference",
                "id": entry.name,
                "label": entry.name,
                "image_path": _rel(entry, workspace),
                "size": _safe_size(entry),
            })
    for image in sorted(root.glob("*/*.png")):
        metadata_path = Path(f"{image}.json")
        metadata = P._safe_read_json(metadata_path) or {}
        request = metadata.get("request", {}) or {}
        response = metadata.get("response", {}) or {}
        usage = response.get("usage", {}) or {}
        rel = image.relative_to(root).as_posix()
        out.append({
            "kind": "i2i_test_result",
            "id": rel,
            "label": image.name,
            "model": image.parent.name,
            "task": request.get("task", ""),
            "part": str(request.get("task", "")).removeprefix("character_clothing_"),
            "character_label": request.get("character_label", ""),
            "model_version": response.get("model_version", ""),
            "total_tokens": usage.get("totalTokenCount"),
            "finish_reasons": response.get("finish_reasons", []) or [],
            "image_path": _rel(image, workspace),
            "json_path": _rel(metadata_path, workspace) if metadata_path.is_file() else "",
            "size": _safe_size(image),
        })
    return out


def _list_tts_test_entities(workspace: Path) -> list[dict[str, Any]]:
    """Enumerate model/speaker TTS comparison runs copied into the workspace."""
    root = workspace / "tts_test" / "outputs"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("*/*/meta.json")):
        run_dir = metadata_path.parent
        model_dir = run_dir.parent.name
        run_id = run_dir.name
        scope, _, speaker_id = run_id.rpartition("_")
        metadata = P._safe_read_json(metadata_path) or {}
        reference = run_dir / "reference_merged.wav"
        outputs = [path for path in sorted(run_dir.glob("*.wav")) if path.name != reference.name]
        out.append({
            "kind": "tts_test_result",
            "id": f"{model_dir}/{run_id}",
            "label": run_id,
            "model": model_dir,
            "model_name": metadata.get("model", model_dir),
            "scope": scope,
            "speaker_id": speaker_id,
            "reference_text": metadata.get("reference_text", ""),
            "synthesis_text": metadata.get("synthesis_text", ""),
            "reference_path": _rel(reference, workspace) if reference.is_file() else "",
            "output_paths": [
                {"filename": path.name, "path": _rel(path, workspace), "size": _safe_size(path)}
                for path in outputs
            ],
            "json_path": _rel(metadata_path, workspace),
        })
    return out


def _scope_from_dir_name(name: str) -> str:
    """``1080P_global_0040`` → ``global_0040``; pass-through if already prefixed."""
    if name.startswith("global_"):
        return name
    parts = name.split("_global_", 1)
    if len(parts) == 2:
        return f"global_{parts[1]}"
    return name


# --------------------------------------------------------------------------- #
# new stage payload getters
# --------------------------------------------------------------------------- #
def get_global_clip_payload(workspace: Path, scope: str) -> dict[str, Any]:
    """Return one global clip's mp4 path + metadata.

    *scope* may be ``global_0040`` or ``1080P_global_0040``.
    """
    scope_id = _scope_from_dir_name(scope)
    root = workspace / "global_clips"
    if not root.is_dir():
        return {"ok": False, "error": "global_clips directory not found"}
    # accept either bare name or 1080P_<scope>
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_file() or child.suffix.lower() != ".mp4":
            continue
        if child.stem == scope or child.stem.endswith(f"_{scope_id}"):
            candidates.append(child)
    if not candidates:
        # try a literal match of the supplied scope string
        literal = root / f"{scope}.mp4"
        if literal.is_file():
            candidates.append(literal)
    if not candidates:
        return {"ok": False, "error": f"global clip not found: {scope}"}
    clip = candidates[0]
    try:
        size = clip.stat().st_size
    except OSError:
        size = 0
    # Look for sibling metadata
    meta_path = clip.with_suffix(".json")
    meta = P._safe_read_json(meta_path) if meta_path.is_file() else None
    return {
        "ok": True,
        "scope_id": scope_id,
        "clip_id": clip.stem,
        "filename": clip.name,
        "path": _rel(clip, workspace),
        "size": size,
        "exists": True,
        "metadata": meta or {},
        "metadata_path": _rel(meta_path, workspace) if meta_path.is_file() else "",
    }


def get_scene_payload(workspace: Path, scope: str,
                      scene_opt: bool = False) -> dict[str, Any]:
    """Return the ``scene_index.json`` + ``candidate_scores.json`` payload
    for one global scope, plus a lightweight candidate list.
    """
    scope_id = _scope_from_dir_name(scope)
    root = _scene_root(workspace, scene_opt=scene_opt)
    if root is None:
        return {"ok": False, "error": "scene directory not found"}
    # Match by scope_id or full dir name
    scene_dir: Path | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name == scope or child.name.endswith(f"_{scope_id}") \
                or _scope_from_dir_name(child.name) == scope_id:
            scene_dir = child
            break
    if scene_dir is None:
        return {"ok": False, "error": f"scene not found for scope: {scope}"}

    idx_path = scene_dir / "scene_index.json"
    cand_path = scene_dir / "candidate_scores.json"
    scene_index = P._safe_read_json(idx_path) or {}
    candidate_scores = P._safe_read_json(cand_path) or {}

    # candidate list — prefer candidate_scores (newer) then scene_index
    candidates: list[dict[str, Any]] = []
    raw_candidates = candidate_scores.get("candidates") or scene_index.get("candidates") or []
    if isinstance(raw_candidates, list):
        for c in raw_candidates:
            if not isinstance(c, dict):
                continue
            candidates.append({
                "frame_index": c.get("frame_index"),
                "timestamp_sec": c.get("timestamp_sec"),
                "scene_cluster_id": c.get("scene_cluster_id"),
                "total_score": c.get("total_score"),
                "selection_score": c.get("selection_score"),
                "selected_rank": c.get("selected_rank"),
                "quality_gate": c.get("quality_gate"),
                "person_gate": c.get("person_gate"),
                "dino_similarity_to_main": c.get("dino_similarity_to_main"),
                "same_scene_as_main": c.get("same_scene_as_main"),
            })

    # scene list (segmented entries from scene_index)
    scenes: list[dict[str, Any]] = []
    for s in scene_index.get("scenes", []) or []:
        if not isinstance(s, dict):
            continue
        scenes.append({
            "scene_id": s.get("scene_id") or s.get("id"),
            "start_sec": s.get("start_sec") or s.get("start_time"),
            "end_sec": s.get("end_sec") or s.get("end_time"),
            "frame_count": s.get("frame_count"),
            "preview_path": s.get("preview_path") or s.get("preview"),
        })

    return {
        "ok": True,
        "scope_id": scope_id,
        "scene_dir": _rel(scene_dir, workspace),
        "scene_opt": scene_opt,
        "scene_index": scene_index,
        "scene_index_path": _rel(idx_path, workspace),
        "candidate_scores": candidate_scores,
        "candidate_scores_path": _rel(cand_path, workspace),
        "candidates": candidates,
        "scenes": scenes,
        "candidate_count": len(candidates),
    }


def get_key_object_payload(workspace: Path, scope: str) -> dict[str, Any]:
    """Return object observations + crop listing for one scope.

    Filters observations by ``semantic_scope_id`` matching *scope* (which may
    be ``global_0040`` or ``1080P_global_0040``).  If the upstream pipeline
    did not write ``semantic_scope_id`` (it is null on this dataset), all
    observations are returned and the crops listing is unfiltered — callers
    can group client-side by ``canonical_name``.
    """
    scope_id = _scope_from_dir_name(scope) if scope else ""
    obs_path = workspace / "key_object" / "grounding" / "object_observations.jsonl"
    observations: list[dict[str, Any]] = []
    scope_unset_in_data = False
    if obs_path.is_file():
        all_rows = P._load_jsonl(obs_path)
        # Detect whether the dataset carries scope info at all — if every
        # row's scope_id is empty, ``scope_id`` filtering would drop
        # everything, so we surface them all.
        if all_rows and not any(
            (r.get("semantic_scope_id") or r.get("scope_id")) for r in all_rows
        ):
            scope_unset_in_data = True
        for row in all_rows:
            row_scope = str(row.get("semantic_scope_id") or row.get("scope_id") or "")
            row_scope_id = _scope_from_dir_name(row_scope) if row_scope else ""
            if scope_id and not scope_unset_in_data and row_scope_id \
                    and row_scope_id != scope_id:
                continue
            observations.append({
                "candidate_id": row.get("candidate_id"),
                "scope_id": row.get("semantic_scope_id") or row.get("scope_id"),
                "canonical_name": row.get("canonical_name"),
                "shot_id": row.get("shot_id"),
                "frame_index": row.get("frame_index"),
                "timestamp_sec": row.get("timestamp_sec"),
                "bbox": row.get("selected_bbox_xyxy") or row.get("mask_bbox_xyxy"),
                "mask_area_pixels": row.get("mask_area_pixels"),
                "mask_area_ratio": row.get("mask_area_ratio"),
                "status": row.get("status"),
                "grounding_phrases": row.get("grounding_phrases"),
                "assets": row.get("assets"),
            })

    # crops listing: enumerate key_object/grounding/crops/* (flat layout).
    # Crop filenames start with ``<candidate_id>`` (e.g.
    # ``SHOT_772_OC1_SHOT_772_F000000.jpg``), so we filter by candidate_id
    # belonging to the requested scope when possible.
    crops_root = workspace / "key_object" / "grounding" / "crops"
    crops: list[dict[str, Any]] = []
    if crops_root.is_dir():
        # Build a set of candidate_ids belonging to the current scope, so
        # we can prune crops from other scopes when the caller asked for
        # one specific scope.
        scope_candidate_ids: set[str] | None = None
        if scope_id:
            scope_candidate_ids = {
                str(o.get("candidate_id") or "")
                for o in observations
                if o.get("candidate_id")
            }
        for entry in sorted(crops_root.iterdir()):
            if not entry.is_file():
                continue
            # Filename pattern: <candidate_id>_<...>.jpg where candidate_id
            # itself contains underscores (e.g. ``SHOT_772_OC1``).  Recover it
            # by stripping the trailing ``_SHOT_<id>_F<frame>.<ext>`` suffix.
            stem = entry.stem
            cid = ""
            parts = stem.split("_SHOT_")
            if len(parts) >= 2 and "_OC" in parts[0]:
                cid = parts[0]
            else:
                cid = stem.split("_", 1)[0]
            if scope_candidate_ids is not None and scope_candidate_ids \
                    and not scope_unset_in_data \
                    and cid not in scope_candidate_ids:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            crops.append({
                "filename": entry.name,
                "path": _rel(entry, workspace),
                "size": size,
                "candidate_id": cid,
                "scope_unset": scope_unset_in_data,
            })

    return {
        "ok": True,
        "scope_id": scope_id,
        "observations": observations,
        "observation_count": len(observations),
        "crops": crops,
        "crop_count": len(crops),
    }


def get_i2i_bundle_payload(workspace: Path, bundle_id: str) -> dict[str, Any]:
    """Return the full JSON content of an i2i bundle.

    *bundle_id* is the slash-joined relative path without ``.json`` (as
    returned by :func:`_list_i2i_bundle_entities`), e.g.
    ``scene/scene_global_0040_segment_001`` or ``object/object_SHOT_772_OC1``.
    """
    root = workspace / "i2i_bundles"
    if not root.is_dir():
        return {"ok": False, "error": "i2i_bundles directory not found"}
    rel = Path(bundle_id + ".json")
    target = root / rel
    if not target.is_file():
        return {"ok": False, "error": f"i2i bundle not found: {bundle_id}"}
    payload = P._safe_read_json(target)
    if payload is None:
        return {"ok": False, "error": f"failed to parse i2i bundle: {bundle_id}"}
    return {
        "ok": True,
        "bundle_id": bundle_id,
        "path": _rel(target, workspace),
        "filename": target.name,
        "bundle": payload,
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _extract_shot_id(tracklet_id: str) -> str | None:
    """``SHOT_1080P_SHOT_00790_T000`` → ``SHOT_1080P_SHOT_00790``."""
    if "_T" not in tracklet_id:
        return None
    return tracklet_id.rsplit("_T", 1)[0]


def _list_media(directory: Path, suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in suffixes:
            continue
        out.append({
            "filename": entry.name,
            "path": str(entry.resolve()),
            "size": entry.stat().st_size,
        })
    return out


def _count_media_recursive(directory: Path, suffixes: tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    if any(directory.glob("*/*")) and not any(directory.glob("*.*")):
        return sum(1 for sub in directory.iterdir() if sub.is_dir()
                   for f in sub.iterdir() if f.is_file() and f.suffix.lower() in suffixes)
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in suffixes)


def _list_media_recursive(directory: Path, suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Walk one level of subdirectories so we surface legacy layout
    ``face/other/*.jpg`` and ``body/unknown/partial/<app>/*.jpg``.
    """
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    if any(directory.glob("*/*")) and not any(directory.glob("*.*")):
        # has only subdirs → walk one level
        for sub in sorted(directory.iterdir()):
            if not sub.is_dir():
                continue
            for entry in sorted(sub.iterdir()):
                if entry.is_file() and entry.suffix.lower() in suffixes:
                    out.append({
                        "filename": f"{sub.name}/{entry.name}",
                        "path": str(entry.resolve()),
                        "size": entry.stat().st_size,
                        "view": sub.name,
                    })
        if out:
            return out
    return _list_media(directory, suffixes)


def _rel(path: Path | None, workspace: Path) -> str:
    """Return a workspace-relative path string or empty if path is None."""
    if path is None:
        return ""
    try:
        # ``absolute`` normalises ``..`` without dereferencing pipeline-created
        # symlinks (0916 intentionally links its shot clips to the source run).
        return str(path.absolute().relative_to(workspace.absolute()))
    except (ValueError, OSError):
        return str(path.resolve())


def _shots_root(workspace: Path) -> Path | None:
    for candidate in (
        workspace / "assets" / "visual_person_v2" / "tracklets" / "shots",
        workspace / "assets" / "visual_person" / "tracklets" / "shots",
        workspace / "tracklets" / "shots",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _character_dir_for(workspace: Path, scope_id: str,
                       subject_id: str) -> Path | None:
    for base in (workspace / "characters", workspace / "characters" / "characters"):
        candidate = base / scope_id / subject_id
        if candidate.is_dir():
            return candidate
    return None


def _has_characters_dir(workspace: Path) -> bool:
    return any(
        (workspace / base / scope).is_dir()
        for base in ("characters", "characters/characters")
        for scope in ("global_0040", "global_0041", "global_0042")
    ) or any(
        child.is_dir() and child.name.startswith("global_")
        for base in (workspace / "characters", workspace / "characters" / "characters")
        if base.is_dir()
        for child in base.iterdir()
    )


def _has_voice_dir(workspace: Path) -> bool:
    for base in (workspace / "characters", workspace / "characters" / "characters",
                 workspace / "voice_by_role_src"):
        if not base.is_dir():
            continue
        for scope in base.iterdir():
            if not scope.is_dir() or not scope.name.startswith("global_"):
                continue
            for subj in scope.iterdir():
                if subj.is_dir() and subj.name.startswith("S"):
                    return True
    return False


def _has_global_clips_dir(workspace: Path) -> bool:
    root = workspace / "global_clips"
    if not root.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() == ".mp4"
               for child in root.iterdir())


def _scene_root(workspace: Path, scene_opt: bool) -> Path | None:
    """Return the per-global scene root for *scene_opt* variants.

    The default scene stage reads from ``scene/``; the optimised variant
    reads from ``scene_opt_marginal/`` if present, otherwise falls back to
    ``scene/`` so callers always see something.
    """
    if scene_opt:
        for candidate in (workspace / "scene_opt_marginal", workspace / "scene_opt",
                          workspace / "scene"):
            if candidate.is_dir():
                return candidate
        return None
    for candidate in (workspace / "scene",):
        if candidate.is_dir():
            return candidate
    return None


def _has_scene_dir(workspace: Path, prefer_scene_opt: bool) -> bool:
    """Return True if a usable per-global scene directory exists.

    *prefer_scene_opt* selects between the standard ``scene/`` layout (False)
    and the optimised ``scene_opt_marginal/`` variant (True).
    """
    root = _scene_root(workspace, scene_opt=prefer_scene_opt)
    if root is None:
        return False
    # require at least one per-global subdir OR a scene_index manifest
    if (root / "scene_segment_index.json").is_file():
        return True
    for child in root.iterdir():
        if child.is_dir():
            return True
    return False


def _has_key_object_dir(workspace: Path) -> bool:
    root = workspace / "key_object"
    if not root.is_dir():
        return False
    if (root / "grounding" / "object_observations.jsonl").is_file():
        return True
    if (root / "grounding_queue.jsonl").is_file():
        return True
    if any(root.glob("**/*.jsonl")):
        return True
    return False


def _has_i2i_bundles_dir(workspace: Path) -> bool:
    root = workspace / "i2i_bundles"
    if not root.is_dir():
        return False
    return any(root.rglob("*.json"))
