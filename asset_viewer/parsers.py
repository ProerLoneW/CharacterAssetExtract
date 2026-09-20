"""Schema parsers for the asset viewer pipeline router.

The viewer reads JSON/JSONL manifests produced by both pipeline chains and
exposes a *uniform* dataclass layer to the HTTP handlers in
``asset_viewer/server.py``.  Each parser returns paths resolved relative
to the workspace root so the media proxy can validate them safely.

The functions deliberately mirror the loader helpers in
``extract_full_pipeline/global_align_pipeline/global_tracklet_aligner.py``
(``_load_jsonl``, ``_shot_number``, ``_scope_id``) so we stay in sync with
the schema string constants declared there (``SCHEMA_VERSION``).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("asset_viewer.parsers")

# Schema strings — kept in sync with the producers
SCHEMA_GLOBAL_ALIGNMENT = "global-tracklet-character-alignment-v1"
SCHEMA_GLOBAL_MATERIALIZATION = "global-character-materialization-v1"
SCHEMA_TRACKLET_V2 = "shot-tracklets-v2"
SCHEMA_ASSET_BANK_V2 = "character-asset-bank-v2"
SCHEMA_ROLE_ALIGNMENT_V2 = "character-role-alignment-v2"

SHOT_NUMBER_PATTERN = re.compile(r"(\d+)$")
SUBJECT_PATTERN = re.compile(r"S[1-9][0-9]*$")


# --------------------------------------------------------------------------- #
# generic IO helpers
# --------------------------------------------------------------------------- #
def _safe_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning ``None`` on any parse/IO error.

    The viewer must not crash on a single corrupted manifest; bad files are
    logged at WARNING level and the caller is expected to skip them.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("failed to read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("expected JSON object at %s, got %s", path, type(payload).__name__)
        return None
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blanks and non-object rows.  Mirrors
    ``global_tracklet_aligner._load_jsonl`` so we keep the same validation
    semantics as the upstream pipeline.
    """
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            LOGGER.warning("%s:%d JSONL parse error: %s", path, line_number, exc)
            continue
        if not isinstance(row, dict):
            LOGGER.warning("%s:%d expected JSON object, got %s",
                           path, line_number, type(row).__name__)
            continue
        rows.append(row)
    return rows


def _shot_number(value: Any) -> int | None:
    """Extract trailing digits from a shot id ("SHOT_00790" → 790)."""
    match = SHOT_NUMBER_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def _scope_id(global_id: Any) -> str:
    """Return the canonical ``global_NNNN`` form for a numeric id."""
    raw = str(global_id)
    return f"global_{int(raw):04d}" if raw.isdigit() else raw


def _to_relative(path: str | Path, workspace: Path) -> str:
    """Best-effort path relativization against the workspace root.

    Paths recorded in the upstream JSONs come from a GPU server and may
    point outside the workspace, but for the test runs they live directly
    under the workspace.  When the path cannot be relativized (different
    mount) we leave it absolute so the media handler can decide.
    """
    if path is None:
        return ""
    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(workspace.resolve()))
    except (ValueError, OSError):
        return str(p)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class MaterializationRow:
    scope_id: str
    subject_id: str
    tracklet_count: int
    face_copied: int
    body_copied: int
    rgba_copied: int
    voice_copied: int
    contributing_tracklets: int
    missing_files: int

    @property
    def scope_subject(self) -> str:
        return f"{self.scope_id}:{self.subject_id}"


@dataclass
class RoleCharacter:
    scope_id: str
    subject_id: str
    tracklet_count: int
    tracklet_ids: list[str]
    mean_alignment_confidence: float
    visual_dir: str
    voice_dir: str
    source_alignment: str


@dataclass
class TrackletRow:
    tracklet_id: str
    shot_id: str
    start_time: float
    end_time: float
    observation_count: int
    face_count: int
    body_count: int
    best_face_quality: float | None
    best_body_quality: float | None
    cannot_links_count: int
    face_dir: str   # relative path containing face crops (may be empty)
    body_dir: str   # relative path containing body crops (may be empty)


@dataclass
class GlobalAlignmentRow:
    scope_id: str
    global_id: Any
    status: str
    tracklet_count: int
    assigned_tracklet_count: int
    stitched_video: str   # workspace-relative
    alignments: list[dict[str, Any]]
    unmatched_subjects: list[str]
    quality_issues: list[str]


@dataclass
class AssetBankCharacter:
    character_id: str
    source_candidate_id: str
    tracklet_ids: list[str]
    shot_ids: list[str]
    identity_confidence: float
    face_assets: list[dict[str, Any]]
    body_assets: list[dict[str, Any]]
    representative_set: dict[str, str]
    coverage: dict[str, Any]


# --------------------------------------------------------------------------- #
# specialised parsers
# --------------------------------------------------------------------------- #
def parse_materialization(workspace: Path,
                           path: Path | None = None) -> list[MaterializationRow]:
    """Parse ``materialization_summary.json`` (new chain)."""
    target = path or workspace / "materialization_summary.json"
    payload = _safe_read_json(target)
    if not payload:
        return []
    rows: list[MaterializationRow] = []
    for entry in payload.get("characters", []):
        scope_subject = str(entry.get("scope_subject", ""))
        if ":" not in scope_subject:
            continue
        scope_id, subject_id = scope_subject.split(":", 1)
        try:
            rows.append(MaterializationRow(
                scope_id=scope_id,
                subject_id=subject_id,
                tracklet_count=int(entry.get("tracklet_count", 0)),
                face_copied=int(entry.get("face_copied", 0)),
                body_copied=int(entry.get("body_copied", 0)),
                rgba_copied=int(entry.get("rgba_copied", 0)),
                voice_copied=int(entry.get("voice_copied", 0)),
                contributing_tracklets=int(entry.get("contributing_tracklets", 0)),
                missing_files=int(entry.get("missing_files", 0)),
            ))
        except (TypeError, ValueError) as exc:
            LOGGER.warning("materialization row %r skipped: %s", scope_subject, exc)
    return rows


def parse_role_character(workspace: Path, scope_id: str,
                         subject_id: str) -> RoleCharacter | None:
    """Parse ``characters/<scope>/<S#>/role_character.json`` (new chain).

    The legacy chain stores a different ``role_character.json`` under
    ``characters/characters/<scope>/<S#>/``.  Callers should try the new
    path first, then fall back to the legacy location; this helper handles
    both transparently via ``character_dir_for`` in pipeline_router.
    """
    raise NotImplementedError  # handled inline by router to keep both layouts


def parse_tracklet_signatures(workspace: Path,
                              shot_id: str | int) -> list[TrackletRow]:
    """Parse one shot's ``tracklet_signatures.json``."""
    number = _shot_number(shot_id) if not isinstance(shot_id, int) else shot_id
    if number is None:
        return []
    candidates: Iterable[Path] = (
        workspace / "assets" / "visual_person_v2" / "tracklets" / "shots" / f"SHOT_1080P_SHOT_{number:05d}" / "tracklet_signatures.json",
        workspace / "assets" / "visual_person" / "tracklets" / "shots" / f"SHOT_1080P_SHOT_{number:05d}" / "tracklet_signatures.json",
        workspace / "tracklets" / "shots" / f"SHOT_1080P_SHOT_{number:05d}" / "tracklet_signatures.json",
    )
    for path in candidates:
        payload = _safe_read_json(path)
        if not payload:
            continue
        rows: list[TrackletRow] = []
        for entry in payload.get("tracklet_signatures", []):
            try:
                rows.append(TrackletRow(
                    tracklet_id=str(entry.get("tracklet_id")),
                    shot_id=str(entry.get("shot_id", shot_id)),
                    start_time=float(entry.get("start_time", 0.0)),
                    end_time=float(entry.get("end_time", 0.0)),
                    observation_count=len(entry.get("observation_ids", []) or []),
                    face_count=len(entry.get("face_gallery", []) or []),
                    body_count=len(entry.get("body_gallery", []) or []),
                    best_face_quality=_safe_float(entry.get("best_face_quality")),
                    best_body_quality=_safe_float(entry.get("best_body_quality")),
                    cannot_links_count=int(entry.get("cannot_links_count", 0)),
                    face_dir="",
                    body_dir="",
                ))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("tracklet row skipped in %s: %s", path, exc)
        return rows
    return []


def parse_global_alignment(workspace: Path) -> list[GlobalAlignmentRow]:
    """Parse ``global_alignment/global_tracklet_alignment.jsonl`` (new chain)."""
    jsonl = workspace / "global_alignment" / "global_tracklet_alignment.jsonl"
    out: list[GlobalAlignmentRow] = []
    for row in _load_jsonl(jsonl):
        try:
            out.append(GlobalAlignmentRow(
                scope_id=str(row.get("global_scope_id") or _scope_id(row.get("global_id"))),
                global_id=row.get("global_id"),
                status=str(row.get("status", "unknown")),
                tracklet_count=int(row.get("tracklet_count", 0)),
                assigned_tracklet_count=int(row.get("assigned_tracklet_count") or 0),
                stitched_video=_to_relative(row.get("stitched_video", ""), workspace),
                alignments=list(row.get("tracklet_alignments") or []),
                unmatched_subjects=[str(s) for s in row.get("unmatched_semantic_subject_ids", []) or []],
                quality_issues=[str(s) for s in row.get("quality_issues", []) or []],
            ))
        except (TypeError, ValueError) as exc:
            LOGGER.warning("global_alignment row skipped: %s", exc)
    return out


def parse_global_alignment_summary(workspace: Path) -> dict[str, Any]:
    payload = _safe_read_json(workspace / "global_alignment" / "summary.json") or {}
    return payload


def parse_asset_bank_registry(workspace: Path) -> tuple[list[AssetBankCharacter], dict[str, Any]]:
    """Parse ``assets/visual_person_v2/asset_bank/registry.json`` (legacy chain)."""
    path = workspace / "assets" / "visual_person_v2" / "asset_bank" / "registry.json"
    payload = _safe_read_json(path)
    if not payload:
        return [], {}
    out: list[AssetBankCharacter] = []
    for entry in payload.get("characters", []):
        try:
            out.append(AssetBankCharacter(
                character_id=str(entry.get("character_id")),
                source_candidate_id=str(entry.get("source_candidate_id", "")),
                tracklet_ids=[str(t) for t in entry.get("tracklet_ids", []) or []],
                shot_ids=[str(s) for s in entry.get("shot_ids", []) or []],
                identity_confidence=float(entry.get("identity_confidence", 0.0) or 0.0),
                face_assets=_collect_assets(entry.get("face_gallery")),
                body_assets=_collect_assets(entry.get("body_gallery")),
                representative_set=dict(entry.get("representative_set") or {}),
                coverage=dict(entry.get("coverage") or {}),
            ))
        except (TypeError, ValueError) as exc:
            LOGGER.warning("asset_bank character row skipped: %s", exc)
    return out, payload


def parse_role_alignment(workspace: Path) -> dict[str, Any]:
    """Parse ``characters/role_alignment.json`` (legacy chain top-level)."""
    return _safe_read_json(workspace / "characters" / "role_alignment.json") or {}


# --------------------------------------------------------------------------- #
# private helpers
# --------------------------------------------------------------------------- #
def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def _collect_assets(gallery: Any) -> list[dict[str, Any]]:
    if not isinstance(gallery, dict):
        return []
    out: list[dict[str, Any]] = []
    for view, items in gallery.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                copied = dict(item)
                copied.setdefault("view", view)
                out.append(copied)
    return out