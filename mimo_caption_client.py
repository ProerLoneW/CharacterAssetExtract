#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared atomic helpers for the MiMo three-level caption pipeline.

This module groups every reusable primitive that the global / shot / segment
captioners rely on.  Originally lived in
``/gemini/platform/aigc/lss/Interns/lsj/video_process/scripts/annotate_shots_mimo.py``
and ``annotate_global_mimo.py``; it is extracted here so each stage stays a
single-purpose module.

Responsibilities:

        * load ``.env`` files dropped next to the original repo
        * build MiMo chat-completion ``content`` payloads (video_url / input_video)
        * unwrap text from OpenAI/MiMo response shapes (content / reasoning)
        * extract a JSON object from a possibly noisy free-form response
        * resolve ``shots.jsonl`` / ``global_clips.jsonl`` / ``global_captions.jsonl``
          manifests and load their records (with clip-path lookups)
        * persist per-item JSON / raw-response files and a consolidated JSONL

The module does **not** perform network IO, validate schemas, or schedule
retries. Those are owned by the per-stage modules
:mod:`mimo_global_captioner`, :mod:`mimo_shot_captioner` and
:mod:`mimo_segment_captioner`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Path constants pointing at the original MiMo three-level caption repo.
# Update SOURCE_VIDEO_PROCESS_ROOT when the upstream repo moves.
# ---------------------------------------------------------------------------
SOURCE_VIDEO_PROCESS_ROOT = Path(
    "/gemini/platform/public/aigc/lss/Interns/lsj/video_process"
)
SOURCE_PROMPTS_DIR = SOURCE_VIDEO_PROCESS_ROOT / "prompts"
SOURCE_GLOBAL_PROMPT = SOURCE_PROMPTS_DIR / "global_caption_prompt_en.md"
SOURCE_SHOT_PROMPT = SOURCE_PROMPTS_DIR / "shot_caption_prompt_en.md"
SOURCE_SEGMENT_PROMPT = SOURCE_PROMPTS_DIR / "segment_caption_prompt_en.md"
SOURCE_ENV_FILE = SOURCE_VIDEO_PROCESS_ROOT / "env" / ".env"

# --- preserved v2 prompt overlay (role-alignment patches; see docs/) ------
# These point at the *_with_role_alignment.md copies in the v2 repo-root
# ``prompts/`` directory (one level up from this file) and remain selectable
# through ``v2=True`` or ``version="v2"`` for rollback.
SOURCE_PROMPTS_V2_DIR = Path(__file__).resolve().parent.parent / "prompts"
SOURCE_GLOBAL_PROMPT_V2 = SOURCE_PROMPTS_V2_DIR / "global_caption_prompt_en_with_role_alignment.md"
SOURCE_SHOT_PROMPT_V2 = SOURCE_PROMPTS_V2_DIR / "shot_caption_prompt_en_with_role_alignment.md"
SOURCE_SEGMENT_PROMPT_V2 = SOURCE_PROMPTS_V2_DIR / "segment_caption_prompt_en_with_role_alignment.md"

# --- v3 semantic-character prompts -------------------------------------
# V3 keeps the V2 stable-ID contract and adds controlled narrative-role
# evidence consumed by semantic_role_tracklet_refiner.py.
SOURCE_PROMPTS_V3_DIR = Path(__file__).resolve().parent.parent / "prompts"
SOURCE_GLOBAL_PROMPT_V3 = SOURCE_PROMPTS_V3_DIR / "global_caption_prompt_en_semantic_roles_v3.md"
SOURCE_SHOT_PROMPT_V3 = SOURCE_PROMPTS_V3_DIR / "shot_caption_prompt_en_semantic_roles_v3.md"
SOURCE_SEGMENT_PROMPT_V3 = SOURCE_PROMPTS_V3_DIR / "segment_caption_prompt_en_semantic_roles_v3.md"

# The service DNS is the current in-cluster endpoint documented by this repo.
# Do not use the older ClusterIP here: it can change when the Service is
# recreated, while the DNS name remains stable.
DEFAULT_BASE_URL = "http://tcp30000-1058.default.svc.cluster.local:30000/v1"
DEFAULT_MODEL = "MiMo-V2.5"
DEFAULT_API_KEY = "EMPTY"

LOGGER = logging.getLogger("mimo_caption_client")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def load_dotenv(path: Path | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file without overwriting.

    Defaults to the upstream repo's ``env/.env`` so the user only configures
    the model endpoint once. Missing files are silently ignored.
    """
    target = path or SOURCE_ENV_FILE
    if not target or not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\"" ))


def configure_no_proxy_for_url(url: str) -> str:
    """Keep an in-cluster MiMo endpoint away from workstation HTTP proxies.

    The shared execution environment may export ``http_proxy`` for internet
    access.  urllib/httpx otherwise route Kubernetes service traffic through
    that proxy and surface misleading HTTP 503 or disconnected responses.
    """
    hostname = urlsplit(url).hostname
    if not hostname:
        return url
    for key in ("NO_PROXY", "no_proxy"):
        values = [value.strip() for value in os.getenv(key, "").split(",")
                  if value.strip()]
        if hostname not in values:
            values.append(hostname)
        os.environ[key] = ",".join(values)
    return url


def resolve_base_url(override: str | None = None) -> str:
    selected = override or os.getenv("OPENAI_BASE_URL") \
        or os.getenv("OPENAI_API_BASE") or os.getenv("API_BASE") \
        or DEFAULT_BASE_URL
    return configure_no_proxy_for_url(selected)


def resolve_model(override: str | None = None) -> str:
    return override or os.getenv("MODEL_NAME") or DEFAULT_MODEL


def resolve_api_key(override: str | None = None) -> str:
    return override or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------
def default_prompt_path(stage: str, *, v2: bool = False,
                        version: str | None = None) -> Path:
    """Return the absolute path to the upstream prompt for ``stage``.

    ``stage`` is one of ``"global"``, ``"shot"`` or ``"segment"``.
    New callers default to V3 semantic-role prompts. ``v2=True`` is retained
    for backward compatibility, and ``version="v1"|"v2"|"v3"`` provides an
    explicit rollback switch. Missing local overlays fall back V3 -> V2 -> V1.
    """
    mapping_v1 = {
        "global": SOURCE_GLOBAL_PROMPT,
        "shot": SOURCE_SHOT_PROMPT,
        "segment": SOURCE_SEGMENT_PROMPT,
    }
    mapping_v2 = {
        "global": SOURCE_GLOBAL_PROMPT_V2,
        "shot": SOURCE_SHOT_PROMPT_V2,
        "segment": SOURCE_SEGMENT_PROMPT_V2,
    }
    mapping_v3 = {
        "global": SOURCE_GLOBAL_PROMPT_V3,
        "shot": SOURCE_SHOT_PROMPT_V3,
        "segment": SOURCE_SEGMENT_PROMPT_V3,
    }
    try:
        v1 = mapping_v1[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {stage!r}; expected global/shot/segment") from exc
    selected = version or ("v2" if v2 else "v3")
    if selected not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unknown prompt version: {selected!r}; expected v1/v2/v3")
    if selected == "v1":
        return v1
    v2_path, v3_path = mapping_v2[stage], mapping_v3[stage]
    if selected == "v2":
        return v2_path if v2_path.is_file() else v1
    if v3_path.is_file():
        return v3_path
    return v2_path if v2_path.is_file() else v1


# ---------------------------------------------------------------------------
# JSONL manifest helpers
# ---------------------------------------------------------------------------
def find_manifest(input_path: Path, expected_name: str) -> Path:
    """Locate a manifest given either a path to the file or a directory."""
    if input_path.is_file():
        if input_path.name != expected_name:
            raise ValueError(f"Input file must be {expected_name}: {input_path}")
        return input_path
    candidates = sorted(input_path.rglob(expected_name))
    if not candidates:
        raise FileNotFoundError(f"No {expected_name} found under {input_path}")
    if len(candidates) > 1:
        LOGGER.warning("Multiple %s files found; using %s", expected_name, candidates[0])
    return candidates[0]


def find_shot_manifest(input_path: Path) -> Path:
    return find_manifest(input_path, "shots.jsonl")


def find_global_clip_manifest(input_path: Path) -> Path:
    return find_manifest(input_path, "global_clips.jsonl")


def find_global_caption_manifest(input_path: Path) -> Path:
    return find_manifest(input_path, "global_captions.jsonl")


def find_shot_caption_manifest(input_path: Path) -> Path:
    return find_manifest(input_path, "shot_captions.jsonl")


def find_segment_caption_manifest(input_path: Path) -> Path:
    return find_manifest(input_path, "segment_captions.jsonl")


def resolve_clip_path(raw_path: str | Path | None, manifest_path: Path) -> Path:
    """Resolve ``clip_path`` relative to its manifest's ancestors."""
    if not raw_path:
        raise ValueError("clip_path is required")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for base in manifest_path.parents:
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return path.resolve()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts; raise helpful errors on bad rows."""
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Atomically-ish write ``records`` to ``path`` line-buffered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def persist_item(
    individual_dir: Path,
    raw_dir: Path,
    prefix: str,
    index: int,
    payload: dict,
    response_text: str | None,
) -> None:
    """Persist one item as ``<prefix>_NNNNN.json`` plus an optional raw text."""
    individual_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = individual_dir / f"{prefix}_{index:05d}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if response_text:
        (raw_dir / f"{prefix}_{index:05d}.txt").write_text(response_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# MiMo content payload construction
# ---------------------------------------------------------------------------
def build_video_content(prompt: str, encoded: str, request_format: str) -> list[dict]:
    """Build the OpenAI ``messages.content`` array for one video request."""
    if request_format == "video_url":
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
        ]
    if request_format == "input_video":
        return [
            {"type": "text", "text": prompt},
            {"type": "input_video", "input_video": {"data": encoded, "format": "mp4"}},
        ]
    raise ValueError(f"Unsupported video request format: {request_format}")


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------
def extract_text(response: Any) -> str:
    """Extract plain text from an OpenAI chat completion response."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            if value:
                parts.append(str(value))
        return "\n".join(parts).strip()
    return str(content)


def extract_response_text(response: Any) -> str:
    """Like :func:`extract_text` but also falls back to ``reasoning_content``.

    Mirrors the upstream ``annotate_segments_mimo.py`` behaviour that is needed
    when the MiMo endpoint exposes the answer under ``reasoning_content`` instead
    of ``content``.
    """
    message = response.choices[0].message
    for value in (
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
    ):
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            if parts:
                return "\n".join(parts).strip()
    return ""


def response_diagnostic(response: Any) -> str:
    """Serialize the returned message when no textual content is available."""
    try:
        message = response.choices[0].message
        if hasattr(message, "model_dump"):
            return json.dumps(message.model_dump(), ensure_ascii=False, default=str)
        if hasattr(message, "dict"):
            return json.dumps(message.dict(), ensure_ascii=False, default=str)
        return repr(message)
    except Exception:  # noqa: BLE001
        return repr(response)


# ---------------------------------------------------------------------------
# JSON extraction from noisy responses
# ---------------------------------------------------------------------------
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str) -> tuple[dict | None, str | None]:
    """Pull a single JSON object out of a possibly noisy free-form response."""
    cleaned = (text or "").strip()
    cleaned = _THINK_PATTERN.sub("", cleaned).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "response is not a JSON object"
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            return None, "response does not contain a JSON object"
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(cleaned[start:], start=start):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start : index + 1])
                        return (parsed, None) if isinstance(parsed, dict) else (None, "response is not a JSON object")
                    except json.JSONDecodeError as exc:
                        return None, f"invalid JSON: {exc}"
        return None, "response does not contain a JSON object"


def encode_video_b64(video_path: Path) -> str:
    """Synchronously read a file and return its base64 encoding."""
    return base64.b64encode(video_path.read_bytes()).decode("ascii")


def unwrap_nested_caption(record: dict, key: str) -> dict:
    """Merge nested ``record[key][key]`` with the outer wrapper.

    Upstream responses wrap the schema under the same key while keeping other
    fields at the outer level. Segment stage merges both so identity and
    timeline evidence are not lost.
    """
    import copy as _copy

    value = record.get(key)
    if not isinstance(value, dict):
        return {}
    nested = value.get(key)
    if not isinstance(nested, dict):
        return value
    merged = {name: _copy.deepcopy(item) for name, item in value.items() if name != key}
    merged.update(_copy.deepcopy(nested))
    return merged


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "SOURCE_ENV_FILE",
    "SOURCE_GLOBAL_PROMPT",
    "SOURCE_GLOBAL_PROMPT_V2",
    "SOURCE_GLOBAL_PROMPT_V3",
    "SOURCE_PROMPTS_DIR",
    "SOURCE_PROMPTS_V2_DIR",
    "SOURCE_PROMPTS_V3_DIR",
    "SOURCE_SEGMENT_PROMPT",
    "SOURCE_SEGMENT_PROMPT_V2",
    "SOURCE_SEGMENT_PROMPT_V3",
    "SOURCE_SHOT_PROMPT",
    "SOURCE_SHOT_PROMPT_V2",
    "SOURCE_SHOT_PROMPT_V3",
    "SOURCE_VIDEO_PROCESS_ROOT",
    "build_video_content",
    "configure_no_proxy_for_url",
    "default_prompt_path",
    "encode_video_b64",
    "extract_response_text",
    "extract_text",
    "find_global_caption_manifest",
    "find_global_clip_manifest",
    "find_manifest",
    "find_segment_caption_manifest",
    "find_shot_caption_manifest",
    "find_shot_manifest",
    "load_dotenv",
    "load_jsonl",
    "parse_json_response",
    "persist_item",
    "resolve_api_key",
    "resolve_base_url",
    "resolve_clip_path",
    "resolve_model",
    "response_diagnostic",
    "unwrap_nested_caption",
    "write_jsonl",
]


if __name__ == "__main__":  # pragma: no cover - smoke test
    print(f"video_process root: {SOURCE_VIDEO_PROCESS_ROOT}")
    print(f"global prompt:     {SOURCE_GLOBAL_PROMPT.exists()} {SOURCE_GLOBAL_PROMPT}")
    print(f"shot prompt:       {SOURCE_SHOT_PROMPT.exists()} {SOURCE_SHOT_PROMPT}")
    print(f"segment prompt:    {SOURCE_SEGMENT_PROMPT.exists()} {SOURCE_SEGMENT_PROMPT}")
