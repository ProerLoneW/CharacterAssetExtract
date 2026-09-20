"""Asset viewer HTTP server.

Run as::

    python -m extract_full_pipeline.asset_viewer.server \
        --host 0.0.0.0 --port 8910 \
        --workspace-root /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2/workspace

The server implements:

* ``GET /`` — single-page HTML dashboard
* ``GET /api/workspaces`` — discovered workspaces + chain types
* ``GET /api/stages?workspace=…`` — stages and per-workspace KPIs
* ``GET /api/entities?workspace=…&stage=…`` — entities to populate pickers
* ``GET /api/character?workspace=…&scope=…&subject=…`` — character payload
* ``GET /api/global?workspace=…&scope=…`` — single global alignment row
* ``GET /api/tracklet?workspace=…&shot_id=…&tracklet_id=…`` — single tracklet
* ``GET /api/global_clip?workspace=…&scope=…`` — single global clip mp4 payload
* ``GET /api/scene?workspace=…&scope=…&optimized=0|1`` — scene index payload
* ``GET /api/key_object?workspace=…&scope=…`` — object observations payload
* ``GET /api/i2i_bundle?workspace=…&id=…`` — full i2i bundle JSON
* ``GET /media/path?workspace=…&abs=…`` — workspace-scoped media bytes

The architecture mirrors ``review_mv_caption.ReviewServer`` so any
operator familiar with that script recognises the layout immediately.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import webbrowser
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Make this file importable both as ``python server.py`` and as
# ``python -m extract_full_pipeline.asset_viewer.server``.
_HERE = Path(__file__).resolve().parent
_EFP_ROOT = _HERE.parent
if str(_EFP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EFP_ROOT))

from asset_viewer import pipeline_router as R
from asset_viewer.html_template import render_index
from asset_viewer.parsers import _load_jsonl, _safe_read_json, parse_role_character  # noqa: F401

LOGGER = logging.getLogger("asset_viewer.server")

DEFAULT_WORKSPACE_ROOT = Path(
    "/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2/workspace"
)
DEFAULT_PORT = 8910
DEFAULT_HOST = "0.0.0.0"
DEFAULT_WORKSPACE = "movie_full_pipeline_20260916"
DEFAULT_BASE_URL = (
    "/gemini_web/asset_viewer/~58.59.115.26~31443~/movie_full_pipeline_20260916/"
)


# --------------------------------------------------------------------------- #
# workspace lookup with cache
# --------------------------------------------------------------------------- #
class WorkspaceRegistry:
    """Thread-safe lookup of workspace roots and their summary metadata."""

    def __init__(self, root: Path, only_workspace: str = "") -> None:
        self.root = root
        self.only_workspace = only_workspace
        self._lock = threading.RLock()
        self._by_name: dict[str, Path] = {}
        self._summaries: dict[str, dict[str, Any]] = {}

    def refresh(self) -> None:
        with self._lock:
            self._by_name.clear()
            self._summaries.clear()
            if not self.root.is_dir():
                return
            for child in sorted(self.root.iterdir()):
                if not child.is_dir() or not child.name.startswith("movie_full_pipeline_"):
                    continue
                if self.only_workspace and child.name != self.only_workspace:
                    continue
                if R.detect_chain(child) == R.CHAIN_NONE:
                    continue
                self._by_name[child.name] = child.resolve()
                self._summaries[child.name] = {
                    "name": child.name,
                    "path": str(child.resolve()),
                    "chain": R.detect_chain(child),
                    "stages": R.list_stages(child),
                    "scopes": R.list_scopes(child),
                }

    def names(self) -> list[str]:
        with self._lock:
            if not self._by_name:
                self.refresh()
            return list(self._by_name.keys())

    def resolve(self, name: str) -> Path | None:
        with self._lock:
            if not self._by_name:
                self.refresh()
            return self._by_name.get(name)

    def summary(self, name: str) -> dict[str, Any]:
        with self._lock:
            if not self._summaries:
                self.refresh()
            return self._summaries.get(name, {})

    def list_payload(self) -> list[dict[str, Any]]:
        return [self.summary(n) for n in self.names()]


# --------------------------------------------------------------------------- #
# request handler
# --------------------------------------------------------------------------- #
class ViewerRequestHandler(BaseHTTPRequestHandler):
    server: "ViewerServer"

    # ----------------------------------------------------------------- logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
        LOGGER.info("%s - %s", self.address_string(), format % args)

    # ----------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._serve_index()
            elif parsed.path == "/api/workspaces":
                self._send_json({"workspaces": self.server.registry.list_payload()})
            elif parsed.path == "/api/stages":
                self._api_stages(query)
            elif parsed.path == "/api/entities":
                self._api_entities(query)
            elif parsed.path == "/api/character":
                self._api_character(query)
            elif parsed.path == "/api/global":
                self._api_global(query)
            elif parsed.path == "/api/tracklet":
                self._api_tracklet(query)
            elif parsed.path == "/api/global_clip":
                self._api_global_clip(query)
            elif parsed.path == "/api/scene":
                self._api_scene(query)
            elif parsed.path == "/api/key_object":
                self._api_key_object(query)
            elif parsed.path == "/api/i2i_bundle":
                self._api_i2i_bundle(query)
            elif parsed.path == "/api/detail":
                self._api_detail(query)
            elif parsed.path == "/api/artifacts":
                self._api_artifacts(query)
            elif parsed.path == "/api/artifact":
                self._api_artifact(query)
            elif parsed.path == "/media/path":
                self._api_media(query)
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.exception("GET %s failed: %s", parsed.path, exc)
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    # ----------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        self._send_json(
            {"ok": False, "error": "POST endpoints are not implemented in this version."},
            status=501,
        )

    # ----------------------------------------------------------------- helpers
    def _serve_index(self) -> None:
        payload = {
            "workspaces": self.server.registry.list_payload(),
            "default_workspace": self.server.default_workspace,
            "base_url": self.server.config.base_url,
        }
        body = render_index(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "frame-ancestors 'self' *")
        self.end_headers()
        self.wfile.write(body)

    def _api_stages(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None:
            return self._send_json({"ok": False, "error": f"unknown workspace: {name}"}, status=404)
        chain = R.detect_chain(workspace)
        stages = R.list_stages(workspace)
        kpis = _compute_kpis(workspace, stages)
        entities = _entities_for_overview(workspace, stages)
        self._send_json({
            "ok": True,
            "workspace": name,
            "chain": chain,
            "stages": stages,
            "kpis": kpis,
            "entities": entities,
        })

    def _api_entities(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        stage = _q(query, "stage")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not stage:
            return self._send_json({"ok": False, "error": "workspace/stage required"}, status=400)
        entities = R.list_entities(workspace, stage)
        self._send_json({"ok": True, "workspace": name, "stage": stage, "entities": entities})

    def _api_character(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        scope = _q(query, "scope")
        subject = _q(query, "subject")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not scope or not subject:
            return self._send_json({"ok": False, "error": "workspace/scope/subject required"}, status=400)
        payload = R.get_character_payload(workspace, scope, subject)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_global(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        scope = _q(query, "scope")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not scope:
            return self._send_json({"ok": False, "error": "workspace/scope required"}, status=400)
        payload = R.get_global_payload(workspace, scope)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_tracklet(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        shot_id = _q(query, "shot_id")
        tid = _q(query, "tracklet_id")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not shot_id or not tid:
            return self._send_json({"ok": False, "error": "workspace/shot_id/tracklet_id required"},
                                   status=400)
        payload = R.get_tracklet_payload(workspace, shot_id, tid)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_global_clip(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        scope = _q(query, "scope")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not scope:
            return self._send_json({"ok": False, "error": "workspace/scope required"}, status=400)
        payload = R.get_global_clip_payload(workspace, scope)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_scene(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        scope = _q(query, "scope")
        optimized = _q(query, "optimized") or "0"
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not scope:
            return self._send_json({"ok": False, "error": "workspace/scope required"}, status=400)
        scene_opt = optimized in ("1", "true", "True", "yes")
        payload = R.get_scene_payload(workspace, scope, scene_opt=scene_opt)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_key_object(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        scope = _q(query, "scope")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not scope:
            return self._send_json({"ok": False, "error": "workspace/scope required"}, status=400)
        payload = R.get_key_object_payload(workspace, scope)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_i2i_bundle(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        bundle_id = _q(query, "id")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not bundle_id:
            return self._send_json({"ok": False, "error": "workspace/id required"}, status=400)
        payload = R.get_i2i_bundle_payload(workspace, bundle_id)
        if not payload.get("ok"):
            return self._send_json(payload, status=404)
        self._send_json(payload)

    def _api_detail(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        stage = _q(query, "stage")
        item_id = _q(query, "id")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not stage or not item_id:
            return self._send_json({"ok": False, "error": "workspace/stage/id required"}, status=400)
        payload = R.get_stage_detail(workspace, stage, item_id)
        self._send_json(payload, status=200 if payload.get("ok") else 404)

    def _api_artifacts(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        prefix = _q(query, "prefix") or ""
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None:
            return self._send_json({"ok": False, "error": "workspace required"}, status=400)
        payload = R.list_artifacts(workspace, prefix)
        self._send_json(payload, status=200 if payload.get("ok") else 404)

    def _api_artifact(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        path = _q(query, "path")
        workspace = self.server.registry.resolve(name) if name else None
        if workspace is None or not path:
            return self._send_json({"ok": False, "error": "workspace/path required"}, status=400)
        payload = R.get_artifact_payload(workspace, path)
        self._send_json(payload, status=200 if payload.get("ok") else 404)

    def _api_media(self, query: dict[str, list[str]]) -> None:
        name = _q(query, "workspace")
        abs_path = _q(query, "abs")
        if not name or not abs_path:
            return self.send_error(400, "workspace and abs required")
        workspace = self.server.registry.resolve(name)
        if workspace is None:
            return self.send_error(404, "workspace not found")
        target = _safe_within_workspace(workspace, abs_path)
        if target is None or not target.is_file():
            return self.send_error(404, "file not found")
        mime, _ = mimetypes.guess_type(str(target))
        mime = mime or "application/octet-stream"
        size = target.stat().st_size
        start, end = 0, max(size - 1, 0)
        range_header = self.headers.get("Range", "")
        partial = False
        if range_header.startswith("bytes=") and size:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                raw_start, raw_end = match.groups()
                if raw_start:
                    start = min(int(raw_start), size - 1)
                    end = min(int(raw_end), size - 1) if raw_end else size - 1
                elif raw_end:
                    length = min(int(raw_end), size)
                    start, end = size - length, size - 1
                partial = start <= end
        length = end - start + 1 if size else 0
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        try:
            with target.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining:
                    chunk = fh.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Normal when a browser cancels a video range request.
            return
        except OSError as exc:
            LOGGER.warning("media read failed for %s: %s", target, exc)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "frame-ancestors 'self' *")
        self.end_headers()
        self.wfile.write(body)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _q(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _safe_within_workspace(workspace: Path, abs_path: str) -> Path | None:
    """Resolve *abs_path* and ensure it stays within the workspace root.

    The path may arrive as ``/workspace/<name>/relative/path`` (the legacy
    convention used by ``role_character.json`` and friends) or as a plain
    filesystem path.  Both forms are normalised.
    """
    if not abs_path:
        return None
    p = Path(abs_path)
    if not p.is_absolute():
        # treat as workspace-relative
        p = workspace / abs_path.lstrip("/")
    # Preserve pipeline-created symlinks while still rejecting lexical
    # traversal. The 0916 workspace intentionally links shot clips to the
    # source run; dereferencing before the boundary check would hide them.
    resolved_workspace = Path(os.path.abspath(workspace))
    resolved_target = Path(os.path.abspath(p))
    try:
        resolved_target.relative_to(resolved_workspace)
    except ValueError:
        LOGGER.warning("media path outside workspace rejected: %s", resolved_target)
        return None
    return resolved_target


def _compute_kpis(workspace: Path, stages: list[str]) -> dict[str, Any]:
    """Lightweight per-workspace counters for the sidebar."""
    out: dict = {"character_count": 0, "tracklet_count": 0,
                 "assigned_tracklet_count": 0, "voice_count": 0,
                 "shot_count": 0, "segment_count": 0,
                 "caption_count": 0, "artifact_count": 0,
                 "scene_count": 0, "global_clip_count": 0,
                 "key_object_count": 0, "i2i_bundle_count": 0}
    try:
        out["character_count"] = len(R._list_character_entities(workspace))
    except Exception:
        pass
    # tracklet count
    shot_rows: list[Path] = []
    for base in ("assets/visual_person_v2/tracklets/shots",
                 "assets/visual_person/tracklets/shots",
                 "tracklets/shots"):
        d = workspace / base
        if d.is_dir():
            shot_rows = sorted(d.iterdir())
            break
    for shot_dir in shot_rows:
        if not (shot_dir / "tracklet_signatures.json").is_file():
            continue
        try:
            payload = _safe_read_json(shot_dir / "tracklet_signatures.json") or {}
            out["tracklet_count"] += len(payload.get("tracklet_signatures", []) or [])
        except Exception:
            continue
    # alignment
    if (workspace / "global_alignment" / "global_tracklet_alignment.jsonl").is_file():
        for row in _load_jsonl(workspace / "global_alignment" / "global_tracklet_alignment.jsonl"):
            try:
                out["assigned_tracklet_count"] += int(row.get("assigned_tracklet_count", 0) or 0)
            except (TypeError, ValueError):
                pass
    counters = {
        "voice_count": R.STAGE_VOICE,
        "shot_count": R.STAGE_SHOTS,
        "segment_count": R.STAGE_SEGMENTS,
        "global_clip_count": R.STAGE_GLOBAL_CLIPS,
        "key_object_count": R.STAGE_KEY_OBJECT,
        "i2i_bundle_count": R.STAGE_I2I_BUNDLES,
    }
    for key, stage in counters.items():
        if stage in stages:
            try:
                out[key] = len(R.list_entities(workspace, stage))
            except Exception:
                pass
    try:
        out["caption_count"] = sum(
            len(R.list_entities(workspace, stage))
            for stage in (R.STAGE_CAPTIONS_SHOT, R.STAGE_CAPTIONS_GLOBAL,
                          R.STAGE_CAPTIONS_SEGMENT)
            if stage in stages
        )
    except Exception:
        pass
    try:
        out["scene_count"] = sum(
            len((R.get_scene_payload(workspace, entity["id"], scene_opt=False).get("scenes") or []))
            for entity in R.list_entities(workspace, R.STAGE_SCENE)
        ) if R.STAGE_SCENE in stages else 0
    except Exception:
        pass
    try:
        out["artifact_count"] = sum(
            int(group.get("file_count", 0)) for group in R.list_artifact_groups(workspace)
        )
    except Exception:
        pass
    return out


def _entities_for_overview(workspace: Path, stages: list[str]) -> list[dict[str, Any]]:
    """Pick the entities for the default landing page.

    We always expose Stage 5 characters when present because that's the
    most useful overview; otherwise fall back to the first available
    stage so the page never starts blank when artefacts exist.
    """
    if R.STAGE_CHARACTERS in stages:
        return R.list_entities(workspace, R.STAGE_CHARACTERS)
    if stages:
        return R.list_entities(workspace, stages[0])
    return []


# --------------------------------------------------------------------------- #
# server bootstrap
# --------------------------------------------------------------------------- #
class ViewerConfig:
    host: str
    port: int
    workspace_root: Path
    default_workspace: str
    only_workspace: str
    base_url: str
    open_browser: bool

    def __init__(self, host: str, port: int, workspace_root: Path,
                 default_workspace: str, only_workspace: str,
                 base_url: str, open_browser: bool) -> None:
        self.host = host
        self.port = port
        self.workspace_root = workspace_root
        self.default_workspace = default_workspace
        self.only_workspace = only_workspace
        self.base_url = base_url
        self.open_browser = open_browser


class ViewerServer(ThreadingHTTPServer):
    """Threaded HTTP server hosting the asset viewer dashboard."""

    def __init__(self, config: ViewerConfig) -> None:
        self.config = config
        self.registry = WorkspaceRegistry(config.workspace_root, config.only_workspace)
        self.default_workspace = config.default_workspace
        # Bind, then capture the actual port (useful when port=0).
        super().__init__((config.host, config.port), ViewerRequestHandler)
        self.registry.refresh()
        # If the requested default doesn't exist, fall back to the first
        # discovered workspace.
        if self.default_workspace not in self.registry.names() and self.registry.names():
            self.default_workspace = self.registry.names()[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT), type=Path)
    parser.add_argument("--default-workspace", default="", type=str)
    parser.add_argument("--only-workspace", default="", type=str,
                        help="Expose only this workspace and hide all sibling runs")
    parser.add_argument("--base-url", default=os.environ.get("ASSET_VIEWER_BASE_URL", ""),
                        help="Public reverse-proxy path prefix, e.g. "
                             "/gemini_web/asset_viewer/~host~port~/JOB_ID/")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    config = ViewerConfig(
        host=args.host,
        port=args.port,
        workspace_root=args.workspace_root,
        default_workspace=args.default_workspace,
        only_workspace=args.only_workspace,
        base_url=args.base_url,
        open_browser=not args.no_browser,
    )
    server = ViewerServer(config)
    host = server.server_address[0]
    port = server.server_port
    internal_url = f"http://{host}:{port}/"
    public_url = f"http://58.59.115.26:31443{config.base_url}" if config.base_url else None
    summary = {
        "internal_url": internal_url,
        "public_url": public_url,
        "host": host,
        "port": port,
        "workspace_root": str(config.workspace_root.resolve()),
        "default_workspace": server.default_workspace,
        "workspace_count": len(server.registry.names()),
        "workspaces": server.registry.list_payload(),
        "note": (
            "The public URL is the internal URL with the Gemini reverse-proxy "
            "prefix appended. The platform authorizes at the proxy; no token "
            "is required."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if config.open_browser and host in ("127.0.0.1", "0.0.0.0", "localhost"):
        try:
            webbrowser.open(internal_url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("interrupted; shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
