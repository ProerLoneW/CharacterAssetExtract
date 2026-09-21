# -*- coding: utf-8 -*-
"""媒体服务(文档三 §8)。

media_id 是服务端签发的 opaque 标识:base64url(JSON{run_id, root_id, relative_path})。
客户端不能提交任意绝对路径;解析后必须落在该运行允许的来源根内。
支持 HEAD/Range(206/416/Content-Range/Accept-Ranges),以便视频/音频拖动。
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .envelope import ApiError, MEDIA_FORBIDDEN, MEDIA_MISSING
from .run_registry import RunDescriptor

MIME_OVERRIDES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}

_ALLOWED_SUFFIXES = set(MIME_OVERRIDES) | {".m4a", ".aac", ".ogg", ".gif", ".bmp", ".csv", ".log", ".yaml", ".yml"}

SAFE_REL_RE = re.compile(r"^[A-Za-z0-9_\-./\u4e00-\u9fff]+$")


def make_media_id(run_id: str, root_id: str, relative_path: str) -> str:
    payload = json.dumps({"r": run_id, "k": root_id, "p": relative_path}, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


@dataclass
class MediaResolution:
    run_id: str
    root_id: str
    relative_path: str
    file_path: Path
    mime: str
    size: int
    etag: str


class MediaService:
    def __init__(self, runs: dict[str, RunDescriptor]):
        self.runs = runs

    # -- 签发 -----------------------------------------------------------

    def issue(self, run: RunDescriptor, root_id: str, relative_path: str) -> str:
        return make_media_id(run.run_id, root_id, relative_path)

    def issue_local(self, run: RunDescriptor, absolute_path: Path, label_root: str = "run_root") -> str | None:
        """把已确认位于受控根内的绝对文件转为 media_id;不在受控根内返回 None。"""
        target = absolute_path.resolve()
        for root_id, root in run.roots.items():
            if root.contains(target):
                rel = target.relative_to(root.path).as_posix()
                return make_media_id(run.run_id, root_id, rel)
        return None

    # -- 解析 -----------------------------------------------------------

    def resolve(self, media_id: str) -> MediaResolution:
        try:
            pad = "=" * (-len(media_id) % 4)
            payload = json.loads(base64.urlsafe_b64decode(media_id + pad))
            run_id, root_id, rel = payload["r"], payload["k"], payload["p"]
        except (binascii.Error, UnicodeDecodeError, KeyError, json.JSONDecodeError):
            raise ApiError(MEDIA_FORBIDDEN, "media_id 非法", status_code=404)

        run = self.runs.get(run_id)
        if run is None:
            raise ApiError(MEDIA_FORBIDDEN, "media_id 引用未授权的运行", status_code=404)

        if not SAFE_REL_RE.match(rel) or ".." in Path(rel).parts or Path(rel).is_absolute():
            raise ApiError(MEDIA_FORBIDDEN, "媒体相对路径非法", status_code=404)

        root = run.roots.get(root_id)
        if root is None:
            raise ApiError(MEDIA_FORBIDDEN, "media_id 引用未登记的来源根", status_code=404)

        candidate = (root.path / rel).resolve()
        if not root.contains(candidate):
            raise ApiError(MEDIA_FORBIDDEN, "媒体路径越出受控来源根", status_code=404)
        if not candidate.is_file():
            raise ApiError(MEDIA_MISSING, "媒体文件不存在或已移除", status_code=404)
        if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ApiError(MEDIA_FORBIDDEN, f"不支持的媒体类型: {candidate.suffix}", status_code=404)

        stat = candidate.stat()
        mime = MIME_OVERRIDES.get(candidate.suffix.lower()) or mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        etag = f'"{uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{root_id}:{rel}:{stat.st_mtime_ns}:{stat.st_size}").hex}"'
        return MediaResolution(
            run_id=run_id,
            root_id=root_id,
            relative_path=rel,
            file_path=candidate,
            mime=mime,
            size=stat.st_size,
            etag=etag,
        )

    # -- Range 响应参数 -----------------------------------------------------------

    @staticmethod
    def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
        """解析 Range 头;返回 [start, end] 闭区间;后缀/越界返回 None(由调用方回 416)。"""
        if not header:
            return None
        m = re.match(r"^bytes=(\d*)-(\d*)$", header.strip())
        if not m:
            return None
        start_s, end_s = m.group(1), m.group(2)
        if start_s == "" and end_s == "":
            return None
        if start_s == "":
            # 后缀范围:最后 N 字节
            length = int(end_s)
            if length <= 0:
                return None
            start = max(0, size - length)
            return (start, size - 1)
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        if start >= size or start > end:
            return None
        return (start, min(end, size - 1))


def iter_file_chunks(path: Path, start: int, end: int, chunk_size: int = 1024 * 512):
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def is_media_suffix(suffix: str) -> bool:
    return suffix.lower() in _ALLOWED_SUFFIXES
