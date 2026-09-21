# -*- coding: utf-8 -*-
"""统一返回结构与错误码(文档三 §6.3/§7)。

ResultEnvelope:api_version/doc_revision/data_revision/adapter_version/run_id/module_id/
status{execution,quality,reason}/metrics[]/items[]/page/warnings/provenance。
错误:{error:{code,message,request_id,module_id,artifact_id,location,retryable},doc_revision}。
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

API_VERSION = "1.0"

# 文档三 §7 中的示例错误码
DOCUMENT_UNAVAILABLE = "DOCUMENT_UNAVAILABLE"
DOCUMENT_INVALID = "DOCUMENT_INVALID"
UNSUPPORTED_PROTOCOL = "UNSUPPORTED_PROTOCOL"
UNREGISTERED_ADAPTER = "UNREGISTERED_ADAPTER"
AMBIGUOUS_ARTIFACT = "AMBIGUOUS_ARTIFACT"
UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
MEDIA_MISSING = "MEDIA_MISSING"
MEDIA_FORBIDDEN = "MEDIA_FORBIDDEN"
DOC_REVISION_EXPIRED = "DOC_REVISION_EXPIRED"
NOT_FOUND = "NOT_FOUND"
BAD_REQUEST = "BAD_REQUEST"
RUN_FORBIDDEN = "RUN_FORBIDDEN"


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        module_id: str | None = None,
        artifact_id: str | None = None,
        location: dict | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.module_id = module_id
        self.artifact_id = artifact_id
        self.location = location
        self.retryable = retryable

    def to_payload(self, doc_revision: str | None) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": uuid.uuid4().hex[:12],
                "module_id": self.module_id,
                "artifact_id": self.artifact_id,
                "location": self.location,
                "retryable": self.retryable,
            },
            "doc_revision": doc_revision,
        }


# ---------------------------------------------------------------------------
# 实体键:文档一 §6.5 的复合对象 {kind, scope_id, local_id}
# ---------------------------------------------------------------------------

_KEY_CHARSET_OK = re.compile(r"^[A-Za-z0-9_.\-:=]+$")


@dataclass
class EntityKey:
    kind: str
    scope_id: str | None
    local_id: str

    def canonical(self) -> str:
        return f"{self.kind}:{self.scope_id or '-'}:{self.local_id}"

    def encode(self) -> str:
        raw = self.canonical().encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, encoded: str) -> "EntityKey":
        try:
            pad = "=" * (-len(encoded) % 4)
            raw = base64.urlsafe_b64decode(encoded + pad).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            raise ApiError(BAD_REQUEST, "entity 参数不是合法的实体键编码", status_code=400)
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ApiError(BAD_REQUEST, f"entity 参数结构非法: {raw[:60]}", status_code=400)
        kind, scope, local = parts
        return cls(kind=kind, scope_id=None if scope == "-" else scope, local_id=local)


# ---------------------------------------------------------------------------
# 结果信封
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    id: str
    label: str
    value: Any
    unit: str = ""
    denominator: Any = None
    granularity: str = "selected-run"
    meaning: str = ""


@dataclass
class MediaRef:
    media_id: str | None
    kind: str  # video/audio/image
    label: str
    mime: str = ""
    duration_sec: float | None = None
    availability: str = "available"  # available/missing/forbidden/unconfigured
    detail: str = ""


@dataclass
class TimeRef:
    timebase: str
    start_sec: float | None
    end_sec: float | None
    source_range: list[float] | None = None
    media_id: str | None = None


@dataclass
class Entity:
    key: EntityKey
    title: str
    status: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    media_refs: list[MediaRef] = field(default_factory=list)
    time_refs: list[TimeRef] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def media_ref_to_dict(m: MediaRef) -> dict:
    d = asdict(m)
    if m.media_id:
        d["url"] = f"media/{m.media_id}"
    return d


def entity_to_dict(e: Entity, include_fields: bool = True) -> dict:
    return {
        "entity_id": e.key.encode(),
        "key": {"kind": e.key.kind, "scope_id": e.key.scope_id, "local_id": e.key.local_id},
        "title": e.title,
        "status": e.status,
        "summary": e.summary,
        "media_refs": [media_ref_to_dict(m) for m in e.media_refs],
        "time_refs": [asdict(t) for t in e.time_refs],
        "relations": e.relations,
        "fields": e.fields if include_fields else {},
        "provenance": e.provenance,
    }


def build_envelope(
    *,
    doc_revision: str | None,
    data_revision: str,
    adapter_version: str,
    run_id: str,
    module_id: str,
    status: dict,
    metrics: list[Metric] | None = None,
    items: list[dict] | None = None,
    page: dict | None = None,
    warnings: list[str] | None = None,
    provenance: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    env = {
        "api_version": API_VERSION,
        "doc_revision": doc_revision,
        "data_revision": data_revision,
        "adapter_version": adapter_version,
        "run_id": run_id,
        "module_id": module_id,
        "status": status,
        "metrics": [asdict(m) for m in (metrics or [])],
        "items": items or [],
        "page": page,
        "warnings": warnings or [],
        "provenance": provenance or {},
    }
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# 分页游标:只读数据的偏移游标
# ---------------------------------------------------------------------------


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        pad = "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(cursor + pad))
        offset = int(data["o"])
        return max(0, offset)
    except Exception:
        raise ApiError(BAD_REQUEST, "cursor 参数非法", status_code=400)
