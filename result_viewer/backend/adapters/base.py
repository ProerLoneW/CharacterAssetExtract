# -*- coding: utf-8 -*-
"""适配器协议与共享基础设施(文档三 §4)。

Adapter 只读真实 schema 并归一化为 Entity/媒体引用;不做文件搜索、不做 schema 猜测。
RunContext 为同一运行内的适配器提供共享的清单缓存(shots/globals/scope 映射),
缓存键包含输入指纹与 adapter 版本,文件变化才重新解析(文档三 §6.4)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..envelope import ApiError, UNSUPPORTED_SCHEMA, entity_to_dict
from ..media_service import MediaService
from ..run_registry import ArtifactBinding, RunDescriptor


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    execution: str  # unconfigured/not_run/partial/succeeded/failed/unknown
    quality: str = "unreviewed"
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleSummary:
    status: dict[str, Any]
    metrics: list[dict]
    filters_available: dict[str, list]
    scopes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    item_count: int | None = None


class ModuleAdapter:
    """所有数据适配器的基类。子类声明 adapter_id/module_id/required_artifacts。"""

    adapter_id: str = ""
    adapter_version: str = "0.1.0"
    module_id: str = ""
    required_artifacts: tuple[str, ...] = ()

    def __init__(self, media: MediaService):
        self.media = media

    # -- 生命周期探针 -------------------------------------------------------

    def probe(self, run: RunDescriptor, ctx: "RunContext") -> ProbeResult:
        missing = [a for a in self.required_artifacts if run.binding_for(a) is None]
        if missing:
            return ProbeResult(
                execution="unconfigured",
                reason=f"该运行未绑定产物: {', '.join(missing)}",
            )
        absent = [a for a in self.required_artifacts if run.resolve_binding_path(a) is None]
        if absent:
            return ProbeResult(
                execution="not_run",
                reason=f"产物已绑定但文件不存在或不可读: {', '.join(absent)}",
            )
        return ProbeResult(execution="succeeded")

    # -- 数据接口 -----------------------------------------------------------

    def summary(self, run: RunDescriptor, ctx: "RunContext") -> ModuleSummary:
        raise NotImplementedError

    def list_items(
        self,
        run: RunDescriptor,
        ctx: "RunContext",
        filters: dict[str, str | None],
        cursor: int,
        limit: int,
    ) -> tuple[list[dict], dict]:
        raise NotImplementedError

    def get_item(self, run: RunDescriptor, ctx: "RunContext", entity_key) -> dict | None:
        raise NotImplementedError


class RegisteredOnlyAdapter(ModuleAdapter):
    """已登记能力但尚未实现数据解析的 adapter。

    文档校验要求文档二的 adapter 名必须能解析到注册能力;这类适配器绑定后
    也不会假装成功——返回 unconfigured/unsupported_schema,提示需要扩展实现。
    """

    def __init__(self, adapter_id: str, module_id: str, note: str, media: MediaService):
        super().__init__(media)
        self.adapter_id = adapter_id
        self.module_id = module_id
        self.note = note

    def probe(self, run: RunDescriptor, ctx: "RunContext") -> ProbeResult:
        missing = [a for a in self.required_artifacts if run.binding_for(a) is None]
        if missing:
            return ProbeResult(execution="unconfigured", reason=f"该运行未绑定产物: {', '.join(missing)}")
        return ProbeResult(
            execution="unknown",
            reason=f"adapter {self.adapter_id} 已注册但本版本未实现数据解析({self.note})",
        )

    def summary(self, run: RunDescriptor, ctx: "RunContext") -> ModuleSummary:
        p = self.probe(run, ctx)
        return ModuleSummary(
            status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
            metrics=[],
            filters_available={},
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        p = self.probe(run, ctx)
        page = {"limit": limit, "next_cursor": None, "total": 0}
        return (
            [],
            page | {"note": p.reason},
        )

    def get_item(self, run, ctx, entity_key):
        return None


# ---------------------------------------------------------------------------
# 运行上下文:共享清单缓存
# ---------------------------------------------------------------------------


def file_fingerprint(path: Path | None) -> tuple[float, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return None


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """容错读取 JSONL:坏行/半行记录行号与警告,保留其余行(文档三 §6.4)。"""
    rows: list[dict] = []
    warnings: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path.name}:{lineno}: 行损坏,已跳过({exc.msg})")
                continue
            if not isinstance(obj, dict):
                warnings.append(f"{path.name}:{lineno}: 非 JSON 对象行,已跳过")
                continue
            rows.append(obj)
    return rows, warnings


def shot_canonical_id(raw: Any) -> str | None:
    """shot ID 归一化:768 / 'SHOT_00768' / 'SHOT_1080P_SHOT_00768' -> 'SHOT_00768'。"""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return f"SHOT_{raw:05d}"
    text = str(raw)
    m = None
    for pattern in (r"SHOT_(\d{3,})$", r"shot_(\d{3,})$"):
        m = re.search(pattern, text)
        if m:
            break
    if m:
        return f"SHOT_{int(m.group(1)):05d}"
    if text.isdigit():
        return f"SHOT_{int(text):05d}"
    return None


def scope_canonical_id(global_raw: Any) -> str | None:
    if isinstance(global_raw, bool):
        return None
    if isinstance(global_raw, int):
        return f"global_{global_raw:04d}"
    text = str(global_raw)
    m = re.search(r"(\d+)$", text)
    if m:
        return f"global_{int(m.group(1)):04d}"
    return None


class RunContext:
    """按 (mtime,size) 指纹缓存共享清单;跨请求复用,指纹变化自动失效。"""

    def __init__(self, run: RunDescriptor):
        self.run = run
        self._cache: dict[str, tuple[Any, Any]] = {}
        self.warnings: list[str] = []

    # -- 指纹 ------------------------------------------------------------

    def data_revision(self) -> str:
        parts: list[str] = []
        for artifact_id in sorted(self.run.bindings):
            path = self.run.resolve_binding_path(artifact_id)
            fp = file_fingerprint(path)
            parts.append(f"{artifact_id}:{fp}")
        import hashlib

        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    # -- 共享清单 -----------------------------------------------------------

    def _cached(self, key: str, loader):
        path, loader = loader
        fp = file_fingerprint(path)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]
        value = loader() if path is not None and fp is not None else None
        self._cache[key] = (fp, value)
        return value

    def cached_jsonl(self, artifact_id: str, cache_key: str | None = None):
        """按产物 ID 读取 JSONL,带指纹缓存;返回 rows 或 None(未绑定/不可读)。"""
        key = cache_key or f"jsonl:{artifact_id}"
        path = self.run.resolve_binding_path(artifact_id)
        if path is None:
            return None

        def load():
            rows, warns = read_jsonl(path)
            self.warnings.extend(warns)
            return rows

        return self._cached(key, (path, load))

    def globals_rows(self) -> list[dict] | None:
        return self.cached_jsonl("globals", "globals")

    def shots_rows(self) -> list[dict] | None:
        return self.cached_jsonl("shots", "shots")

    def scopes(self) -> list[str]:
        rows = self.globals_rows()
        if not rows:
            return []
        out = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if sid and sid not in out:
                out.append(sid)
        return out

    def scope_meta(self) -> dict[str, dict]:
        """scope_id -> {global_id, source_range, duration, shot_ids, clip_path_raw}"""
        rows = self.globals_rows() or []
        meta: dict[str, dict] = {}
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if not sid:
                continue
            meta[sid] = {
                "global_id": row.get("global_id"),
                "source_range": row.get("time_range_sec"),
                "duration_sec": row.get("duration_sec"),
                "shot_ids": [shot_canonical_id(s) for s in row.get("shot_ids") or []],
                "raw": row,
            }
        return meta

    def shot_to_scope(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for sid, meta in self.scope_meta().items():
            for shot in meta["shot_ids"]:
                if shot:
                    mapping[shot] = sid
        return mapping

    def shots_in_scope(self, scope_id: str | None) -> list[str]:
        meta = self.scope_meta()
        if scope_id and scope_id in meta:
            return [s for s in meta[scope_id]["shot_ids"] if s]
        rows = self.shots_rows() or []
        out = []
        for row in rows:
            sid = shot_canonical_id(row.get("shot_id"))
            if sid:
                out.append(sid)
        return out

    def subject_registry(self, scope_id: str) -> list[dict]:
        """从全局描述提取该 scope 的人物注册表(供 subject 筛选与展示)。"""
        rows = self.cached_jsonl("global_captions", "global_captions") or []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if sid == scope_id:
                inner = row.get("global_caption") or {}
                reg = inner.get("subject_registry")
                if isinstance(reg, list):
                    return [r for r in reg if isinstance(r, dict)]
        return []


def entity_page(entities: list, cursor: int, limit: int, to_dict=entity_to_dict) -> tuple[list[dict], dict]:
    window = entities[cursor : cursor + limit]
    next_cursor = cursor + limit if cursor + limit < len(entities) else None
    page = {
        "limit": limit,
        "next_cursor": next_cursor,
        "total": len(entities),
    }
    return [to_dict(e) for e in window], page
