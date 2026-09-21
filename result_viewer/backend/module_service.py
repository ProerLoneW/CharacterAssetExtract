# -*- coding: utf-8 -*-
"""模块读取服务:把文档快照、运行绑定与适配器组合为统一结果。

- /runs/{run}/modules 的每模块状态/指标/可用筛选;
- items / item detail 的调度与 ResultEnvelope 封装;
- 产物 preview(JSONL 行范围)。
"""

from __future__ import annotations

from .adapters import get_adapter
from .adapters.base import RunContext
from .config import RuntimeConfig
from .document_store import DocumentStore
from .envelope import (
    ApiError,
    BAD_REQUEST,
    DOC_REVISION_EXPIRED,
    NOT_FOUND,
    RUN_FORBIDDEN,
    build_envelope,
    decode_cursor,
)
from .media_service import MediaService
from .run_registry import RunDescriptor, RunRegistry

_RUN_CONTEXTS: dict[str, RunContext] = {}


def run_context(registry: RunRegistry, run_id: str) -> RunContext:
    ctx = _RUN_CONTEXTS.get(run_id)
    if ctx is None:
        run = registry.get(run_id)
        if run is None:
            raise ApiError(RUN_FORBIDDEN, f"运行未授权或不存在: {run_id}", status_code=404)
        ctx = RunContext(run)
        _RUN_CONTEXTS[run_id] = ctx
    return ctx


class ModuleService:
    def __init__(self, config: RuntimeConfig, store: DocumentStore, runs: RunRegistry, media: MediaService, adapters: dict):
        self.config = config
        self.store = store
        self.runs = runs
        self.media = media
        self.adapters = adapters

    # -- 内部工具 -----------------------------------------------------------

    def _snapshot(self, doc_revision: str | None):
        snap = self.store.snapshot_for(doc_revision)
        if snap is None or not snap.ok:
            if doc_revision is not None:
                raise ApiError(
                    DOC_REVISION_EXPIRED,
                    "请求的 doc_revision 已过期或不可用,请重新获取 spec",
                    status_code=409,
                    retryable=True,
                )
            from .envelope import DOCUMENT_UNAVAILABLE

            raise ApiError(DOCUMENT_UNAVAILABLE, "文档不可用:没有已通过校验的快照", status_code=503, retryable=True)
        return snap

    def _run(self, run_id: str) -> RunDescriptor:
        run = self.runs.get(run_id)
        if run is None:
            raise ApiError(RUN_FORBIDDEN, f"运行未授权或不存在: {run_id}", status_code=404)
        return run

    def _module_meta(self, snap, module_id: str) -> dict:
        for m in snap.spec["modules"]:
            if m["id"] == module_id:
                return m
        raise ApiError(NOT_FOUND, f"模块不存在: {module_id}", status_code=404, module_id=module_id)

    # -- API 实现 -----------------------------------------------------------

    def spec_payload(self, doc_revision: str | None = None) -> dict:
        snap = self._snapshot(doc_revision)
        spec = snap.spec
        return {
            "api_version": "1.0",
            "doc_revision": snap.doc_revision,
            "compiled_at": snap.compiled_at,
            "system": {k: v for k, v in spec["system"].items() if not k.startswith("_")},
            "module_count": spec["module_count"],
            "modules": [
                {
                    "id": m["id"],
                    "title": m["title"],
                    "order": m["order"],
                    "group": m["group"],
                    "lifecycle": m["lifecycle"],
                    "implementation": m["implementation"],
                    "summary": m["summary"],
                    "depends_on": m["depends_on"],
                    "optional_depends_on": m["optional_depends_on"],
                    "inputs": m["inputs"],
                    "outputs": m["outputs"],
                    "view": m["view"],
                    "sections": m["sections"],
                    "source_refs": m["source_refs"],
                    "evidence": m["evidence"],
                }
                for m in spec["modules"]
            ],
            "groups": [
                g for g in spec["system"]["groups"]
            ],
            "routes": spec["system"]["routes"],
            "content_sections": spec["content_sections"],
        }

    def runs_payload(self) -> dict:
        out = []
        for run in self.runs.all_runs():
            ctx = run_context(self.runs, run.run_id)
            out.append(
                {
                    "run_id": run.run_id,
                    "video_id": run.video_id,
                    "title": run.title,
                    "layout_profile": run.layout_profile,
                    "route_id": run.route_id,
                    "produced_code_revision": run.produced_code_revision,
                    "created_at": run.created_at,
                    "data_revision": ctx.data_revision(),
                    "scopes": ctx.scopes(),
                }
            )
        return {"api_version": "1.0", "doc_revision": self.store.revision(), "runs": out}

    def modules_payload(self, run_id: str, doc_revision: str | None) -> dict:
        snap = self._snapshot(doc_revision)
        run = self._run(run_id)
        ctx = run_context(self.runs, run_id)
        modules = []
        for meta in snap.spec["modules"]:
            adapter = get_adapter(self.adapters, meta["view"]["adapter"])
            probe = adapter.probe(run, ctx)
            entry = {
                "module_id": meta["id"],
                "title": meta["title"],
                "order": meta["order"],
                "group": meta["group"],
                "lifecycle": meta["lifecycle"],
                "view_enabled": meta["view"]["enabled"],
                "renderer": meta["view"]["renderer"],
                "adapter": meta["view"]["adapter"],
                "filters": meta["view"]["filters"],
                "primary_artifact": meta["view"]["primary_artifact"],
                "status": {"execution": probe.execution, "quality": probe.quality, "reason": probe.reason},
                "bound_artifacts": [
                    {"artifact_id": aid, "available": run.resolve_binding_path(aid) is not None}
                    for aid in self._module_artifact_ids(meta)
                    if run.binding_for(aid) is not None
                ],
            }
            if probe.execution in ("succeeded", "partial"):
                summary = adapter.summary(run, ctx)
                entry["status"] = summary.status
                entry["metrics"] = [_metric_to_dict(m) for m in summary.metrics]
                entry["filters_available"] = summary.filters_available
                entry["scopes"] = summary.scopes
                entry["warnings"] = summary.warnings
            else:
                entry["metrics"] = []
                entry["filters_available"] = {k: [] for k in meta["view"]["filters"]}
                entry["scopes"] = ctx.scopes()
                entry["warnings"] = []
            modules.append(entry)
        return {
            "api_version": "1.0",
            "doc_revision": snap.doc_revision,
            "data_revision": ctx.data_revision(),
            "run_id": run_id,
            "modules": modules,
        }

    def items_payload(
        self,
        run_id: str,
        module_id: str,
        doc_revision: str | None,
        filters: dict[str, str | None],
        cursor: str | None,
        limit: int,
    ) -> dict:
        snap = self._snapshot(doc_revision)
        run = self._run(run_id)
        ctx = run_context(self.runs, run_id)
        meta = self._module_meta(snap, module_id)
        allowed_filters = set(meta["view"]["filters"])
        unknown = [k for k in filters if k not in allowed_filters]
        # 文档三 §3.5:筛选项仅按模块 view.filters 渲染;未知筛选直接拒绝,避免静默忽略
        clean = {}
        for k, v in filters.items():
            if k in allowed_filters and v not in (None, ""):
                clean[k] = v
        adapter = get_adapter(self.adapters, meta["view"]["adapter"])
        offset = decode_cursor(cursor)
        items, page = adapter.list_items(run, ctx, clean, offset, limit)
        probe = adapter.probe(run, ctx)
        summary = None
        if probe.execution in ("succeeded", "partial", "unknown"):
            try:
                summary = adapter.summary(run, ctx)
            except Exception:
                summary = None
        page_notes = page.pop("note", None)
        return build_envelope(
            doc_revision=snap.doc_revision,
            data_revision=ctx.data_revision(),
            adapter_version=f"{adapter.adapter_id}/{adapter.adapter_version}",
            run_id=run_id,
            module_id=module_id,
            status=(summary.status if summary else {"execution": probe.execution, "quality": probe.quality, "reason": probe.reason}),
            metrics=(summary.metrics if summary else []),
            items=items,
            page=page,
            warnings=(list(dict.fromkeys(ctx.warnings)) + ([page_notes] if page_notes else []))[:20],
            provenance={
                "artifact_ids": self._module_artifact_ids(meta),
                "filters_applied": clean,
                "filters_rejected": unknown,
            },
        )

    def item_payload(self, run_id: str, module_id: str, entity_encoded: str, doc_revision: str | None) -> dict:
        from .envelope import EntityKey

        snap = self._snapshot(doc_revision)
        run = self._run(run_id)
        ctx = run_context(self.runs, run_id)
        meta = self._module_meta(snap, module_id)
        key = EntityKey.decode(entity_encoded)
        adapter = get_adapter(self.adapters, meta["view"]["adapter"])
        item = adapter.get_item(run, ctx, key)
        if item is None:
            raise ApiError(NOT_FOUND, f"实体不存在或不属于该模块: {key.canonical()}", status_code=404, module_id=module_id)
        return build_envelope(
            doc_revision=snap.doc_revision,
            data_revision=ctx.data_revision(),
            adapter_version=f"{adapter.adapter_id}/{adapter.adapter_version}",
            run_id=run_id,
            module_id=module_id,
            status={"execution": "succeeded", "quality": item.get("status", {}).get("quality", "unreviewed")},
            items=[item],
            page={"limit": 1, "next_cursor": None, "total": 1},
            provenance={"artifact_ids": self._module_artifact_ids(meta)},
        )

    def artifact_preview_payload(self, run_id: str, artifact_id: str, offset: int, limit: int) -> dict:
        run = self._run(run_id)
        path = run.resolve_binding_path(artifact_id)
        if path is None:
            raise ApiError(NOT_FOUND, f"产物未绑定或文件不存在: {artifact_id}", status_code=404, artifact_id=artifact_id)
        if path.suffix.lower() not in (".jsonl", ".json"):
            raise ApiError(BAD_REQUEST, "仅支持 JSON/JSONL 文本产物的受控预览", status_code=400, artifact_id=artifact_id)
        if path.suffix.lower() == ".json":
            data = path.read_text(encoding="utf-8", errors="replace")
            lines = data.splitlines()
            window = lines[offset : offset + limit]
            return {
                "artifact_id": artifact_id,
                "kind": "json",
                "offset": offset,
                "limit": limit,
                "total_lines": len(lines),
                "lines": window,
            }
        rows, warnings = [], []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if lineno <= offset:
                    continue
                if len(rows) >= limit:
                    break
                rows.append({"line": lineno, "text": line.rstrip("\n")[:4000]})
        return {
            "artifact_id": artifact_id,
            "kind": "jsonl",
            "offset": offset,
            "limit": limit,
            "warnings": warnings,
            "lines": rows,
        }

    def lineage_payload(self, run_id: str, entity_encoded: str, doc_revision: str | None) -> dict:
        snap = self._snapshot(doc_revision)
        self._run(run_id)
        from .envelope import EntityKey

        key = EntityKey.decode(entity_encoded)
        # 一跳溯源:实体所在 scope 的上游清单 + 模块级依赖(文档三 §7 默认一跳)
        scope_id = key.scope_id
        modules = []
        for meta in snap.spec["modules"]:
            modules.append(
                {
                    "module_id": meta["id"],
                    "title": meta["title"],
                    "depends_on": meta["depends_on"],
                    "optional_depends_on": meta["optional_depends_on"],
                }
            )
        return {
            "api_version": "1.0",
            "doc_revision": snap.doc_revision,
            "run_id": run_id,
            "entity": {"kind": key.kind, "scope_id": scope_id, "local_id": key.local_id},
            "scope_id": scope_id,
            "note": "模块级一跳依赖;实体级关系见各模块 item 的 relations 字段",
            "modules": modules,
        }

    def _module_artifact_ids(self, meta: dict) -> list[str]:
        return [o["id"] for o in meta.get("outputs", [])]


def _metric_to_dict(m) -> dict:
    from dataclasses import asdict

    return asdict(m)
