# -*- coding: utf-8 -*-
"""视频结构适配器:镜头清单(shots-v1)、全局片段(globals-v1)、导出片段(segment-media-v1)。

共同点:实体键使用规范化 shot/scope ID;时间引用区分 source 与 global_local;
媒体路径来自旧 manifest 绝对路径,经运行的重写规则映射回本地受控根,
映射不到时保留 provenance 并显示 missing(文档三 §6.2/§8)。
"""

from __future__ import annotations

from pathlib import Path

from ..envelope import Entity, EntityKey, MediaRef, Metric, TimeRef
from ..run_registry import RunDescriptor
from .base import (
    ModuleAdapter,
    RunContext,
    entity_page,
    read_jsonl,
    scope_canonical_id,
    shot_canonical_id,
)


def _video_media_ref(run: RunDescriptor, adapter: ModuleAdapter, raw_path, kind: str, label: str, duration=None) -> MediaRef:
    if not raw_path:
        return MediaRef(media_id=None, kind=kind, label=label, availability="unconfigured", detail="清单未提供媒体路径")
    resolved = run.rewrite_media_path(str(raw_path))
    if resolved is None:
        return MediaRef(
            media_id=None,
            kind=kind,
            label=label,
            duration_sec=duration,
            availability="missing",
            detail=f"来源运行文件未收录到本地: {Path(str(raw_path)).name}",
        )
    root_id, target = resolved
    media_id = adapter.media.issue_local(run, target, root_id)
    if media_id is None:
        return MediaRef(media_id=None, kind=kind, label=label, availability="forbidden", detail="文件越出受控来源根")
    return MediaRef(
        media_id=media_id,
        kind=kind,
        label=label,
        mime="video/mp4",
        duration_sec=duration,
        availability="available",
    )


class ShotsAdapter(ModuleAdapter):
    """镜头清单(shot_split)。"""

    adapter_id = "shots-v1"
    module_id = "shot_split"
    required_artifacts = ("shots",)

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows = ctx.shots_rows()
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes()},
            )
        detected = sum(1 for r in rows if r.get("boundary_type") == "detected_cut")
        with_media = sum(1 for r in rows if run.rewrite_media_path(str(r.get("clip_path") or "")))
        return _summary_status(
            metrics=[
                Metric(id="shot_count", label="镜头总数", value=len(rows), unit="个", meaning="切镜输出的镜头清单行数"),
                Metric(id="detected_cuts", label="检测切点", value=detected, unit="个", denominator=len(rows), meaning="boundary_type=detected_cut 的行数"),
                Metric(id="local_media", label="本地可播镜头视频", value=with_media, unit="个", denominator=len(rows), meaning="重写规则命中且文件存在的镜头"),
            ],
            filters={"scope": ctx.scopes(), "shot": [shot_canonical_id(r.get("shot_id")) for r in rows][:200]},
            scopes=ctx.scopes(),
            warnings=ctx.warnings,
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = ctx.shots_rows() or []
        scope = filters.get("scope")
        shot_filter = filters.get("shot")
        allowed = set(ctx.shots_in_scope(scope)) if scope else None
        entities: list[Entity] = []
        shot_to_scope = ctx.shot_to_scope()
        for row in rows:
            sid = shot_canonical_id(row.get("shot_id"))
            if sid is None:
                continue
            if allowed is not None and sid not in allowed:
                continue
            if shot_filter and sid != shot_filter:
                continue
            tr = row.get("time_range_sec") or [None, None]
            media = _video_media_ref(run, self, row.get("clip_path"), "video", "镜头视频", row.get("duration_sec"))
            entities.append(
                Entity(
                    key=EntityKey(kind="shot", scope_id=shot_to_scope.get(sid), local_id=sid),
                    title=f"镜头 {sid}",
                    status={"execution": "succeeded" if row.get("clip_path") is not None else "succeeded_no_media"},
                    summary=f"时长 {row.get('duration_sec', 0):.2f} 秒 · {row.get('boundary_type', 'unknown')}",
                    media_refs=[media],
                    time_refs=[
                        TimeRef(timebase="source", start_sec=tr[0], end_sec=tr[1], media_id=media.media_id)
                    ],
                    relations=[
                        {"type": "scope", "module_id": "global_group", "label": f"所属全局片段 {shot_to_scope.get(sid, '未知')}"}
                    ]
                    if shot_to_scope.get(sid)
                    else [],
                    fields={
                        "shot_id": sid,
                        "raw_shot_id": row.get("shot_id"),
                        "duration_sec": row.get("duration_sec"),
                        "frame_range": row.get("frame_range"),
                        "fps": row.get("fps"),
                        "boundary_type": row.get("boundary_type"),
                        "raw_scene_index": row.get("raw_scene_index"),
                    },
                    provenance={"artifact_ids": ["shots"], "source_clip_path": row.get("clip_path")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "shot":
            return None
        rows = ctx.shots_rows() or []
        for row in rows:
            if shot_canonical_id(row.get("shot_id")) == entity_key.local_id:
                items, _ = self.list_items(run, ctx, {"shot": entity_key.local_id}, 0, 1)
                return items[0] if items else None
        return None


class GlobalsAdapter(ModuleAdapter):
    """全局片段清单(global_group);也是全站 scope 选择的数据来源。"""

    adapter_id = "globals-v1"
    module_id = "global_group"
    required_artifacts = ("globals",)

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows = ctx.globals_rows()
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes()},
            )
        total_shots = sum(r.get("shot_count", 0) for r in rows)
        durations = [r.get("duration_sec") or 0 for r in rows]
        return _summary_status(
            metrics=[
                Metric(id="global_count", label="全局片段", value=len(rows), unit="个", meaning="global_clips.jsonl 行数,不是语义场景数"),
                Metric(id="shots_covered", label="覆盖镜头", value=total_shots, unit="个", meaning="各 global shot_count 之和"),
                Metric(
                    id="duration_total",
                    label="总时长",
                    value=round(sum(durations), 2),
                    unit="秒",
                    meaning="各 global duration_sec 之和(本运行收录范围)",
                ),
            ],
            filters={"scope": ctx.scopes()},
            scopes=ctx.scopes(),
            warnings=ctx.warnings,
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = ctx.globals_rows() or []
        scope = filters.get("scope")
        meta = ctx.scope_meta()
        entities = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if sid is None:
                continue
            if scope and sid != scope:
                continue
            tr = row.get("time_range_sec") or [None, None]
            media = _video_media_ref(run, self, row.get("clip_path"), "video", "全局片段视频", row.get("duration_sec"))
            entities.append(
                Entity(
                    key=EntityKey(kind="global", scope_id=sid, local_id=sid),
                    title=f"全局片段 {sid}",
                    status={"execution": "succeeded"},
                    summary=(
                        f"时长 {(row.get('duration_sec') or 0):.1f} 秒 · 含 {row.get('shot_count')} 个镜头 · "
                        f"场景检测 {row.get('scene_detector')}"
                    ),
                    media_refs=[media],
                    time_refs=[TimeRef(timebase="source", start_sec=tr[0], end_sec=tr[1], media_id=media.media_id)],
                    relations=[
                        {"type": "shots", "module_id": "shot_split", "label": f"包含 {row.get('shot_count')} 个镜头"},
                        {"type": "caption", "module_id": "global_describe", "label": "全局描述与人物注册表"},
                    ],
                    fields={
                        "global_id": row.get("global_id"),
                        "shot_count": row.get("shot_count"),
                        "scene_id": row.get("scene_id"),
                        "scene_detector": row.get("scene_detector"),
                        "export_mode": row.get("export_mode"),
                        "source_time_range": tr,
                        "duration_sec": row.get("duration_sec"),
                        "shot_ids": [shot_canonical_id(s) for s in row.get("shot_ids") or []],
                    },
                    provenance={"artifact_ids": ["globals"], "source_clip_path": row.get("clip_path")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "global":
            return None
        items, _ = self.list_items(run, ctx, {"scope": entity_key.scope_id}, 0, 1)
        return items[0] if items else None


class SegmentMediaAdapter(ModuleAdapter):
    """导出片段清单(segment_export):描述与切片的对应。"""

    adapter_id = "segment-media-v1"
    module_id = "segment_export"
    required_artifacts = ("segment_media",)

    def _rows(self, ctx: RunContext):
        path = self.run_path(ctx)
        if path is None:
            return None, []
        rows, warns = read_jsonl(path)
        return rows, warns

    def run_path(self, ctx: RunContext):
        return ctx.run.resolve_binding_path("segment_media")

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows, warns = self._rows(ctx)
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes()},
            )
        ok = [r for r in rows if r.get("success")]
        media_ok = [r for r in rows if run.rewrite_media_path(str(r.get("segment_clip_path") or ""))]
        return _summary_status(
            metrics=[
                Metric(id="segment_count", label="导出片段", value=len(rows), unit="个", meaning="segments.jsonl 行数"),
                Metric(id="cut_success", label="切片成功", value=len(ok), unit="个", denominator=len(rows), meaning="success=true 的行数"),
                Metric(id="local_media", label="本地可播", value=len(media_ok), unit="个", denominator=len(rows), meaning="重写命中且文件存在的片段"),
            ],
            filters={"scope": ctx.scopes(), "segment": sorted({str(r.get("segment_id")) for r in rows})},
            scopes=ctx.scopes(),
            warnings=warns,
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        rows, warns = self._rows(ctx)
        rows = rows or []
        scope = filters.get("scope")
        seg_filter = filters.get("segment")
        entities = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if scope and sid != scope:
                continue
            seg_label = f"{int(row.get('segment_id') or 0):03d}"
            if seg_filter and seg_label != seg_filter.zfill(3):
                continue
            tr = row.get("time_range_sec") or [None, None]
            src = row.get("source_time_range_sec") or [None, None]
            media = _video_media_ref(run, self, row.get("segment_clip_path"), "video", "片段视频", row.get("duration_sec"))
            entities.append(
                Entity(
                    key=EntityKey(kind="segment", scope_id=sid, local_id=seg_label),
                    title=f"片段 {sid or '?'}:{seg_label}",
                    status={"execution": "succeeded" if row.get("success") else "failed"},
                    summary=(row.get("description") or "")[:120],
                    media_refs=[media],
                    time_refs=[
                        TimeRef(timebase="global_local", start_sec=tr[0], end_sec=tr[1], media_id=media.media_id),
                        TimeRef(timebase="source", start_sec=src[0], end_sec=src[1]),
                    ],
                    relations=[
                        {"type": "caption", "module_id": "segment_describe", "label": "片段描述"},
                        *(
                            [{"type": "scope", "module_id": "global_group", "label": f"所属全局片段 {sid}"}]
                            if sid
                            else []
                        ),
                    ],
                    fields={
                        "segment_id": row.get("segment_id"),
                        "shot_ids": [shot_canonical_id(s) for s in row.get("shot_ids") or []],
                        "duration_sec": row.get("duration_sec"),
                        "event_phase": row.get("event_phase"),
                        "description": row.get("description"),
                        "scene_description": row.get("scene_description"),
                    },
                    provenance={"artifact_ids": ["segment_media"], "source_clip_path": row.get("segment_clip_path")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "segment":
            return None
        items, _ = self.list_items(run, ctx, {"scope": entity_key.scope_id, "segment": entity_key.local_id}, 0, 1)
        return items[0] if items else None


def _summary_status(metrics, filters, scopes=None, warnings=None):
    from .base import ModuleSummary

    return ModuleSummary(
        status={"execution": "succeeded", "quality": "unreviewed", "reason": None},
        metrics=metrics,
        filters_available=filters,
        scopes=scopes or [],
        warnings=warnings or [],
    )
