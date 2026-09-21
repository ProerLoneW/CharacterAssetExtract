# -*- coding: utf-8 -*-
"""语义描述适配器:全局描述(global-caption-v1)、镜头描述(shot-caption-v1)、
片段描述(segment-caption-v1)。

处理文档三 §4 列出的真实差异:
- global/shot caption 的同名嵌套:global_caption.global_caption / shot_caption.shot_caption;
- segment_captions 一行是 global:展开 segments[],实体键 (scope, segment_id);
- 保留 raw 层级供审计(detailed 视图 / 开发抽屉)。
"""

from __future__ import annotations

from ..envelope import Entity, EntityKey, Metric, TimeRef
from ..run_registry import RunDescriptor
from .base import (
    ModuleAdapter,
    RunContext,
    entity_page,
    read_jsonl,
    scope_canonical_id,
    shot_canonical_id,
)


def _load_rows(ctx: RunContext, artifact_id: str):
    """统一入口:RunContext 带指纹缓存的 JSONL 读取。"""
    return ctx.cached_jsonl(artifact_id, artifact_id)


def _subject_chip(reg: dict) -> str:
    sid = reg.get("subject_id", "?")
    desc = (reg.get("identity_description") or "").strip()
    tier = reg.get("importance_tier") or ""
    return f"{sid} · {desc[:60]}{' · ' + tier if tier else ''}"


class GlobalCaptionAdapter(ModuleAdapter):
    """全局描述与说话归属(global_describe)。

    主产物 global_captions;attributed_asr 作为第二产物在 detail 中提供对白与候选证据。
    """

    adapter_id = "global-caption-v1"
    module_id = "global_describe"
    required_artifacts = ("global_captions",)

    def _asr_rows(self, ctx: RunContext):
        return ctx.cached_jsonl("attributed_asr", "attributed_asr")

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows = _load_rows(ctx, "global_captions")
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "subject": []},
                scopes=ctx.scopes(),
            )
        ok = [r for r in rows if r.get("success")]
        asr = self._asr_rows(ctx) or []
        candidates = [a for a in asr if a.get("audio_asset_candidate")]
        metrics = [
            Metric(id="caption_rows", label="全局描述", value=len(ok), unit="条", denominator=len(rows), meaning="success=true 的行数/总行数"),
            Metric(id="asr_entries", label="归属对白", value=len(asr), unit="条", meaning="speaker_attributed_asr 按 (global_id, asr_id) 去重的行数"),
            Metric(id="audio_candidates", label="候选音频", value=len(candidates), unit="条", denominator=len(asr), meaning="audio_asset_candidate=true;候选资格,不是审核通过"),
        ]
        subjects: list[str] = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            for reg in ctx.subject_registry(sid or ""):
                if reg.get("subject_id") not in subjects:
                    subjects.append(reg.get("subject_id"))
        return _mk_summary(metrics, {"scope": ctx.scopes(), "subject": subjects}, ctx.scopes())

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = _load_rows(ctx, "global_captions")
        rows = rows or []
        scope = filters.get("scope")
        subject = filters.get("subject")
        entities = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if scope and sid != scope:
                continue
            inner = row.get("global_caption") or {}
            registry = [r for r in (inner.get("subject_registry") or []) if isinstance(r, dict)]
            if subject:
                if not any(r.get("subject_id") == subject for r in registry):
                    continue
            deep = inner.get("global_caption") or {}
            description = deep.get("global_description") or ""
            media = None
            fields = {
                "global_id": row.get("global_id"),
                "model": row.get("model"),
                "annotation_confidence": inner.get("annotation_confidence"),
                "schema_version": inner.get("schema_version"),
                "global_description": description,
                "global_scene_description": deep.get("global_scene_description"),
                "overall_soundscape": deep.get("overall_soundscape"),
                "non_diegetic_music": deep.get("non_diegetic_music"),
                "key_topics": deep.get("key_topics") or [],
                "subjects": [
                    {
                        "subject_id": r.get("subject_id"),
                        "identity_description": r.get("identity_description"),
                        "visual_cues": r.get("visual_cues") or [],
                        "voice_cues": r.get("voice_cues") or [],
                        "importance_tier": r.get("importance_tier"),
                        "role_label": r.get("role_label"),
                        "first_seen_sec": r.get("first_seen_sec"),
                        "last_seen_sec": r.get("last_seen_sec"),
                        "confidence": r.get("confidence"),
                    }
                    for r in registry
                ],
                "event_timeline": (inner.get("global_event_timeline") or [])[:60],
                "asr_speaker_attributions": (inner.get("asr_speaker_attributions") or [])[:200],
                "quality_issues": inner.get("quality_issues"),
            }
            entities.append(
                Entity(
                    key=EntityKey(kind="global_caption", scope_id=sid, local_id=sid or str(row.get("global_id"))),
                    title=f"{sid} 全局描述",
                    status={"execution": "succeeded" if row.get("success") else "failed"},
                    summary=(description or "(无描述)")[0:140],
                    time_refs=[TimeRef(timebase="global_local", start_sec=0.0, end_sec=inner.get("video_duration_sec"))],
                    relations=[
                        {"type": "scope", "module_id": "global_group", "label": f"全局片段 {sid}"},
                        {"type": "dialogue", "module_id": "voice_preview", "label": "按 ASR 切出的声音候选"},
                        {"type": "shots", "module_id": "shot_describe", "label": "镜头描述(继承此注册表)"},
                    ],
                    fields=fields,
                    provenance={
                        "artifact_ids": ["global_captions", "attributed_asr"],
                        "model": row.get("model"),
                        "raw_keys": list(row.keys()),
                    },
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "global_caption":
            return None
        items, _ = self.list_items(run, ctx, {"scope": entity_key.scope_id}, 0, 1)
        return items[0] if items else None


class ShotCaptionAdapter(ModuleAdapter):
    """镜头描述(shot_describe)。"""

    adapter_id = "shot-caption-v1"
    module_id = "shot_describe"
    required_artifacts = ("shot_captions",)

    def _rows(self, ctx: RunContext):
        return _load_rows(ctx, "shot_captions")

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows = self._rows(ctx)
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "shot": []},
                scopes=ctx.scopes(),
            )
        ok = [r for r in rows if r.get("success")]
        obj_candidates = sum(len(((r.get("shot_caption") or {}).get("key_object_candidates")) or []) for r in rows)
        metrics = [
            Metric(id="caption_ok", label="成功镜头描述", value=len(ok), unit="个", denominator=len(rows), meaning="success=true 行数/清单行数"),
            Metric(id="object_candidates", label="关键物体候选", value=obj_candidates, unit="个", meaning="各镜头 key_object_candidates 之和;仅提案,不代表已定位"),
        ]
        shots = [shot_canonical_id(r.get("shot_id")) for r in rows]
        return _mk_summary(metrics, {"scope": ctx.scopes(), "shot": [s for s in shots if s][:300]}, ctx.scopes())

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = self._rows(ctx)
        rows = rows or []
        scope = filters.get("scope")
        shot_filter = filters.get("shot")
        allowed = set(ctx.shots_in_scope(scope)) if scope else None
        shot_to_scope = ctx.shot_to_scope()
        entities = []
        for row in rows:
            sid = shot_canonical_id(row.get("shot_id"))
            if sid is None:
                continue
            if allowed is not None and sid not in allowed:
                continue
            if shot_filter and sid != shot_filter:
                continue
            inner = row.get("shot_caption") or {}
            deep = inner.get("shot_caption") or {}
            description = deep.get("shot_description") or ""
            obj = inner.get("key_object_candidates") or deep.get("key_object_candidates") or []
            entities.append(
                Entity(
                    key=EntityKey(kind="shot_caption", scope_id=shot_to_scope.get(sid), local_id=sid),
                    title=f"镜头 {sid} 描述",
                    status={"execution": "succeeded" if row.get("success") else "failed"},
                    summary=(description or "(无描述)")[0:140],
                    time_refs=[TimeRef(timebase="shot_local", start_sec=0.0, end_sec=row.get("duration_sec"))],
                    relations=[
                        {"type": "shot", "module_id": "shot_split", "label": f"镜头 {sid} 结构信息"},
                        *(
                            [{"type": "scope", "module_id": "global_describe", "label": "所属全局描述与人物注册表"}]
                            if shot_to_scope.get(sid)
                            else []
                        ),
                    ],
                    fields={
                        "shot_id": sid,
                        "shot_description": description,
                        "shot_scene_description": deep.get("shot_scene_description"),
                        "overall_soundscape": deep.get("overall_soundscape"),
                        "shot_type": inner.get("shot_type"),
                        "global_subject_ids": row.get("global_subject_ids"),
                        "subjects": inner.get("subjects") or [],
                        "key_object_candidates": [
                            {
                                "local_object_id": o.get("local_object_id"),
                                "canonical_name": o.get("canonical_name"),
                                "grounding_phrases": o.get("grounding_phrases") or [],
                                "priority": o.get("priority"),
                                "visible_time_range_sec": o.get("visible_time_range_sec"),
                            }
                            for o in obj
                            if isinstance(o, dict)
                        ],
                        "dialogue": inner.get("dialogue") or [],
                        "on_screen_text": inner.get("on_screen_text") or [],
                    },
                    provenance={"artifact_ids": ["shot_captions"], "model": row.get("model")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "shot_caption":
            return None
        items, _ = self.list_items(run, ctx, {"shot": entity_key.local_id}, 0, 1)
        return items[0] if items else None


class SegmentCaptionAdapter(ModuleAdapter):
    """片段描述(segment_describe):一行 global 展开 segments[]。"""

    adapter_id = "segment-caption-v1"
    module_id = "segment_describe"
    required_artifacts = ("segment_captions",)

    def _expanded(self, ctx: RunContext):
        rows = _load_rows(ctx, "segment_captions")
        if rows is None:
            return None
        expanded = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            inner = row.get("segment_caption") or {}
            for seg in inner.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                expanded.append((sid, seg, row))
        return expanded

    def summary(self, run: RunDescriptor, ctx: RunContext):
        expanded = self._expanded(ctx)
        if expanded is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "segment": []},
                scopes=ctx.scopes(),
            )
        ok_globals = {sid for sid, seg, row in expanded if row.get("success")}
        metrics = [
            Metric(id="segments", label="描述片段", value=len(expanded), unit="个", meaning="segment_caption.segments 展开后的段数;不是行数"),
            Metric(id="global_rows_ok", label="成功 global 行", value=len(ok_globals), unit="条", denominator=None, meaning="success=true 的 global 行(展开前)"),
        ]
        seg_ids = sorted({f"{int(s.get('segment_id') or 0):03d}" for _, s, _ in expanded})
        return _mk_summary(metrics, {"scope": ctx.scopes(), "segment": seg_ids}, ctx.scopes(), list(ctx.warnings))

    def list_items(self, run, ctx, filters, cursor, limit):
        expanded = self._expanded(ctx)
        expanded = expanded or []
        scope = filters.get("scope")
        seg_filter = filters.get("segment")
        entities = []
        for sid, seg, row in expanded:
            if scope and sid != scope:
                continue
            seg_label = f"{int(seg.get('segment_id') or 0):03d}"
            if seg_filter and seg_label != seg_filter.zfill(3):
                continue
            tr = seg.get("time_range_sec") or [None, None]
            src = seg.get("source_time_range_sec") or [None, None]
            entities.append(
                Entity(
                    key=EntityKey(kind="segment_caption", scope_id=sid, local_id=seg_label),
                    title=f"{sid}:{seg_label} 片段描述",
                    status={"execution": "succeeded" if row.get("success") else "failed"},
                    summary=(seg.get("description") or seg.get("segment_caption", {}).get("segment_description") or "")[:140],
                    time_refs=[
                        TimeRef(timebase="global_local", start_sec=tr[0], end_sec=tr[1]),
                        TimeRef(timebase="source", start_sec=src[0], end_sec=src[1]),
                    ],
                    relations=[
                        {"type": "media", "module_id": "segment_export", "label": f"片段视频 {sid}:{seg_label}"},
                        {"type": "scope", "module_id": "global_describe", "label": f"全局描述 {sid}"},
                    ],
                    fields={
                        "segment_id": seg.get("segment_id"),
                        "upstream_segment_id": seg.get("upstream_segment_id"),
                        "shot_ids": [shot_canonical_id(s) for s in seg.get("shot_ids") or []],
                        "description": seg.get("description"),
                        "scene_description": seg.get("scene_description"),
                        "event_phase": seg.get("event_phase"),
                        "key_subject_ids": seg.get("key_subject_ids") or [],
                        "dialogue": seg.get("dialogue") or [],
                        "key_object_candidates": seg.get("key_object_candidates") or [],
                        "segment_description": (seg.get("segment_caption") or {}).get("segment_description"),
                        "schema_version": (row.get("segment_caption") or {}).get("schema_version"),
                        "segment_planning_source": (row.get("segment_caption") or {}).get("segment_planning_source"),
                    },
                    provenance={"artifact_ids": ["segment_captions"], "model": row.get("model")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "segment_caption":
            return None
        items, _ = self.list_items(run, ctx, {"scope": entity_key.scope_id, "segment": entity_key.local_id}, 0, 1)
        return items[0] if items else None


def _mk_summary(metrics, filters, scopes=None, warnings=None):
    from .base import ModuleSummary

    return ModuleSummary(
        status={"execution": "succeeded", "quality": "unreviewed", "reason": None},
        metrics=metrics,
        filters_available=filters,
        scopes=scopes or [],
        warnings=warnings or [],
    )
