# -*- coding: utf-8 -*-
"""声音资产适配器:角色声音候选(voice_preview / voice-preview-v2)。

按文档二 §16 的状态语义:
- review_status=valid/invalid 是旧分类(valid=候选,不是审核通过);
- 只有存在 voice_asset_manifest.jsonl 时才有 pending/approved 审核态(本运行未生成);
- WAV 路径来自 review_audio_relative_path,相对 voice_by_role_src 根。
"""

from __future__ import annotations

from ..envelope import Entity, EntityKey, MediaRef, Metric, TimeRef
from ..run_registry import RunDescriptor
from .base import ModuleAdapter, RunContext, entity_page, scope_canonical_id

ADMISSION_LABELS = {
    "pending_audio_quality_validation": "待音频质量校验",
    "admitted": "已准入",
}


class VoicePreviewAdapter(ModuleAdapter):
    adapter_id = "voice-preview-v2"
    module_id = "voice_preview"
    required_artifacts = ("voice_previews",)

    def _voice_root(self, run: RunDescriptor):
        binding = run.binding_for("voice_previews")
        if binding is None:
            return None
        root = binding.absolute_path(run.roots)
        return root.parent if root is not None else None

    def _rows(self, ctx: RunContext):
        return ctx.cached_jsonl("voice_previews", "voice_previews")

    def summary(self, run: RunDescriptor, ctx: RunContext):
        rows = self._rows(ctx)
        if rows is None:
            p = self.probe(run, ctx)
            from .base import ModuleSummary

            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "subject": []},
                scopes=ctx.scopes(),
            )
        candidates = [r for r in rows if r.get("audio_asset_candidate")]
        cut_ok = [r for r in rows if r.get("cut_success")]
        subjects = sorted({str(r.get("speaker_id")) for r in rows if r.get("speaker_id")})
        metrics = [
            Metric(id="preview_total", label="ASR 声音预览", value=len(rows), unit="条", meaning="按 (global_id, asr_id) 去重的切片清单行数"),
            Metric(id="candidates", label="候选音频", value=len(candidates), unit="条", denominator=len(rows), meaning="audio_asset_candidate=true;仅候选资格,不表示人工审核通过"),
            Metric(id="cut_success", label="切片成功", value=len(cut_ok), unit="条", denominator=len(rows), meaning="cut_success=true 的行数"),
        ]
        return _mk(metrics, {"scope": ctx.scopes(), "subject": subjects}, ctx.scopes(), list(ctx.warnings))

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = self._rows(ctx) or []
        scope = filters.get("scope")
        subject = filters.get("subject")
        status_f = filters.get("status")
        voice_root = self._voice_root(run)
        entities = []
        for row in rows:
            sid = scope_canonical_id(row.get("global_id"))
            if scope and sid != scope:
                continue
            spk = str(row.get("speaker_id") or "unknown")
            if subject and spk != subject:
                continue
            asr_id = str(row.get("asr_id") or "")
            is_candidate = bool(row.get("audio_asset_candidate"))
            if status_f == "candidate" and not is_candidate:
                continue
            if status_f == "non_candidate" and is_candidate:
                continue
            rel = row.get("review_audio_relative_path")
            media: MediaRef
            if rel and voice_root is not None and (voice_root / rel).is_file():
                media_id = self.media.issue(run, "run_root", _posix(voice_root, rel, run))
                media = MediaRef(media_id=media_id, kind="audio", label="ASR 切片音频", mime="audio/wav", duration_sec=(row.get("signal_stats") or {}).get("duration_sec"), availability="available")
            else:
                media = MediaRef(media_id=None, kind="audio", label="ASR 切片音频", availability="missing", detail="切片文件未收录到本地")
            tr = row.get("time_range_sec") or [None, None]
            src = row.get("source_time_range_sec") or [None, None]
            rejection = row.get("audio_asset_rejection_reasons") or []
            quality = "candidate" if is_candidate else "rejected"
            entities.append(
                Entity(
                    key=EntityKey(kind="voice_preview", scope_id=sid, local_id=f"{spk}:{asr_id}"),
                    title=f"{asr_id} · {sid}:{spk}",
                    status={
                        "execution": "succeeded" if row.get("cut_success") else "failed",
                        "quality": quality,
                        "candidate": is_candidate,
                        "reason": "、".join(rejection) if rejection else None,
                    },
                    summary=(row.get("text") or "")[:120],
                    media_refs=[media],
                    time_refs=[
                        TimeRef(timebase="global_local", start_sec=tr[0], end_sec=tr[1], media_id=media.media_id),
                        TimeRef(timebase="source", start_sec=src[0], end_sec=src[1]),
                    ],
                    relations=[
                        {"type": "subject", "module_id": "global_describe", "label": f"{sid} 人物注册表({spk})"},
                        {"type": "scope", "module_id": "global_group", "label": f"全局片段 {sid}"},
                    ],
                    fields={
                        "asr_id": asr_id,
                        "speaker_id": spk,
                        "text": row.get("text"),
                        "language": row.get("language"),
                        "attribution_confidence": row.get("attribution_confidence"),
                        "attribution_evidence_type": row.get("attribution_evidence_type"),
                        "attribution_evidence": row.get("attribution_evidence"),
                        "speaker_attribution_accepted": row.get("speaker_attribution_accepted"),
                        "audio_asset_candidate": is_candidate,
                        "audio_asset_admission_status": row.get("audio_asset_admission_status"),
                        "audio_asset_rejection_reasons": rejection,
                        "overlap_status": row.get("overlap_status"),
                        "is_off_screen": row.get("is_off_screen"),
                        "review_status": row.get("review_status"),
                        "review_status_meaning": "valid/invalid 是候选分类,不是审核结论",
                        "signal_stats": row.get("signal_stats"),
                        "cut_error": row.get("cut_error"),
                    },
                    provenance={"artifact_ids": ["voice_previews"], "wav_relative_path": rel},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "voice_preview":
            return None
        spk, _, asr = entity_key.local_id.partition(":")
        items, _ = self.list_items(run, ctx, {"subject": spk}, 0, 200)
        for it in items:
            if it["fields"].get("asr_id") == asr:
                return it
        return None


def _posix(voice_root, rel: str, run: RunDescriptor) -> str:
    """计算 rel 相对运行根(root_id=run_root)的相对路径。"""
    try:
        return (voice_root / rel).resolve().relative_to(run.roots["run_root"].path).as_posix()
    except (KeyError, ValueError):
        return rel.replace("\\", "/")


def _mk(metrics, filters, scopes, warnings):
    from .base import ModuleSummary

    return ModuleSummary(
        status={"execution": "succeeded", "quality": "unreviewed", "reason": None},
        metrics=metrics,
        filters_available=filters,
        scopes=scopes,
        warnings=warnings,
    )
