# -*- coding: utf-8 -*-
"""人物资产适配器:人物轨迹提取(tracklets-v2)、旧链角色身份绑定(characters-legacy-v1)。

轨迹:root manifest 汇总 + 每 shot 的 tracklet_signatures;画廊图片路径是旧运行
绝对路径,经运行重写规则映射回本地 observations/。
旧链角色:role_alignment.json(投票/冲突) + characters/<scope>/<S-ID>/role_character.json
(视觉资产清单);声音准入缺席时显示 voice_skipped_reason,不伪造审核状态。
"""

from __future__ import annotations

import re

from ..envelope import Entity, EntityKey, MediaRef, Metric, TimeRef
from ..run_registry import RunDescriptor
from .base import (
    ModuleAdapter,
    ModuleSummary,
    ProbeResult,
    RunContext,
    entity_page,
    read_json,
    shot_canonical_id,
)

SHOT_DIR_RE = re.compile(r"SHOT_(?:.*_)?SHOT_(\d+)$")


def _shot_dir_shot_id(dir_name: str) -> str | None:
    """SHOT_1080P_SHOT_00768 -> SHOT_00768"""
    m = SHOT_DIR_RE.match(dir_name)
    if m:
        return f"SHOT_{int(m.group(1)):05d}"
    m2 = re.search(r"(\d{3,})$", dir_name)
    if m2:
        return f"SHOT_{int(m2.group(1)):05d}"
    return None


class TrackletsAdapter(ModuleAdapter):
    """人物轨迹证据目录(person_tracks)。"""

    adapter_id = "tracklets-v2"
    module_id = "person_tracks"
    required_artifacts = ("tracklets",)

    def _root(self, run: RunDescriptor):
        path = run.resolve_binding_path("tracklets")
        return path if path is not None and path.is_dir() else None

    def _root_manifest(self, run: RunDescriptor, ctx: RunContext):
        root = self._root(run)
        if root is None:
            return None
        key = "tracklets_root_manifest"
        fp = (root / "manifest.json").stat()
        ck = (key, fp.st_mtime_ns, fp.st_size)
        hit = ctx._cache.get(ck)
        if hit is not None:
            return hit[1]
        value = read_json(root / "manifest.json")
        ctx._cache[ck] = (None, value)
        return value

    def _shot_dirs(self, run: RunDescriptor) -> list[tuple[str, object]]:
        root = self._root(run)
        if root is None:
            return []
        out = []
        for child in sorted((root / "shots").iterdir()):
            if child.is_dir() and (child / "tracklet_signatures.json").is_file():
                out.append((child.name, child))
        return out

    def _signatures(self, run: RunDescriptor, ctx: RunContext, shot_dir):
        key = f"tracklets_sig:{shot_dir.name}"
        fp = (shot_dir / "tracklet_signatures.json").stat()
        ck = (key, fp.st_mtime_ns, fp.st_size)
        hit = ctx._cache.get(ck)
        if hit is not None:
            return hit[1]
        value = read_json(shot_dir / "tracklet_signatures.json")
        ctx._cache[ck] = (None, value)
        return value

    def probe(self, run: RunDescriptor, ctx: RunContext):
        p = super().probe(run, ctx)
        if p.execution == "succeeded":
            root = self._root(run)
            if root is None or not (root / "manifest.json").is_file():
                return ProbeResult(execution="not_run", reason="tracklets 根缺少 manifest.json")
        return p

    def summary(self, run: RunDescriptor, ctx: RunContext):
        manifest = self._root_manifest(run, ctx)
        if manifest is None:
            p = self.probe(run, ctx)
            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "shot": []},
                scopes=ctx.scopes(),
            )
        tracklet_count = manifest.get("tracklet_count")
        filtered = manifest.get("filtered_tracklet_count")
        shots_completed = manifest.get("shots_completed")
        metrics = [
            Metric(id="tracklets", label="轨迹数量", value=tracklet_count, unit="条", meaning="保留轨迹计数;不是人物数量"),
            Metric(id="filtered_tracklets", label="过滤轨迹", value=filtered, unit="条", meaning="证据不足被过滤的轨迹,不计入展示主体"),
            Metric(id="shots_completed", label="完成镜头", value=shots_completed, unit="个", meaning="成功产出轨迹清单的镜头数"),
        ]
        shots = [_shot_dir_shot_id(name) for name, _ in self._shot_dirs(run)]
        return ModuleSummary(
            status={"execution": "succeeded" if manifest.get("status") == "ok" else "partial", "quality": "unreviewed", "reason": None},
            metrics=metrics,
            filters_available={"scope": ctx.scopes(), "shot": [s for s in shots if s]},
            scopes=ctx.scopes(),
            warnings=list(ctx.warnings),
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        scope = filters.get("scope")
        shot_filter = filters.get("shot")
        allowed = set(ctx.shots_in_scope(scope)) if scope else None
        shot_to_scope = ctx.shot_to_scope()
        entities = []
        for name, shot_dir in self._shot_dirs(run):
            sid = _shot_dir_shot_id(name)
            if sid is None:
                continue
            if allowed is not None and sid not in allowed:
                continue
            if shot_filter and sid != shot_filter:
                continue
            sig = self._signatures(run, ctx, shot_dir)
            sigs = sig.get("tracklet_signatures") or []
            overlay_rel = self._media_under_run(run, shot_dir / "tracklet_overlay.mp4")
            overlay = (
                MediaRef(media_id=self.media.issue(run, "run_root", overlay_rel), kind="video", label="带框轨迹视频", mime="video/mp4", availability="available")
                if overlay_rel
                else MediaRef(media_id=None, kind="video", label="带框轨迹视频", availability="missing", detail="overlay 未收录")
            )
            entities.append(
                Entity(
                    key=EntityKey(kind="shot_tracklets", scope_id=shot_to_scope.get(sid), local_id=sid),
                    title=f"镜头 {sid} 轨迹",
                    status={"execution": "succeeded"},
                    summary=f"{len(sigs)} 条轨迹 · 带框视频{'可播' if overlay_rel else '缺失'}",
                    media_refs=[overlay],
                    relations=[
                        {"type": "shot", "module_id": "shot_split", "label": f"镜头 {sid} 结构信息"},
                        {"type": "caption", "module_id": "shot_describe", "label": "镜头描述"},
                        {"type": "align", "module_id": "person_align", "label": "全局归属(本运行未收录)"},
                    ],
                    fields={
                        "shot_dir": name,
                        "tracklet_count": len(sigs),
                        "tracklets": [
                            {
                                "tracklet_id": t.get("tracklet_id"),
                                "start_time": t.get("start_time"),
                                "end_time": t.get("end_time"),
                                "observation_count": len(t.get("observation_ids") or []),
                                "face_count": len(t.get("face_gallery") or []),
                                "body_count": len(t.get("body_gallery") or []),
                                "best_quality": (t.get("best_quality") or {}),
                            }
                            for t in sigs
                        ],
                    },
                    provenance={"artifact_ids": ["tracklets"], "schema_version": sig.get("schema_version")},
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind not in ("shot_tracklets", "tracklet"):
            return None
        if entity_key.kind == "shot_tracklets":
            items, _ = self.list_items(run, ctx, {"shot": entity_key.local_id}, 0, 1)
            if not items:
                return None
            base = items[0]
            # 展开该镜头所有轨迹的画廊
            sig = self._signatures_for_shot(run, ctx, entity_key.local_id)
            if sig:
                galleries = []
                for t in sig.get("tracklet_signatures") or []:
                    face = self._gallery_media(run, t.get("face_gallery") or [])
                    body = self._gallery_media(run, t.get("body_gallery") or [])
                    galleries.append(
                        {
                            "tracklet_id": t.get("tracklet_id"),
                            "start_time": t.get("start_time"),
                            "end_time": t.get("end_time"),
                            "cannot_links": len(t.get("cannot_links") or []),
                            "face": face[:6],
                            "body": body[:6],
                        }
                    )
                base["fields"]["tracklet_galleries"] = galleries
            return base
        # 单条轨迹:local_id = shot:tracklet
        shot_sid, _, tracklet_id = entity_key.local_id.partition(":")
        sig = self._signatures_for_shot(run, ctx, shot_sid)
        if not sig:
            return None
        for t in sig.get("tracklet_signatures") or []:
            if t.get("tracklet_id") == tracklet_id:
                return {
                    "entity_id": entity_key.encode(),
                    "key": {"kind": entity_key.kind, "scope_id": entity_key.scope_id, "local_id": entity_key.local_id},
                    "title": tracklet_id,
                    "status": {"execution": "succeeded", "quality": "unreviewed"},
                    "summary": f"{shot_sid} 内轨迹 · {t.get('start_time')}–{t.get('end_time')} 秒",
                    "media_refs": self._gallery_media(run, t.get("face_gallery") or [])[:8]
                    + self._gallery_media(run, t.get("body_gallery") or [])[:8],
                    "time_refs": [
                        TimeRef(timebase="shot_local", start_sec=t.get("start_time"), end_sec=t.get("end_time")).__dict__
                    ],
                    "relations": [
                        {"type": "shot", "module_id": "person_tracks", "label": f"所属镜头 {shot_sid}"},
                    ],
                    "fields": {
                        "tracklet_id": tracklet_id,
                        "observation_count": len(t.get("observation_ids") or []),
                        "cannot_links": t.get("cannot_links") or [],
                        "quality": t.get("quality") or t.get("best_quality") or {},
                    },
                    "provenance": {"artifact_ids": ["tracklets"]},
                }
        return None

    # -- 内部 -----------------------------------------------------------

    def _signatures_for_shot(self, run: RunDescriptor, ctx: RunContext, shot_sid: str):
        for name, shot_dir in self._shot_dirs(run):
            if _shot_dir_shot_id(name) == shot_sid:
                return self._signatures(run, ctx, shot_dir)
        return None

    def _media_under_run(self, run: RunDescriptor, absolute: object) -> str | None:
        try:
            resolved = run.roots["run_root"].path.joinpath(absolute).resolve() if not absolute.is_absolute() else absolute.resolve()
        except AttributeError:
            return None
        rel = None
        for root in run.roots.values():
            try:
                rel = resolved.relative_to(root.path).as_posix()
                break
            except ValueError:
                continue
        return rel

    def _gallery_media(self, run: RunDescriptor, gallery: list[dict]) -> list[MediaRef]:
        out: list[MediaRef] = []
        for item in gallery:
            raw = item.get("crop_path")
            if not raw:
                continue
            resolved = run.rewrite_media_path(str(raw))
            if resolved is None:
                out.append(
                    MediaRef(
                        media_id=None,
                        kind="image",
                        label=f"裁图 {item.get('observation_id', '')[-8:]}",
                        availability="missing",
                        detail="源运行裁图未收录",
                    )
                )
                continue
            root_id, target = resolved
            rel = self._media_under_run(run, target)
            if rel is None:
                continue
            out.append(
                MediaRef(
                    media_id=self.media.issue(run, "run_root", rel),
                    kind="image",
                    label=f"{item.get('observation_id', '裁图')[-12:]}",
                    mime="image/jpeg",
                    availability="available",
                )
            )
        return out


class LegacyCharactersAdapter(ModuleAdapter):
    """旧链角色身份绑定(legacy_bind):role_alignment + 角色目录。"""

    adapter_id = "characters-legacy-v1"
    module_id = "legacy_bind"
    required_artifacts = ("role_binding",)

    def _alignment(self, run: RunDescriptor, ctx: RunContext):
        path = run.resolve_binding_path("role_binding")
        if path is None:
            return None
        key = "role_alignment"
        fp = path.stat()
        ck = (key, fp.st_mtime_ns, fp.st_size)
        hit = ctx._cache.get(ck)
        if hit is not None:
            return hit[1]
        value = read_json(path)
        ctx._cache[ck] = (None, value)
        return value

    def _character_dirs(self, run: RunDescriptor):
        root = run.resolve_binding_path("legacy_characters")
        if root is None or not root.is_dir():
            return []
        out = []
        for scope_dir in sorted(root.iterdir()):
            if not scope_dir.is_dir():
                continue
            for subj_dir in sorted(scope_dir.iterdir()):
                rc = subj_dir / "role_character.json"
                if subj_dir.is_dir() and rc.is_file():
                    out.append((scope_dir.name, subj_dir.name, subj_dir))
        return out

    def summary(self, run: RunDescriptor, ctx: RunContext):
        alignment = self._alignment(run, ctx)
        if alignment is None:
            p = self.probe(run, ctx)
            return ModuleSummary(
                status={"execution": p.execution, "quality": p.quality, "reason": p.reason},
                metrics=[],
                filters_available={"scope": ctx.scopes(), "subject": []},
                scopes=ctx.scopes(),
            )
        dirs = self._character_dirs(run)
        metrics = [
            Metric(id="character_count", label="物化角色", value=alignment.get("character_count"), unit="个", meaning="role_alignment 声明的 CHR 数(含未物化)"),
            Metric(id="materialized", label="角色目录", value=len(dirs), unit="个", meaning="characters/<scope>/<S-ID> 实际目录数;以 scope:S-ID 为单位"),
            Metric(id="pending_candidates", label="待定身份", value=alignment.get("pending_candidate_count"), unit="个", meaning="未晋升 CHR 的 pending 候选,不是确认角色"),
        ]
        subjects = sorted({subj for _, subj, _ in dirs})
        return ModuleSummary(
            status={"execution": "succeeded", "quality": "unreviewed", "reason": None},
            metrics=metrics,
            filters_available={"scope": ctx.scopes(), "subject": subjects},
            scopes=ctx.scopes(),
            warnings=list(ctx.warnings),
        )

    def list_items(self, run, ctx, filters, cursor, limit):
        scope = filters.get("scope")
        subject = filters.get("subject")
        alignment = self._alignment(run, ctx) or {}
        entities = []
        for scope_name, subj, subj_dir in self._character_dirs(run):
            if scope and scope_name != scope:
                continue
            if subject and subj != subject:
                continue
            rc = read_json(subj_dir / "role_character.json")
            alignment_info = rc.get("alignment") or {}
            images = self._character_images(run, subj_dir, rc)
            voice_reason = rc.get("voice_skipped_reason")
            metrics = {
                "face": sum(1 for m in images if "face" in m.label),
                "body": sum(1 for m in images if "body" in m.label),
                "representative": sum(1 for m in images if "representative" in m.label),
            }
            entities.append(
                Entity(
                    key=EntityKey(kind="legacy_character", scope_id=scope_name, local_id=subj),
                    title=f"{scope_name}:{subj}",
                    status={
                        "execution": "succeeded",
                        "quality": "unreviewed",
                        "binding": alignment_info.get("status"),
                        "voice": "skipped" if voice_reason else "none",
                        "reason": f"声音未交付: {voice_reason}" if voice_reason else None,
                    },
                    summary=f"视觉身份 {rc.get('visual_character_id')} · 投票 {alignment_info.get('vote_share')} · {metrics['face']} 脸图/{metrics['body']} 身体图",
                    media_refs=images[:24],
                    relations=[
                        {"type": "scope", "module_id": "global_describe", "label": f"{scope_name} 人物注册表"},
                        {"type": "voice", "module_id": "voice_preview", "label": f"{subj} 的候选声音预览"},
                        {"type": "identity", "module_id": "legacy_identity", "label": f"视觉身份 {rc.get('visual_character_id')}(本运行详情未接入)"},
                    ],
                    fields={
                        "semantic_scope_id": rc.get("semantic_scope_id"),
                        "subject_id": rc.get("subject_id"),
                        "visual_character_id": rc.get("visual_character_id"),
                        "alignment": alignment_info,
                        "visual_asset_count": len(rc.get("visual_assets") or []),
                        "voice_asset_count": len(rc.get("voice_assets") or []),
                        "voice_skipped_reason": voice_reason,
                        "image_counts": metrics,
                    },
                    provenance={
                        "artifact_ids": ["role_binding", "legacy_characters"],
                        "schema_version": alignment.get("schema_version"),
                    },
                )
            )
        return entity_page(entities, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        if entity_key.kind != "legacy_character":
            return None
        items, _ = self.list_items(run, ctx, {"scope": entity_key.scope_id, "subject": entity_key.local_id}, 0, 1)
        if not items:
            return None
        item = items[0]
        # detailed: 附 role_alignment 中该 CHR 的跨 scope 视图
        alignment = self._alignment(run, ctx) or {}
        chr_id = item["fields"].get("visual_character_id")
        entry = (alignment.get("chr_to_roles") or {}).get(chr_id)
        if entry:
            item["fields"]["chr_by_scope"] = entry.get("by_scope")
            item["fields"]["chr_tracklet_count"] = entry.get("tracklet_count")
        return item

    def _character_images(self, run: RunDescriptor, subj_dir, role_character: dict):
        out: list[MediaRef] = []
        for rel in role_character.get("visual_assets") or []:
            if not isinstance(rel, str) or rel.endswith(".json"):
                continue
            target = (subj_dir / "visual" / rel).resolve()
            if not target.is_file():
                continue
            root_rel = None
            for root in run.roots.values():
                try:
                    root_rel = target.relative_to(root.path).as_posix()
                    break
                except ValueError:
                    continue
            if root_rel is None:
                continue
            kind = "representative" if "representative" in rel else ("face" if "face" in rel else "body")
            out.append(
                MediaRef(
                    media_id=self.media.issue(run, "run_root", root_rel),
                    kind="image",
                    label=rel.split("/")[-1],
                    mime="image/jpeg",
                    availability="available",
                    detail=kind,
                )
            )
        order = {"representative": 0, "face": 1, "body": 2}
        out.sort(key=lambda m: order.get(m.detail or "body", 3))
        return out
