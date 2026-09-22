"""Readers for the current mainline's explicit artifact contracts."""
import re
from pathlib import Path
from ..envelope import Entity, EntityKey, MediaRef, Metric, entity_to_dict
from .base import (ModuleAdapter, ModuleSummary, entity_page, read_json,
                   scope_canonical_id, shot_canonical_id)


class MainlineAdapter(ModuleAdapter):
    def media_ref(self, run, path, label, kind='image'):
        resolved = run.rewrite_media_path(str(path))
        media_id = self.media.issue_local(run, resolved[1]) if resolved else None
        return MediaRef(media_id=media_id, kind=kind, label=label,
                        availability='available' if media_id else 'missing')

    def entities(self, run, ctx):
        raise NotImplementedError

    def summary(self, run, ctx):
        probe = self.probe(run, ctx)
        entities = self.entities(run, ctx) if probe.execution == 'succeeded' else []
        return ModuleSummary(
            status={'execution': probe.execution, 'quality': 'unreviewed', 'reason': probe.reason},
            metrics=[Metric(id='records', label='结果记录', value=len(entities), unit='条',
                            meaning='当前已写入的产物记录数；不代表人工审核通过')],
            filters_available={'scope': ctx.scopes()}, scopes=ctx.scopes(),
            item_count=len(entities), warnings=list(ctx.warnings))

    def list_items(self, run, ctx, filters, cursor, limit):
        rows = self.entities(run, ctx)
        for key, wanted in filters.items():
            if not wanted:
                continue
            if key == 'scope':
                rows = [r for r in rows if r.key.scope_id == wanted]
            elif key in ('subject', 'shot', 'status', 'variant'):
                field = {
                    'subject': 'subject_id', 'shot': 'shot_id',
                    'status': 'status', 'variant': 'variant',
                }[key]
                rows = [r for r in rows if r.fields.get(field) == wanted]
        return entity_page(rows, cursor, limit)

    def get_item(self, run, ctx, entity_key):
        return next((entity_to_dict(r) for r in self.entities(run, ctx)
                     if r.key == entity_key), None)


class SegmentPlanAdapter(MainlineAdapter):
    adapter_id = 'segment-plan-v1'
    module_id = 'segment_plan'
    required_artifacts = ('segment_plan',)

    def entities(self, run, ctx):
        return [Entity(key=EntityKey('segment_plan', scope_canonical_id(r.get('global_id')), str(i)),
                       title=f"片段 {r.get('segment_id', i + 1)}", summary=str(r.get('time_range_sec', '')),
                       status={'execution': 'succeeded'}, fields=r,
                       provenance={'artifact_ids': ['segment_plan']})
                for i, r in enumerate(ctx.cached_jsonl('segment_plan') or [])]


class AsrTimelineAdapter(MainlineAdapter):
    adapter_id = 'asr-v1'
    module_id = 'asr_timeline'
    required_artifacts = ('authoritative_asr',)

    def entities(self, run, ctx):
        return [Entity(key=EntityKey('asr', None, str(r.get('asr_id', i))),
                       title=f"{r.get('start', 0):.2f}–{r.get('end', 0):.2f}s",
                       summary=r.get('text', ''), fields=r, status={'execution': 'succeeded'},
                       provenance={'artifact_ids': ['authoritative_asr']})
                for i, r in enumerate(ctx.cached_jsonl('authoritative_asr') or [])]


class GlobalAlignmentAdapter(MainlineAdapter):
    adapter_id = 'global-alignment-v1'
    module_id = 'person_align'
    required_artifacts = ('direct_alignment',)

    def entities(self, run, ctx):
        source = run.resolve_binding_path('direct_alignment')
        entities = []
        for row in ctx.cached_jsonl('direct_alignment') or []:
            scope = row.get('global_scope_id') or scope_canonical_id(row.get('global_id'))
            video = row.get('stitched_video')
            video_path = (Path(video) if Path(video).is_absolute() else source.parent / 'stitched' / video) if video else None
            refs = [self.media_ref(run, video_path, '人物轨迹归属视频', 'video')] if video_path else []
            entities.append(Entity(
                key=EntityKey('alignment', scope, str(scope)), title=f'{scope} 人物归属',
                status={'execution': 'succeeded' if row.get('status') == 'ok' else 'partial'},
                summary=f"归属 {row.get('assigned_tracklet_count', 0)} / {row.get('tracklet_count', 0)} 条轨迹",
                media_refs=refs, fields={**row, 'description': '轨迹与语义角色的归属及置信度'},
                provenance={'artifact_ids': ['direct_alignment']}))
        return entities


class CharactersGlobalAdapter(MainlineAdapter):
    adapter_id = 'characters-global-v1'
    adapter_version = '1.1.0'
    module_id = 'character_pack'
    required_artifacts = ('characters',)
    supported_schema_versions = {
        'global-character-materialization-v1',
        'global-character-materialization-v2',
    }

    def entities(self, run, ctx):
        root = run.resolve_binding_path('characters')
        entities = []
        for path in sorted(root.glob('*/*/role_character.json')) if root else []:
            row = read_json(path)
            if row.get('schema_version') not in self.supported_schema_versions:
                continue
            scope, subject = row.get('semantic_scope_id'), row.get('subject_id')
            refs = []
            for category in ('face', 'body', 'rgba'):
                for asset in sorted((path.parent / 'visual' / category).glob('*')):
                    if asset.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                        refs.append(self.media_ref(run, asset, category))
            entities.append(Entity(
                key=EntityKey('character', scope, subject), title=f'{scope} · {subject}',
                status={'execution': 'succeeded'},
                summary=f"{row.get('tracklet_count', 0)} 条轨迹 · {len(refs)} 张视觉素材",
                media_refs=refs, fields=row,
                relations=[{'type': 'voice', 'module_id': 'voice_preview', 'label': '角色声音候选'},
                           {'type': 'alignment', 'module_id': 'person_align', 'label': '人物归属证据'}],
                provenance={'artifact_ids': ['characters']}))
        return entities


class ObjectQueueAdapter(MainlineAdapter):
    adapter_id = 'object-queue-v2'
    module_id = 'object_queue'
    required_artifacts = ('object_candidates',)

    def entities(self, run, ctx):
        entities = []
        for index, row in enumerate(ctx.cached_jsonl(self.required_artifacts[0]) or []):
            shot = shot_canonical_id(row.get('shot_id'))
            scope = row.get('semantic_scope_id') or ctx.shot_to_scope().get(shot)
            refs = [self.media_ref(run, value, key) for key, value in (row.get('assets') or {}).items()
                    if value and key in ('frame_path', 'mask_path', 'crop_path', 'rgba_path', 'overlay_path')]
            entities.append(Entity(
                key=EntityKey('object', scope, f"{row.get('candidate_id', 'object')}_{index}"),
                title=f"{shot} · {row.get('canonical_name', '物体')}",
                status={'execution': 'succeeded', 'quality': 'unreviewed'},
                summary=str(row.get('status') or row.get('verification_status') or 'unverified'),
                media_refs=refs, fields={**row, 'shot_id': shot},
                provenance={'artifact_ids': list(self.required_artifacts)}))
        return entities


class ObjectObservationsAdapter(ObjectQueueAdapter):
    adapter_id = 'object-observations-v3-v4'
    module_id = 'object_ground'
    required_artifacts = ('object_observations',)

    def entities(self, run, ctx):
        entities = super().entities(run, ctx)
        path = run.resolve_binding_path('object_references')
        if path is None:
            return entities
        try:
            reference_sets = read_json(path)
        except (OSError, ValueError):
            return entities
        by_candidate = {r['candidate_id']: r for r in reference_sets.get('candidates', [])}
        seen = set()
        for entity in entities:
            candidate = entity.fields.get('candidate_id')
            if candidate in seen or candidate not in by_candidate:
                continue
            seen.add(candidate)
            record = by_candidate[candidate]
            entity.fields['reference_set'] = record
            entity.provenance['artifact_ids'].append('object_references')
            for ref in record.get('references', []):
                asset = ref.get('asset', {}).get('rgba')
                if asset:
                    entity.media_refs.append(self.media_ref(
                        run, path.parent / asset, f"多帧参考 · rank {ref.get('rank')}"))
        return entities


class SceneIndexAdapter(MainlineAdapter):
    adapter_id = 'scene-index-v3'
    module_id = 'scene_select'
    required_artifacts = ('scene_assets',)

    def entities(self, run, ctx):
        root = run.resolve_binding_path('scene_assets')
        entities = []
        for path in sorted(root.glob('*/scene_index.json')) if root else []:
            index = read_json(path)
            if index.get('schema_version') != 'important-scene-index-v3':
                continue
            for scene in index.get('scenes', []):
                name = scene['scene_name']
                refs = [self.media_ref(run, path.parent / name / frame['asset'],
                                       f"{frame.get('timestamp_sec', 0):.2f}s · rank {frame.get('rank')}")
                        for frame in scene.get('selected', [])]
                entities.append(Entity(
                    key=EntityKey('scene', None, f'{path.parent.name}_{name}'),
                    title=f'{path.parent.name} · {name}', status={'execution': 'succeeded'},
                    summary=f"重要度 {scene.get('importance_score', 0):.3f} · {len(refs)} 张参考图",
                    media_refs=refs, fields=scene, provenance={'artifact_ids': ['scene_assets']}))
        return entities


class GeneratedReviewAdapter(MainlineAdapter):
    """Shared summary/filter behavior for generated comparison artifacts."""

    def summary(self, run, ctx):
        probe = self.probe(run, ctx)
        entities = self.entities(run, ctx) if probe.execution == 'succeeded' else []
        succeeded = sum(1 for entity in entities if entity.fields.get('status') == 'succeeded')
        variants = sorted({str(entity.fields.get('variant')) for entity in entities
                           if entity.fields.get('variant')})
        subjects = sorted({str(entity.fields.get('subject_id')) for entity in entities
                           if entity.fields.get('subject_id')})
        scopes = sorted({entity.key.scope_id for entity in entities if entity.key.scope_id})
        return ModuleSummary(
            status={'execution': probe.execution, 'quality': 'unreviewed', 'reason': probe.reason},
            metrics=[
                Metric(id='records', label='对比任务', value=len(entities), unit='条',
                       meaning='本次运行落盘的生成对比记录数'),
                Metric(id='succeeded', label='生成成功', value=succeeded,
                       denominator=len(entities), unit='条', meaning='产物文件可读的记录数'),
            ],
            filters_available={'scope': scopes, 'subject': subjects, 'variant': variants,
                               'status': sorted({str(e.fields.get('status')) for e in entities
                                                 if e.fields.get('status')})},
            scopes=scopes, item_count=len(entities), warnings=list(ctx.warnings))


class I2ITestAdapter(GeneratedReviewAdapter):
    adapter_id = 'i2i-test-v1'
    adapter_version = '1.0.0'
    module_id = 'i2i_review'
    required_artifacts = ('i2i_results',)

    def entities(self, run, ctx):
        root = run.resolve_binding_path('i2i_results')
        manifest_path = root / 'manifest.json' if root else None
        if manifest_path is None or not manifest_path.is_file():
            return []
        manifest = read_json(manifest_path)
        entities = []
        for index, row in enumerate(manifest.get('tasks') or []):
            task_id = str(row.get('task_id') or f'task_{index:04d}')
            kind = str(row.get('kind') or 'asset')
            model = str(row.get('model') or manifest.get('model') or 'unknown')
            scope_match = re.search(r'global_\d+', task_id)
            scope = scope_match.group(0) if scope_match else None
            if scope is None:
                shot_match = re.search(r'SHOT_\d+', task_id)
                if shot_match:
                    scope = ctx.shot_to_scope().get(shot_canonical_id(shot_match.group(0)))
            media_refs = [self.media_ref(run, source, f'参考图 {i + 1}')
                          for i, source in enumerate(row.get('inputs') or [])]
            if row.get('output'):
                media_refs.append(self.media_ref(run, row['output'], f'生成结果 · {model}'))
            source_status = str(row.get('status') or 'unknown')
            # 断点续跑把已有图片标为 skipped_existing;对只读 viewer 而言产物仍是成功可用。
            status = 'succeeded' if source_status in ('succeeded', 'skipped_existing') else source_status
            fields = {**row, 'source_status': source_status, 'variant': model, 'status': status}
            entities.append(Entity(
                key=EntityKey('i2i', scope, f'{kind}_{task_id}_{index}'),
                title=f'{kind} · {task_id}',
                status={'execution': 'succeeded' if fields['status'] == 'succeeded' else 'failed',
                        'quality': 'unreviewed'},
                summary=str(row.get('description') or model), media_refs=media_refs, fields=fields,
                provenance={'artifact_ids': ['i2i_results'], 'manifest': 'manifest.json'}))
        return entities


class TTSTestAdapter(GeneratedReviewAdapter):
    adapter_id = 'tts-test-v1'
    adapter_version = '1.0.0'
    module_id = 'tts_review'
    required_artifacts = ('tts_results',)

    def entities(self, run, ctx):
        root = run.resolve_binding_path('tts_results')
        entities = []
        for index, meta_path in enumerate(sorted(root.glob('*/*/meta.json')) if root else []):
            meta = read_json(meta_path)
            model_key = meta_path.parent.parent.name
            speaker_key = meta_path.parent.name
            match = re.match(r'^(global_\d+)_(.+)$', speaker_key)
            scope, subject = (match.group(1), match.group(2)) if match else (None, speaker_key)
            media_refs = []
            reference_wav = meta.get('reference_wav')
            if reference_wav:
                media_refs.append(self.media_ref(run, reference_wav, '合并参考音频', 'audio'))
            reference_name = Path(reference_wav).name if reference_wav else None
            for wav in sorted(meta_path.parent.glob('*.wav')):
                if wav.name == reference_name or wav.name.startswith('reference_'):
                    continue
                media_refs.append(self.media_ref(run, wav, f'合成结果 · {model_key}', 'audio'))
            fields = {
                **meta,
                'variant': model_key,
                'scope': scope,
                'subject_id': subject,
                'status': 'succeeded' if len(media_refs) >= 2 else 'partial',
            }
            entities.append(Entity(
                key=EntityKey('tts', scope, f'{model_key}_{speaker_key}_{index}'),
                title=f'{scope or "unknown"} · {subject} · {model_key}',
                status={'execution': fields['status'], 'quality': 'unreviewed'},
                summary=f"{meta.get('model', model_key)} · 参考与合成音频对比",
                media_refs=media_refs, fields=fields,
                provenance={'artifact_ids': ['tts_results'],
                            'meta': meta_path.relative_to(root).as_posix()}))
        return entities
