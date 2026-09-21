// caption-detail 渲染器:全局/镜头/片段三级结构化描述。
// 默认展示一句话内容、人物卡、事件条;对白与证据按需展开;原始 JSON 在开发详情(文档三 §2 模块正文要求)。

import { useState } from 'react'
import { FieldsTable, JsonDrawer } from '../components/JsonDrawer'
import { ExecutionBadge } from '../components/StatusWidgets'
import type { Item } from '../app/types'
import type { RendererProps } from './registry'
import { EmptyState } from './registry'
import { TimeBars } from './VideoTimeline'

interface SubjectCard {
  subject_id?: string
  identity_description?: string
  visual_cues?: string[]
  voice_cues?: string[]
  importance_tier?: string
  role_label?: string
  first_seen_sec?: number
  last_seen_sec?: number
  confidence?: string
}

interface TimelineEvent {
  time_sec?: number[]
  subject_ids?: string[]
  event?: string
  description?: string
}

export function CaptionDetail(props: RendererProps) {
  const { envelope, entry, loading, error, detailed, selectedEntityId, onSelectEntity, onNavigate } = props
  if (error) return <div className="error-note">{error}</div>
  if (loading && !envelope) return <div className="loading-note">加载中…</div>
  if (!envelope) return null
  if (envelope.items.length === 0) return <EmptyState entry={entry} reason={envelope.status.reason} />

  const selected = envelope.items.find((i) => i.entity_id === selectedEntityId) ?? envelope.items[0]
  const f = selected.fields ?? {}
  const subjects = (f.subjects as SubjectCard[] | undefined) ?? []
  const timeline = (f.event_timeline as TimelineEvent[] | undefined) ?? []
  const dialogue = (f.dialogue as unknown[] | undefined) ?? []
  const asr = (f.asr_speaker_attributions as Record<string, unknown>[] | undefined) ?? []
  const objects = (f.key_object_candidates as Record<string, unknown>[] | undefined) ?? []

  return (
    <div className="caption-detail">
      <ul className="entity-list" role="listbox" aria-label="描述记录">
        {envelope.items.map((item) => (
          <li
            key={item.entity_id}
            role="option"
            aria-selected={item.entity_id === selected.entity_id}
            className={item.entity_id === selected.entity_id ? 'selected' : ''}
          >
            <button onClick={() => onSelectEntity(item.entity_id)}>
              <span className="entity-title">{item.title}</span>
              <span className="entity-summary">{item.summary}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="entity-detail">
        <header>
          <h3>{selected.title}</h3>
          <ExecutionBadge execution={String(selected.status.execution ?? 'unknown')} />
        </header>
        <p className="lead">{String(f.global_description ?? f.shot_description ?? f.description ?? f.segment_description ?? '')}</p>
        {f.global_scene_description ?? f.scene_description ?? f.shot_scene_description ? (
          <p className="dim">{String(f.global_scene_description ?? f.scene_description ?? f.shot_scene_description)}</p>
        ) : null}
        <TimeBars item={selected} />

        {subjects.length > 0 ? (
          <section>
            <h4>人物注册表({subjects.length})</h4>
            <div className="subject-cards">
              {subjects.map((s, i) => (
                <div className="subject-card" key={i}>
                  <div className="subject-id">
                    {String(s.subject_id ?? '?')}
                    {s.importance_tier ? <span className="chip">{tierLabel(String(s.importance_tier))}</span> : null}
                  </div>
                  <p>{s.identity_description}</p>
                  {s.visual_cues?.length ? <p className="dim">视觉:{s.visual_cues.join('、')}</p> : null}
                  {s.voice_cues?.length ? <p className="dim">声音:{s.voice_cues.join('、')}</p> : null}
                  {s.first_seen_sec != null && s.last_seen_sec != null ? (
                    <p className="dim">
                      出现 {fmt(s.first_seen_sec)}–{fmt(s.last_seen_sec)} 秒
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {timeline.length > 0 ? (
          <section>
            <h4>事件时间轴({timeline.length})</h4>
            <ul className="event-timeline">
              {timeline.slice(0, detailed ? 100 : 30).map((e, i) => (
                <li key={i}>
                  <span className="event-time">{e.time_sec ? e.time_sec.map((x) => fmt(x)).join('–') : '?'}</span>
                  <span className="event-subjects">{(e.subject_ids ?? []).join(', ') || '—'}</span>
                  <span className="event-text">{String(e.event ?? e.description ?? '')}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {objects.length > 0 ? (
          <section>
            <h4>关键物体候选({objects.length}) · 仅提案,未定位</h4>
            <div className="chip-row">
              {objects.map((o, i) => (
                <span className="chip" key={i} title={JSON.stringify(o)}>
                  {String(o.canonical_name ?? o.local_object_id ?? i)}
                </span>
              ))}
            </div>
          </section>
        ) : null}

        {asr.length > 0 ? <AsrPanel asr={asr} detailed={detailed} onNavigate={onNavigate} scope={selected.key.scope_id} /> : null}
        {dialogue.length > 0 ? (
          <section>
            <h4>段内对白({dialogue.length})</h4>
            <ul className="dialogue-list">
              {dialogue.map((d, i) => (
                <li key={i}>{dialogueLine(d)}</li>
              ))}
            </ul>
          </section>
        ) : null}

        <div className="relation-row">
          {selected.relations.map((r, i) => (
            <button key={i} className="chip chip-link" onClick={() => onNavigate(r.module_id)}>
              ↗ {r.label}
            </button>
          ))}
        </div>
        <FieldsTable item={selected} detailed={detailed} />
        {detailed ? <JsonDrawer data={selected} /> : null}
      </div>
    </div>
  )
}

function AsrPanel({
  asr,
  detailed,
  onNavigate,
  scope,
}: {
  asr: Record<string, unknown>[]
  detailed: boolean
  onNavigate: RendererProps['onNavigate']
  scope: string | null
}) {
  const [open, setOpen] = useState(false)
  return (
    <section>
      <button className="drawer-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? '▾' : '▸'} 说话归属对白({asr.length} 条 · 含候选标记)
      </button>
      {open ? (
        <ul className="asr-list">
          {asr.slice(0, detailed ? 400 : 100).map((a, i) => {
            const candidate = a.audio_asset_candidate === true
            return (
              <li key={i} className={candidate ? 'asr-candidate' : ''}>
                <span className="asr-time">{fmt(Number(a.start_sec ?? a.time_range_sec_start ?? 0))}</span>
                <span className="asr-speaker">{String(a.speaker_id ?? a.subject_id ?? '?')}</span>
                <span className="asr-text">{String(a.text ?? '')}</span>
                <span className={`chip ${candidate ? 'chip-candidate' : ''}`} title={candidate ? '音频候选资格(非审核通过)' : '非候选'}>
                  {candidate ? '候选' : '非候选'}
                </span>
              </li>
            )
          })}
        </ul>
      ) : null}
      <p className="dim">候选音频可在「角色声音候选」步骤播放。</p>
      <button className="chip chip-link" onClick={() => onNavigate('voice_preview', { scope })}>
        ↗ 查看该 scope 的声音候选
      </button>
    </section>
  )
}

function tierLabel(t: string): string {
  const map: Record<string, string> = { primary: '主要', supporting: '配角', background: '背景' }
  return map[t] ?? t
}

function fmt(v: number): string {
  return Number.isFinite(v) ? v.toFixed(1) : '?'
}

function dialogueLine(d: unknown): string {
  if (typeof d === 'string') return d
  if (d && typeof d === 'object') {
    const o = d as Record<string, unknown>
    const spk = o.speaker_id ?? o.speaker ?? o.subject_id ?? ''
    const text = o.text ?? o.line ?? o.content ?? ''
    return `${spk ? String(spk) + ': ' : ''}${String(text)}`
  }
  return String(d)
}
