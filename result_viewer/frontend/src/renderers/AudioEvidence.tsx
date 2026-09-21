// audio-evidence 渲染器:角色声音候选。
// valid=候选(非审核通过)、invalid=非候选;显示归属证据与准入状态;切换播放自动暂停上一条。

import { FieldsTable, JsonDrawer } from '../components/JsonDrawer'
import { MediaPlayer } from '../components/MediaPlayer'
import { ExecutionBadge } from '../components/StatusWidgets'
import type { RendererProps } from './registry'
import { EmptyState } from './registry'
import { TimeBars } from './VideoTimeline'

export function AudioEvidence(props: RendererProps) {
  const { envelope, entry, loading, error, detailed, selectedEntityId, onSelectEntity, onNavigate } = props
  if (error) return <div className="error-note">{error}</div>
  if (loading && !envelope) return <div className="loading-note">加载中…</div>
  if (!envelope) return null
  if (envelope.items.length === 0) return <EmptyState entry={entry} reason={envelope.status.reason} />

  const selected = envelope.items.find((i) => i.entity_id === selectedEntityId) ?? envelope.items[0]
  const quality = String(selected.status.quality ?? 'unreviewed')

  return (
    <div className="audio-evidence">
      <ul className="entity-list" role="listbox" aria-label="声音候选列表">
        {envelope.items.map((item) => {
          const q = String(item.status.quality ?? '')
          return (
            <li
              key={item.entity_id}
              role="option"
              aria-selected={item.entity_id === selected.entity_id}
              className={item.entity_id === selected.entity_id ? 'selected' : ''}
            >
              <button onClick={() => onSelectEntity(item.entity_id)}>
                <span className={`quality-dot ${q}`} aria-hidden="true" />
                <span className="entity-title">{item.title}</span>
                <span className="entity-summary">{item.summary}</span>
                {q === 'candidate' ? <span className="chip chip-candidate">候选</span> : <span className="chip">非候选</span>}
              </button>
            </li>
          )
        })}
      </ul>
      <div className="entity-detail">
        <header>
          <h3>{selected.title}</h3>
          <ExecutionBadge execution={String(selected.status.execution ?? 'unknown')} reason={selected.status.reason as string | undefined} />
          <span className={`chip ${quality === 'candidate' ? 'chip-candidate' : ''}`}>
            {quality === 'candidate' ? '候选资格(非审核通过)' : '非候选'}
          </span>
        </header>
        <blockquote className="asr-quote">“{String(selected.fields.text ?? '')}”</blockquote>
        {selected.media_refs.map((m, i) => (
          <MediaPlayer key={i} ref_={m} />
        ))}
        <TimeBars item={selected} />
        <EvidenceChips item={selected} />
        <div className="relation-row">
          {selected.relations.map((r, i) => (
            <button key={i} className="chip chip-link" onClick={() => onNavigate(r.module_id, { scope: selected.key.scope_id, subject: subjectOf(selected) })}>
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

function subjectOf(item: { key: { local_id: string } }): string | null {
  const spk = item.key.local_id.split(':')[0]
  return spk || null
}

function EvidenceChips({ item }: { item: { fields: Record<string, unknown> } }) {
  const f = item.fields
  const rows: [string, string][] = []
  const conf = f.attribution_confidence
  if (conf != null) rows.push(['归属置信度', String(conf)])
  if (f.attribution_evidence_type) rows.push(['证据类型', String(f.attribution_evidence_type)])
  if (f.attribution_evidence) rows.push(['证据', String(f.attribution_evidence)])
  if (f.overlap_status) rows.push(['重叠', String(f.overlap_status)])
  if (f.is_off_screen === true) rows.push(['画外音', '是'])
  if (f.audio_asset_admission_status) rows.push(['准入状态', String(f.audio_asset_admission_status)])
  if (!rows.length) return null
  return (
    <div className="evidence-chips">
      {rows.map(([k, v]) => (
        <span className="chip" key={k}>
          {k}:{v}
        </span>
      ))}
    </div>
  )
}
