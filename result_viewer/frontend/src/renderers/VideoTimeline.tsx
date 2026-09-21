// video-timeline 渲染器:镜头清单 / 全局片段 / 导出片段。
// 左列表 + 右选中详情;时间同时标注 source 与 global-local 两种时基(文档三 §3.3)。

import { FieldsTable, JsonDrawer } from '../components/JsonDrawer'
import { MediaPlayer } from '../components/MediaPlayer'
import { ExecutionBadge } from '../components/StatusWidgets'
import type { RendererProps } from './registry'
import { EmptyState } from './registry'

export function VideoTimeline(props: RendererProps) {
  const { envelope, entry, loading, error, detailed, selectedEntityId, onSelectEntity, onNavigate } = props
  if (error) return <div className="error-note">{error}</div>
  if (loading && !envelope) return <div className="loading-note">加载中…</div>
  if (!envelope) return null
  if (envelope.items.length === 0) return <EmptyState entry={entry} reason={envelope.status.reason} />

  const selected = envelope.items.find((i) => i.entity_id === selectedEntityId) ?? envelope.items[0]

  return (
    <div className="video-timeline">
      <ul className="entity-list" role="listbox" aria-label="结果列表">
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
          <ExecutionBadge execution={String(selected.status.execution ?? 'unknown')} reason={selected.status.reason as string | undefined} />
        </header>
        {selected.media_refs.map((m, i) => (
          <MediaPlayer key={i} ref_={m} />
        ))}
        <TimeBars item={selected} />
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

export function TimeBars({ item }: { item: { time_refs: { timebase: string; start_sec: number | null; end_sec: number | null }[] } }) {
  if (!item.time_refs?.length) return null
  return (
    <div className="time-bars">
      {item.time_refs.map((t, i) => (
        <div className="time-bar" key={i}>
          <span className="timebase-label">{timebaseLabel(t.timebase)}</span>
          <span className="time-range">
            {t.start_sec != null ? fmtSec(t.start_sec) : '?'} – {t.end_sec != null ? fmtSec(t.end_sec) : '?'}
          </span>
        </div>
      ))}
    </div>
  )
}

function timebaseLabel(tb: string): string {
  switch (tb) {
    case 'source':
      return '原片时间'
    case 'global_local':
      return '全局片段内时间'
    case 'shot_local':
      return '镜头内时间'
    case 'segment_local':
      return '片段内时间'
    default:
      return tb
  }
}

export function fmtSec(v: number): string {
  const m = Math.floor(v / 60)
  const s = v - m * 60
  return `${m}:${s.toFixed(2).padStart(5, '0')}`
}
