// tracklet-gallery 渲染器:带框视频 + 轨迹列表 + 脸/身体观测画廊(文档二 §10 两层展示)。
// 显示"轨迹数量",不用"人物数量"替代。

import { FieldsTable, JsonDrawer } from '../components/JsonDrawer'
import { MediaPlayer } from '../components/MediaPlayer'
import { ExecutionBadge } from '../components/StatusWidgets'
import type { RendererProps } from './registry'
import { EmptyState } from './registry'
import { TimeBars, fmtSec } from './VideoTimeline'

interface TrackletRow {
  tracklet_id?: string
  start_time?: number
  end_time?: number
  observation_count?: number
  face_count?: number
  body_count?: number
}

interface TrackletGallery {
  tracklet_id?: string
  start_time?: number
  end_time?: number
  cannot_links?: number
  face?: { url?: string; label: string; availability?: string }[]
  body?: { url?: string; label: string; availability?: string }[]
}

export function TrackletGallery(props: RendererProps) {
  const { envelope, entry, loading, error, detailed, selectedEntityId, onSelectEntity, onNavigate } = props
  if (error) return <div className="error-note">{error}</div>
  if (loading && !envelope) return <div className="loading-note">加载中…</div>
  if (!envelope) return null
  if (envelope.items.length === 0) return <EmptyState entry={entry} reason={envelope.status.reason} />

  const selected = envelope.items.find((i) => i.entity_id === selectedEntityId) ?? envelope.items[0]
  const tracklets = (selected.fields.tracklets as TrackletRow[] | undefined) ?? []
  const galleries = (selected.fields.tracklet_galleries as TrackletGallery[] | undefined) ?? []

  return (
    <div className="tracklet-gallery">
      <ul className="entity-list" role="listbox" aria-label="镜头轨迹列表">
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
        {selected.media_refs.map((m, i) => (
          <MediaPlayer key={i} ref_={m} />
        ))}
        <TimeBars item={selected} />

        <section>
          <h4>轨迹({tracklets.length} 条 · 轨迹数不是人物数)</h4>
          <table className="mini-table">
            <thead>
              <tr>
                <th>轨迹 ID</th>
                <th>时间(镜头内)</th>
                <th>观测</th>
                <th>脸/身体证据</th>
              </tr>
            </thead>
            <tbody>
              {tracklets.map((t, i) => (
                <tr key={i}>
                  <td>{String(t.tracklet_id ?? '').split('_T').slice(-1)[0] || t.tracklet_id}</td>
                  <td>
                    {t.start_time != null && t.end_time != null ? `${fmtSec(t.start_time)}–${fmtSec(t.end_time)}` : '—'}
                  </td>
                  <td>{t.observation_count ?? '—'}</td>
                  <td>
                    {t.face_count ?? 0} / {t.body_count ?? 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {galleries.length > 0 ? (
          <section>
            <h4>观测画廊(脸 / 身体裁图)</h4>
            {galleries.map((g, gi) => (
              <div className="tracklet-gallery-row" key={gi}>
                <div className="tracklet-gallery-head">
                  <code>{g.tracklet_id}</code>
                  <span className="dim">
                    {g.start_time != null && g.end_time != null ? `${fmtSec(g.start_time)}–${fmtSec(g.end_time)}` : ''}
                    {g.cannot_links != null && g.cannot_links > 0 ? ` · cannot-link ×${g.cannot_links}` : ''}
                  </span>
                </div>
                <div className="obs-strip">
                  {[...(g.face ?? []), ...(g.body ?? [])].map((img, i) =>
                    img.url ? (
                      <figure key={i}>
                        <img src={img.url} alt={img.label} loading="lazy" />
                        <figcaption>{img.label}</figcaption>
                      </figure>
                    ) : (
                      <div className="media-missing small" key={i}>
                        {img.label} 缺失
                      </div>
                    ),
                  )}
                </div>
              </div>
            ))}
          </section>
        ) : null}

        <div className="relation-row">
          {selected.relations.map((r, i) => (
            <button key={i} className="chip chip-link" onClick={() => onNavigate(r.module_id, { shot: selected.key.local_id })}>
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
