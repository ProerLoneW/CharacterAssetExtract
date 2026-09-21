// character-gallery 渲染器:旧链角色目录(也服务新链角色页)。
// 角色卡明确 scope、视觉身份、投票证据、脸/身体/声音数量;声音缺席显示原因,不伪造审核状态。

import { FieldsTable, JsonDrawer } from '../components/JsonDrawer'
import { ExecutionBadge } from '../components/StatusWidgets'
import type { RendererProps } from './registry'
import { EmptyState } from './registry'

export function CharacterGallery(props: RendererProps) {
  const { envelope, entry, loading, error, detailed, selectedEntityId, onSelectEntity, onNavigate } = props
  if (error) return <div className="error-note">{error}</div>
  if (loading && !envelope) return <div className="loading-note">加载中…</div>
  if (!envelope) return null
  if (envelope.items.length === 0) return <EmptyState entry={entry} reason={envelope.status.reason} />

  const selected = envelope.items.find((i) => i.entity_id === selectedEntityId) ?? envelope.items[0]

  return (
    <div className="character-gallery">
      <div className="character-grid">
        {envelope.items.map((item) => (
          <button
            key={item.entity_id}
            className={`character-card ${item.entity_id === selected.entity_id ? 'selected' : ''}`}
            onClick={() => onSelectEntity(item.entity_id)}
          >
            <div className="character-head">
              <span className="character-name">{item.title}</span>
              <span className="chip">{String(item.fields.visual_character_id ?? '')}</span>
            </div>
            <p className="dim">{item.summary}</p>
            <div className="thumb-strip">
              {item.media_refs
                .filter((m) => m.kind === 'image' && m.availability === 'available' && m.url)
                .slice(0, 4)
                .map((m, i) => (
                  <img key={i} src={m.url!} alt={m.label} loading="lazy" />
                ))}
            </div>
          </button>
        ))}
      </div>
      <div className="entity-detail">
        <header>
          <h3>{selected.title}</h3>
          <ExecutionBadge execution={String(selected.status.execution ?? 'unknown')} reason={selected.status.reason as string | undefined} />
          {selected.status.voice === 'skipped' ? (
            <span className="chip" title={String(selected.fields.voice_skipped_reason ?? '')}>
              声音未交付
            </span>
          ) : null}
        </header>
        <p className="dim">{selected.summary}</p>
        <section className="gallery-grid">
          {selected.media_refs.map((m, i) =>
            m.kind === 'image' ? (
              <figure className="gallery-item" key={i}>
                {m.availability === 'available' && m.url ? (
                  <img src={m.url} alt={m.label} loading="lazy" />
                ) : (
                  <div className="media-missing">{m.label} · 缺失</div>
                )}
                <figcaption>{m.label}</figcaption>
              </figure>
            ) : null,
          )}
        </section>
        <div className="relation-row">
          {selected.relations.map((r, i) => (
            <button
              key={i}
              className="chip chip-link"
              onClick={() =>
                onNavigate(r.module_id, {
                  scope: selected.key.scope_id,
                  subject: selected.key.local_id,
                })
              }
            >
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
