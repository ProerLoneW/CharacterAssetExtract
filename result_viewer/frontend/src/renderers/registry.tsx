// Renderer 注册表(文档三 §4):renderer 名 -> 受控组件。
// 未实现专属组件的类型回退到通用列表/未收录视图;新增模块复用已有 renderer 无需改前端。

import type { ComponentType } from 'react'
import type { Envelope, Item, ModuleEntry, ModuleMeta } from '../app/types'
import { VideoTimeline } from './VideoTimeline'
import { CaptionDetail } from './CaptionDetail'
import { AudioEvidence } from './AudioEvidence'
import { CharacterGallery } from './CharacterGallery'
import { TrackletGallery } from './TrackletGallery'
import { MediaPlayer } from '../components/MediaPlayer'
import { JsonDrawer } from '../components/JsonDrawer'

export interface RendererProps {
  module: ModuleMeta
  entry: ModuleEntry
  envelope: Envelope | null
  loading: boolean
  error: string | null
  detailed: boolean
  selectedEntityId: string | null
  onSelectEntity: (entityId: string | null) => void
  onNavigate: (module: string, extra?: Record<string, string | null>) => void
}

const RENDERERS: Record<string, ComponentType<RendererProps>> = {
  'video-timeline': VideoTimeline,
  'caption-detail': CaptionDetail,
  'audio-evidence': AudioEvidence,
  'character-gallery': CharacterGallery,
  'tracklet-gallery': TrackletGallery,
  // 以下类型已有注册能力,数据绑定后使用通用展示
  'source-overview': VideoTimeline,
  transcript: AudioEvidence,
  'alignment-evidence': CaptionDetail,
  'identity-graph': GenericList,
  'record-table': GenericList,
  'object-evidence': GenericList,
  'scene-gallery': GenericList,
  'generation-bundle': GenericList,
  'image-comparison': GenericList,
  // TTS 对比是“参考 + 合成结果”,不套用声音候选的准入标签。
  'audio-comparison': GenericList,
  'artifact-browser': GenericList,
}

export function rendererFor(name: string): ComponentType<RendererProps> {
  return RENDERERS[name] ?? GenericList
}

export { GenericList }

function GenericList(props: RendererProps) {
  const { envelope, entry, loading, error } = props
  return (
    <div className="renderer generic-list">
      {error ? <div className="error-note">{error}</div> : null}
      {loading ? <div className="loading-note">加载中…</div> : null}
      {!loading && envelope && envelope.items.length === 0 ? (
        <EmptyState entry={entry} reason={envelope.status.reason} />
      ) : null}
      {envelope?.items.map((item) => (
        <article key={item.entity_id} className="item-card">
          <header>
            <h4>{item.title}</h4>
          </header>
          <p className="dim">{item.summary}</p>
          <div className="gallery-grid">
            {item.media_refs.map((media, index) => (
              <figure className="gallery-item" key={index}>
                <MediaPlayer ref_={media} />
                <figcaption>{media.label}</figcaption>
              </figure>
            ))}
          </div>
          <JsonDrawer data={item.fields} title="结果详情" />
        </article>
      ))}
    </div>
  )
}

export function EmptyState({ entry, reason }: { entry: ModuleEntry; reason?: string | null }) {
  return (
    <div className="empty-state">
      <div className="empty-title">该运行未收录此步骤的结果</div>
      <p>
        {reason ?? entry.status.reason ?? '未配置产物绑定。'}
      </p>
      <p className="dim">说明仍可展开;运行该步骤或为对应产物建立绑定后,此处会显示真实数据。</p>
    </div>
  )
}
