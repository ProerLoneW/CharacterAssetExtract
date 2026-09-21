// 媒体播放组件(文档三 §3.3):每次只自动播放用户选择的一个视频/音频;懒加载。
// 缺失媒体显示可解释原因,不阻断其他字段浏览。

import { useEffect, useRef } from 'react'
import type { MediaRef } from '../app/types'

const activeElements = new Set<HTMLMediaElement>()

function registerSinglePlayback(el: HTMLMediaElement) {
  el.addEventListener('play', () => {
    for (const other of activeElements) {
      if (other !== el && !other.paused) other.pause()
    }
  })
  activeElements.add(el)
}

export function MediaPlayer({ ref_: media }: { ref_: MediaRef }) {
  const elRef = useRef<HTMLMediaElement | null>(null)
  useEffect(() => {
    const el = elRef.current
    if (el) registerSinglePlayback(el)
    return () => {
      if (el) activeElements.delete(el)
    }
  }, [])

  if (media.availability !== 'available' || !media.url) {
    return (
      <div className="media-missing" role="note">
        <span className="media-missing-icon" aria-hidden="true">
          {media.kind === 'audio' ? '🔇' : media.kind === 'image' ? '🖼' : '🎬'}
        </span>
        <span>
          {media.label} · {availabilityLabel(media.availability)}
          {media.detail ? ` · ${media.detail}` : ''}
        </span>
      </div>
    )
  }

  if (media.kind === 'image') {
    return (
      <figure className="media-figure">
        <img src={media.url} alt={media.label} loading="lazy" />
        <figcaption>{media.label}</figcaption>
      </figure>
    )
  }

  const duration = media.duration_sec != null ? ` · ${media.duration_sec.toFixed(1)} 秒` : ''
  return (
    <div className="media-player">
      {media.kind === 'video' ? (
        <video ref={elRef as React.RefObject<HTMLVideoElement>} src={media.url} controls preload="metadata" />
      ) : (
        <audio ref={elRef as React.RefObject<HTMLAudioElement>} src={media.url} controls preload="metadata" />
      )}
      <span className="media-caption">
        {media.label}
        {duration}
      </span>
    </div>
  )
}

function availabilityLabel(a: string): string {
  switch (a) {
    case 'missing':
      return '源文件未收录到本运行'
    case 'forbidden':
      return '路径越出受控来源根'
    case 'unconfigured':
      return '未配置'
    default:
      return '不可用'
  }
}
