// spec 加载 + 版本轮询(文档三 §5.2):
// 浏览器约每 2 秒轮询 /api/v1/version(后台标签页降频到 6 秒);
// doc_revision 变化时重新取 spec,尽量保留筛选、滚动与播放状态。

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiRequestError } from './api'
import type { ModuleEntry, RunSummary, Spec, VersionInfo } from './types'

export interface SpecState {
  spec: Spec | null
  error: string | null
  docErrors: VersionInfo['last_document_errors']
  reloading: boolean
}

export function useSpecStreaming(onSpecChange?: (spec: Spec) => void) {
  const [state, setState] = useState<SpecState>({ spec: null, error: null, docErrors: [], reloading: false })
  const revisionRef = useRef<string | null>(null)
  const changeCb = useRef(onSpecChange)
  changeCb.current = onSpecChange

  const loadSpec = useCallback(async () => {
    setState((s) => ({ ...s, reloading: true }))
    try {
      const spec = await api.spec()
      const changed = revisionRef.current !== null && revisionRef.current !== spec.doc_revision
      revisionRef.current = spec.doc_revision
      setState({ spec, error: null, docErrors: [], reloading: false })
      if (changed) changeCb.current?.(spec)
    } catch (e) {
      const message = e instanceof ApiRequestError ? `${e.code}: ${e.message}` : String(e)
      setState((s) => ({ ...s, error: message, reloading: false }))
    }
  }, [])

  useEffect(() => {
    void loadSpec()
  }, [loadSpec])

  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    const tick = async () => {
      if (stopped) return
      try {
        const v = await api.version()
        // 有编译失败的新版本 -> 显示"新说明未生效"(旧快照仍可用)
        if (v.last_document_errors.length > 0) {
          setState((s) => ({ ...s, docErrors: v.last_document_errors }))
          schedule(document.hidden ? 6000 : 2000)
          return
        }
        // 错误消除(文档恢复,即使 revision 回到旧值)也要清掉提示
        setState((s) => (s.docErrors.length ? { ...s, docErrors: [] } : s))
        if (revisionRef.current !== v.doc_revision) {
          await loadSpec()
          return schedule(document.hidden ? 6000 : 2000)
        }
      } catch {
        /* 网络抖动:下轮再试 */
      }
      schedule(document.hidden ? 6000 : 2000)
    }
    const schedule = (ms: number) => {
      timer = window.setTimeout(tick, ms)
    }
    schedule(2000)
    const onVisibility = () => {
      if (!document.hidden && timer) {
        window.clearTimeout(timer)
        schedule(2000)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      stopped = true
      if (timer) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [loadSpec])

  return { ...state, reload: loadSpec }
}

export function useRuns() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    let loadedOnce = false
    const tick = async () => {
      if (stopped) return
      try {
        const response = await api.runs()
        if (!stopped) {
          loadedOnce = true
          setRuns(response.runs)
        }
      } catch {
        // supervisor 重启后端的短暂窗口内保留旧列表,避免下拉框闪空。
        if (!stopped && !loadedOnce) setRuns([])
      }
      if (!stopped) timer = window.setTimeout(tick, document.hidden ? 8000 : 3000)
    }
    void tick()
    const onVisibility = () => {
      if (!document.hidden && timer) {
        window.clearTimeout(timer)
        timer = window.setTimeout(tick, 0)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      stopped = true
      if (timer) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])
  return runs
}

export function useRunModules(runId: string | null, docRevision: string | null) {
  const [entries, setEntries] = useState<ModuleEntry[] | null>(null)
  useEffect(() => {
    let cancelled = false
    setEntries(null)
    if (!runId) return
    api
      .runModules(runId)
      .then((r) => {
        if (!cancelled) setEntries(r.modules)
      })
      .catch(() => {
        if (!cancelled) setEntries([])
      })
    return () => {
      cancelled = true
    }
  }, [runId, docRevision])
  return entries
}
