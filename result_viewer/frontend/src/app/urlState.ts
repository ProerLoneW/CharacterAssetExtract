// URL query 路由状态(文档三 §3.5):
// ?tab=intro|results&run=...&module=...&scope=...&entity=...&view=compact|detailed
// 导航用 pushState;筛选变化用 replaceState 避免每次按键产生历史记录。

import { useCallback, useEffect, useMemo, useState } from 'react'

export interface UrlState {
  tab: 'intro' | 'results'
  run: string | null
  module: string | null
  scope: string | null
  entity: string | null
  view: 'compact' | 'detailed'
}

function parse(): UrlState {
  const q = new URLSearchParams(window.location.search)
  const tab = q.get('tab') === 'results' ? 'results' : 'intro'
  const view = q.get('view') === 'detailed' ? 'detailed' : 'compact'
  return {
    tab,
    run: q.get('run'),
    module: q.get('module'),
    scope: q.get('scope'),
    entity: q.get('entity'),
    view,
  }
}

function build(next: Partial<UrlState>, current: UrlState): string {
  const merged: UrlState = { ...current, ...next }
  const q = new URLSearchParams()
  q.set('tab', merged.tab)
  if (merged.run) q.set('run', merged.run)
  if (merged.module) q.set('module', merged.module)
  if (merged.scope) q.set('scope', merged.scope)
  if (merged.entity) q.set('entity', merged.entity)
  if (merged.view !== 'compact') q.set('view', merged.view)
  const qs = q.toString()
  return qs ? `?${qs}` : window.location.pathname
}

export function useUrlState(): [UrlState, (next: Partial<UrlState>, mode?: 'push' | 'replace') => void] {
  const [state, setState] = useState<UrlState>(parse)

  useEffect(() => {
    const onPop = () => setState(parse())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const update = useCallback(
    (next: Partial<UrlState>, mode: 'push' | 'replace' = 'push') => {
      setState((prev) => {
        const url = build(next, prev)
        if (mode === 'push') window.history.pushState(null, '', url)
        else window.history.replaceState(null, '', url)
        return { ...prev, ...next }
      })
    },
    [],
  )

  return [state, update]
}

export function useQueryMemo<T>(fn: () => T, deps: unknown[]): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(fn, deps)
}

export function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return debounced
}
