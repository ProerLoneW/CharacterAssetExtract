// 模块结果页(文档三 §3.1):固定四部分 ——
// 1) 业务标题与一句用途;2) 流程位置及输入/输出跳转;3) 该步骤的结果;4) 说明/证据抽屉。

import { useEffect, useMemo, useState } from 'react'
import { api, ApiRequestError } from '../api'
import type { Envelope, ModuleEntry, ModuleMeta, Spec } from '../types'
import { rendererFor } from '../../renderers/registry'
import { ModuleDocs } from '../../components/ModuleDocs'
import { ExecutionBadge, LifecycleBadge, MetricChips } from '../../components/StatusWidgets'

export function ModuleResultPage({
  spec,
  module,
  entry,
  runId,
  scope,
  entityId,
  view,
  onSelectEntity,
  onNavigate,
}: {
  spec: Spec
  module: ModuleMeta
  entry: ModuleEntry | null
  runId: string
  scope: string | null
  entityId: string | null
  view: 'compact' | 'detailed'
  onSelectEntity: (entityId: string | null) => void
  onNavigate: (module: string, extra?: Record<string, string | null>) => void
}) {
  const [envelope, setEnvelope] = useState<Envelope | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterValues, setFilterValues] = useState<Record<string, string | null>>({})

  // 切换 scope 时清掉对旧 scope 无意义的筛选(subject 等)
  useEffect(() => {
    setFilterValues((prev) => ({ ...prev, subject: null, shot: null, segment: null }))
  }, [scope])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    const params: Record<string, string | null> = { scope, limit: '100' }
    for (const [k, v] of Object.entries(filterValues)) {
      if (v) params[k] = v
    }
    api
      .items(runId, module.id, params, controller.signal)
      .then((env) => {
        setEnvelope(env)
        setLoading(false)
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setError(e instanceof ApiRequestError ? `${e.code}:${e.message}` : String(e))
        setLoading(false)
      })
    return () => controller.abort()
  }, [runId, module.id, scope, JSON.stringify(filterValues), entry?.status.execution])

  const upstream = useMemo(() => {
    const dep = new Set([...module.depends_on])
    return spec.modules.filter((m) => dep.has(m.id))
  }, [spec, module])
  const downstream = useMemo(() => spec.modules.filter((m) => m.depends_on.includes(module.id)), [spec, module])

  const Renderer = rendererFor(module.view.renderer)

  return (
    <div className="module-result">
      <header className="module-head">
        <div className="module-head-row">
          <h2>{module.title}</h2>
          <LifecycleBadge lifecycle={module.lifecycle} />
          {entry ? <ExecutionBadge execution={entry.status.execution} reason={entry.status.reason} /> : null}
        </div>
        <p className="lead">{module.summary}</p>
        <div className="flow-position">
          <div className="flow-col">
            <span className="flow-label">上游输入</span>
            {upstream.length ? (
              upstream.map((m) => (
                <button key={m.id} className="chip chip-link" onClick={() => onNavigate(m.id)}>
                  ← {m.title}
                </button>
              ))
            ) : (
              <span className="dim">(起点步骤)</span>
            )}
          </div>
          <div className="flow-col">
            <span className="flow-label">下游消费</span>
            {downstream.length ? (
              downstream.map((m) => (
                <button key={m.id} className="chip chip-link" onClick={() => onNavigate(m.id)}>
                  → {m.title}
                </button>
              ))
            ) : (
              <span className="dim">(末端消费)</span>
            )}
          </div>
        </div>
      </header>

      <section className="module-metrics">
        <MetricChips metrics={envelope?.metrics ?? entry?.metrics ?? []} />
        {envelope?.warnings?.length ? (
          <ul className="warning-list">
            {envelope.warnings.map((w, i) => (
              <li key={i}>⚠ {w}</li>
            ))}
          </ul>
        ) : null}
      </section>

      <section className="module-filters">
        {(module.view.filters.includes('scope') ? false : true) && null}
        {module.view.filters
          .filter((f) => f !== 'scope')
          .map((f) => {
            const options = entry?.filters_available?.[f] ?? []
            if (!options.length) return null
            return (
              <label key={f} className="filter-box">
                <span>{filterLabel(f)}</span>
                <select
                  value={filterValues[f] ?? ''}
                  onChange={(e) => setFilterValues((prev) => ({ ...prev, [f]: e.target.value || null }))}
                >
                  <option value="">全部</option>
                  {options.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </label>
            )
          })}
        {module.view.filters.includes('status') && entry?.filters_available?.status?.length ? null : null}
      </section>

      <Renderer
        module={module}
        entry={entry ?? fallbackEntry(module)}
        envelope={envelope}
        loading={loading}
        error={error}
        detailed={view === 'detailed'}
        selectedEntityId={entityId}
        onSelectEntity={onSelectEntity}
        onNavigate={onNavigate}
      />

      <ModuleDocs module={module} />
    </div>
  )
}

function filterLabel(f: string): string {
  const map: Record<string, string> = {
    subject: '人物/主体',
    shot: '镜头',
    segment: '片段',
    status: '状态',
    variant: '版本',
  }
  return map[f] ?? f
}

function fallbackEntry(module: ModuleMeta): ModuleEntry {
  return {
    module_id: module.id,
    title: module.title,
    order: module.order,
    group: module.group,
    lifecycle: module.lifecycle,
    view_enabled: module.view.enabled,
    renderer: module.view.renderer,
    adapter: module.view.adapter,
    filters: module.view.filters,
    primary_artifact: module.view.primary_artifact,
    status: { execution: 'unknown', quality: 'unreviewed', reason: '模块状态加载中' },
    metrics: [],
    filters_available: {},
    scopes: [],
    warnings: [],
    bound_artifacts: [],
  }
}
