// 应用外壳(文档三 §3.1 布局):
// 顶栏 = 标题 + 运行选择 + 全局片段选择 + 简明/详细切换;
// 左侧导航 = 系统介绍 + 流程结果(按分组);窄屏导航折叠为抽屉。

import { useEffect, useMemo, useState } from 'react'
import { ApiRequestError, api } from './api'
import { ModuleResultPage } from './pages/ModuleResultPage'
import { SystemIntroPage } from './pages/SystemIntroPage'
import type { Envelope, Item, ModuleEntry, RunSummary, Spec } from './types'
import { useRuns, useRunModules, useSpecStreaming } from './useSpec'
import { useUrlState } from './urlState'

export function App() {
  const [url, navigate] = useUrlState()
  const [routeId, setRouteId] = useState<string | null>(null)
  const [showAdvancedNav, setShowAdvancedNav] = useState(false)
  const [navOpenMobile, setNavOpenMobile] = useState(false)

  const { spec, error: specError, docErrors, reload: reloadSpec } = useSpecStreaming(() => {
    // 文档热更新后重取本页数据(spec 变化由依赖触发,见下方 useEffect)
  })
  const runs = useRuns()

  // 默认选第一个运行
  useEffect(() => {
    if (!url.run && runs && runs.length > 0) {
      navigate({ run: runs[0].run_id }, 'replace')
    }
  }, [runs, url.run, navigate])

  // 文档版本变化时静默刷新运行模块状态(useRunModules 依赖 docRevision)
  const docRevision = spec?.doc_revision ?? null
  const entries = useRunModules(url.run, docRevision)

  // 初始化路线
  useEffect(() => {
    if (spec && routeId === null) setRouteId(url.module ? routeOfModule(spec, url.module) ?? spec.system.default_route : spec.system.default_route)
  }, [spec, routeId, url.module])

  const currentRun = runs?.find((r) => r.run_id === url.run) ?? null

  // scope 有效性:不在当前运行 scopes 时清除(切换运行后清理旧选择)
  useEffect(() => {
    if (currentRun && url.scope && !currentRun.scopes.includes(url.scope)) {
      navigate({ scope: null, entity: null }, 'replace')
    }
  }, [currentRun, url.scope, navigate])

  const activeModule = useMemo(() => spec?.modules.find((m) => m.id === url.module) ?? null, [spec, url.module])
  const activeEntry = useMemo(() => entries?.find((e) => e.module_id === url.module) ?? null, [entries, url.module])

  // 左侧导航:分组折叠状态(默认全部展开)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      return next
    })
  }

  const onNavigate = (moduleId: string, extra?: Record<string, string | null>) => {
    navigate({ tab: 'results', module: moduleId, entity: null, ...extra })
    setNavOpenMobile(false)
  }

  // 文档错误外壳(文档三 §5.4)
  if (!spec && specError) return <ErrorShell message={`文档加载失败:${specError}`} onRetry={reloadSpec} />
  if (!spec) return <div className="boot">正在加载系统说明…</div>
  if (!spec.system) return <ErrorShell message="文档不可用:没有通过校验的快照" docErrors={docErrors} onRetry={reloadSpec} />

  const navModules = buildNavModules(spec, entries, routeId, showAdvancedNav)
  const routeTitle = spec.routes.find((r) => r.id === routeId)?.title ?? ''

  return (
    <div className={`app-shell ${navOpenMobile ? 'nav-open' : ''}`}>
      <header className="top-bar">
        <button className="nav-toggle" aria-label="切换导航" onClick={() => setNavOpenMobile(!navOpenMobile)}>
          ☰
        </button>
        <span className="app-title">{spec.system.title}</span>
        <div className="top-controls">
          <label className="top-select">
            <span>运行</span>
            <select value={url.run ?? ''} onChange={(e) => navigate({ run: e.target.value || null, scope: null, entity: null })}>
              {(runs ?? []).map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.title}
                </option>
              ))}
            </select>
          </label>
          <label className="top-select">
            <span>全局片段</span>
            <select value={url.scope ?? ''} onChange={(e) => navigate({ scope: e.target.value || null, entity: null })}>
              <option value="">全部</option>
              {(currentRun?.scopes ?? []).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <div className="view-toggle" role="radiogroup" aria-label="视图密度">
            <button role="radio" aria-checked={url.view === 'compact'} className={url.view === 'compact' ? 'active' : ''} onClick={() => navigate({ view: 'compact' }, 'replace')}>
              简明
            </button>
            <button role="radio" aria-checked={url.view === 'detailed'} className={url.view === 'detailed' ? 'active' : ''} onClick={() => navigate({ view: 'detailed' }, 'replace')}>
              详细
            </button>
          </div>
        </div>
        {docErrors.length > 0 ? (
          <span className="doc-error-pill" title={docErrors.map((e) => `${e.code}: ${e.message}`).join('\n')}>
            新说明未生效,继续使用版本 {spec.doc_revision.slice(0, 8)}
          </span>
        ) : null}
      </header>

      <aside className="side-nav">
        <nav>
          <button
            className={`nav-item nav-top ${url.tab === 'intro' ? 'active' : ''}`}
            onClick={() => {
              navigate({ tab: 'intro' })
              setNavOpenMobile(false)
            }}
          >
            <span className="nav-dot dot-accent" aria-hidden="true" />
            <span className="nav-label">系统介绍</span>
            <span className="nav-go" aria-hidden="true">›</span>
          </button>
          <div className="nav-head">
            <span>流程结果</span>
            <span className="nav-route">{routeTitle}</span>
          </div>
          {navModules.map((g) => {
            const collapsed = collapsedGroups.has(g.id)
            return (
              <div key={g.id} className="nav-group">
                <button
                  className="nav-group-title"
                  aria-expanded={!collapsed}
                  onClick={() => toggleGroup(g.id)}
                  title={collapsed ? '展开分组' : '折叠分组'}
                >
                  <span className="nav-chev" aria-hidden="true">{collapsed ? '▸' : '▾'}</span>
                  <span className="nav-group-name">{g.title}</span>
                  <span className="nav-count">{g.modules.length}</span>
                </button>
                {!collapsed &&
                  g.modules.map((m) => (
                    <button
                      key={m.id}
                      className={`nav-item ${url.module === m.id && url.tab === 'results' ? 'active' : ''}`}
                      onClick={() => onNavigate(m.id)}
                      title={`${m.summary}\n点击查看结果页`}
                    >
                      <span className={`nav-dot ${dotClass(m.statusEntry?.status.execution)}`} aria-hidden="true" />
                      <span className="nav-label">{m.title}</span>
                      <span className="nav-go" aria-hidden="true">›</span>
                    </button>
                  ))}
              </div>
            )
          })}
          <label className="toggle-label nav-advanced">
            <input type="checkbox" checked={showAdvancedNav} onChange={(e) => setShowAdvancedNav(e.target.checked)} />
            显示兼容/实验模块
          </label>
          <div className="nav-legend">
            <div><span className="nav-dot dot-ok" aria-hidden="true" /> 有结果 <span className="nav-dot dot-dim" aria-hidden="true" /> 未收录</div>
            <div className="dim">点击模块 → 打开其结果页</div>
          </div>
        </nav>
      </aside>

      <main className="main-area">
        {url.tab === 'intro' || !url.module ? (
          <SystemIntroPage
            spec={spec}
            entries={entries}
            runTitle={currentRun?.title ?? null}
            currentRoute={routeId ?? spec.system.default_route}
            onRouteChange={(rid) => setRouteId(rid)}
            onOpenModule={(mid) => onNavigate(mid)}
          />
        ) : activeModule && url.run ? (
          <ModuleResultPage
            key={`${url.module}:${url.run}:${docRevision ?? ''}`}
            spec={spec}
            module={activeModule}
            entry={activeEntry}
            runId={url.run}
            scope={url.scope}
            entityId={url.entity}
            view={url.view}
            onSelectEntity={(eid) => navigate({ entity: eid }, 'replace')}
            onNavigate={onNavigate}
          />
        ) : (
          <div className="empty-state">
            <div className="empty-title">未选择运行</div>
            <p className="dim">请先在顶栏选择一个运行。</p>
          </div>
        )}
      </main>
    </div>
  )
}

interface NavGroupView {
  id: string
  title: string
  modules: { id: string; title: string; summary: string; statusEntry?: ModuleEntry }[]
}

function buildNavModules(spec: Spec, entries: ModuleEntry[] | null, routeId: string | null, showAdvanced: boolean): NavGroupView[] {
  const route = spec.routes.find((r) => r.id === routeId)
  const inRoute = new Set(route?.modules ?? [])
  const statusOf = (id: string) => entries?.find((e) => e.module_id === id)
  const groups = [...spec.groups].sort((a, b) => a.order - b.order)
  const out: NavGroupView[] = []
  for (const g of groups) {
    const mods = spec.modules
      .filter((m) => m.group === g.id)
      .filter((m) => (showAdvanced ? m.view.enabled : m.view.enabled && inRoute.has(m.id)))
      .sort((a, b) => a.order - b.order)
      .map((m) => ({ id: m.id, title: m.title, summary: m.summary, statusEntry: statusOf(m.id) }))
    if (mods.length) out.push({ id: g.id, title: g.title, modules: mods })
  }
  return out
}

function dotClass(execution?: string): string {
  if (execution === 'succeeded') return 'dot-ok'
  if (execution === 'partial') return 'dot-warn'
  if (execution === 'failed') return 'dot-err'
  return 'dot-dim'
}

function routeOfModule(spec: Spec, moduleId: string): string | null {
  for (const r of spec.routes) {
    if (r.modules.includes(moduleId)) return r.id
  }
  return null
}

function ErrorShell({ message, docErrors, onRetry }: { message: string; docErrors?: { code: string; message: string; line: number | null; field: string | null }[]; onRetry: () => void }) {
  return (
    <div className="error-shell">
      <h1>结果展示服务</h1>
      <p className="lead">{message}</p>
      {docErrors && docErrors.length ? (
        <div className="doc-error-list">
          <h3>文档校验错误(最近一次)</h3>
          <ul>
            {docErrors.map((e, i) => (
              <li key={i}>
                <code>{e.code}</code> 第 {e.line ?? '?'} 行{e.field ? ` 字段 ${e.field}` : ''}:{e.message}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <button onClick={onRetry}>重试</button>
    </div>
  )
}
