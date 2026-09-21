// 系统介绍页(文档三 §3.2):系统标题/摘要、overview 业务解释、
// 可交互流程图(分层 DAG)+ 右侧模块详情面板。

import { useEffect, useState } from 'react'
import { Blocks } from '../../components/Blocks'
import { ModuleDetailPanel } from '../../components/ModuleDetailPanel'
import { ModuleFlowDiagram } from '../../components/ModuleFlowDiagram'
import type { ModuleEntry, Spec } from '../types'

export function SystemIntroPage({
  spec,
  entries,
  runTitle,
  currentRoute,
  onRouteChange,
  onOpenModule,
}: {
  spec: Spec
  entries: ModuleEntry[] | null
  runTitle: string | null
  currentRoute: string
  onRouteChange: (routeId: string) => void
  onOpenModule: (moduleId: string) => void
}) {
  const [showAll, setShowAll] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // 路线或可见范围变化后,选中模块可能不在图中:清除选中
  useEffect(() => {
    setSelectedId(null)
  }, [currentRoute, showAll])

  const route = spec.routes.find((r) => r.id === currentRoute) ?? spec.routes.find((r) => r.id === spec.system.default_route) ?? spec.routes[0]
  const selectedModule = selectedId ? (spec.modules.find((m) => m.id === selectedId) ?? null) : null
  const selectedEntry = selectedId ? (entries?.find((e) => e.module_id === selectedId) ?? null) : null

  return (
    <div className="system-intro">
      <header className="intro-head">
        <h2>{spec.system.title}</h2>
        <p className="lead">{spec.system.summary}</p>
        <p className="dim">
          说明版本 {spec.system.content_version}({spec.system.updated_at}) · 模块 {spec.module_count} 个 · 快照编译{' '}
          {spec.compiled_at}
          {runTitle ? ` · 当前运行:${runTitle}` : ''}
        </p>
      </header>

      <section className="intro-route">
        <h3>讲解路线</h3>
        <div className="chip-row" role="radiogroup" aria-label="路线选择">
          {spec.routes.map((r) => (
            <button
              key={r.id}
              role="radio"
              aria-checked={r.id === route?.id}
              className={`chip chip-link ${r.id === route?.id ? 'chip-active' : ''}`}
              onClick={() => onRouteChange(r.id)}
            >
              {r.title}
            </button>
          ))}
        </div>
        {route ? <p className="dim">{route.description}</p> : null}
      </section>

      <section>
        <h3>系统是怎么工作的</h3>
        <Blocks blocks={spec.content_sections['overview'] ?? []} />
      </section>

      <section className="intro-map">
        <div className="intro-map-head">
          <h3>流程图</h3>
          <label className="toggle-label">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            显示历史兼容 / 外部模块
          </label>
        </div>
        <p className="dim">列位置 = 依赖深度(左→右),同一列内按分组排列;连线方向即数据流方向。页面顺序不代表执行顺序。</p>
        <div className="flow-layout">
          <ModuleFlowDiagram
            spec={spec}
            entries={entries}
            routeId={route?.id ?? spec.system.default_route}
            showAll={showAll}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          <ModuleDetailPanel spec={spec} module={selectedModule} entry={selectedEntry} onOpenModule={onOpenModule} />
        </div>
      </section>
    </div>
  )
}
