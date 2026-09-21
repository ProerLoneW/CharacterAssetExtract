// 流程图右侧的模块详情面板:功能说明(文档二六段正文)、产物契约、上下游,以及
// 「跳转到产物页面」入口(文档三 §3.2:节点展示中文名、用途和运行状态,点击进入结果)。

import { Blocks } from './Blocks'
import { ExecutionBadge, LifecycleBadge } from './StatusWidgets'
import type { ModuleEntry, ModuleMeta, Spec } from '../app/types'

export function ModuleDetailPanel({
  spec,
  module,
  entry,
  onOpenModule,
}: {
  spec: Spec
  module: ModuleMeta | null
  entry: ModuleEntry | null
  onOpenModule: (moduleId: string) => void
}) {
  if (!module) {
    return (
      <aside className="module-panel">
        <div className="module-panel-empty">
          <p className="dim">点击左侧流程图中的模块节点,这里会展示:</p>
          <ul className="dim">
            <li>模块功能(来自文档二的说明)</li>
            <li>产物契约(路径、类型、时间基准、schema)</li>
            <li>上游输入与下游消费</li>
            <li>跳转到该模块的结果页</li>
          </ul>
        </div>
      </aside>
    )
  }

  const upstream = [
    ...module.depends_on.map((id) => ({ id, optional: false })),
    ...module.optional_depends_on.map((id) => ({ id, optional: true })),
  ]
    .map((u) => ({ ...u, meta: spec.modules.find((m) => m.id === u.id) }))
    .filter((u) => u.meta)
  const downstream = spec.modules.filter((m) => m.depends_on.includes(module.id) || m.optional_depends_on.includes(module.id))
  const bound = new Set(entry?.bound_artifacts.map((b) => b.artifact_id) ?? [])

  return (
    <aside className="module-panel">
      <header className="module-panel-head">
        <h3>{module.title}</h3>
        <LifecycleBadge lifecycle={module.lifecycle} />
        {entry ? <ExecutionBadge execution={entry.status.execution} reason={entry.status.reason} /> : null}
      </header>
      <p className="dim">{module.summary}</p>

      <section>
        <h4>功能说明</h4>
        <Blocks blocks={module.sections['作用与判断'] ?? []} />
      </section>

      <section>
        <h4>产物契约({module.outputs.length})</h4>
        {module.outputs.length === 0 ? (
          <p className="dim">本模块不直接拥有产物。</p>
        ) : (
          <ul className="artifact-list">
            {module.outputs.map((o) => (
              <li key={o.id} className={bound.has(o.id) ? 'bound' : ''}>
                <div className="artifact-head">
                  <span className="artifact-title">{o.title}</span>
                  {bound.has(o.id) ? <span className="chip chip-candidate">本运行已绑定</span> : <span className="chip">未绑定</span>}
                </div>
                <dl className="artifact-meta">
                  <div>
                    <dt>类型</dt>
                    <dd>{o.kind}</dd>
                  </div>
                  <div>
                    <dt>时间基准</dt>
                    <dd>{o.timebase}</dd>
                  </div>
                  <div>
                    <dt>schema</dt>
                    <dd>{o.schema}</dd>
                  </div>
                  <div>
                    <dt>必需</dt>
                    <dd>{o.required ? '必需产出' : '选配'}</dd>
                  </div>
                  <div>
                    <dt>记录键</dt>
                    <dd>{o.record_key.length ? o.record_key.join(' + ') : '(无行记录)'}</dd>
                  </div>
                </dl>
                <div className="artifact-paths">
                  {Object.entries(o.paths).map(([profile, patterns]) => (
                    <div key={profile}>
                      <code>{profile}</code>
                      <span className="dim"> {patterns.join(' | ')}</span>
                    </div>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h4>上游输入({upstream.length})</h4>
        {upstream.length ? (
          <div className="chip-row">
            {upstream.map((u) => (
              <button key={u.id} className="chip chip-link" title={u.meta!.summary} onClick={() => onOpenModule(u.id)}>
                ← {u.meta!.title}
                {u.optional ? '(可选)' : ''}
              </button>
            ))}
          </div>
        ) : (
          <p className="dim">(起点步骤,无上游)</p>
        )}
        <h4>下游消费({downstream.length})</h4>
        {downstream.length ? (
          <div className="chip-row">
            {downstream.map((m) => (
              <button key={m.id} className="chip chip-link" title={m.summary} onClick={() => onOpenModule(m.id)}>
                → {m.title}
              </button>
            ))}
          </div>
        ) : (
          <p className="dim">(末端消费,无下游)</p>
        )}
      </section>

      <div className="module-panel-jump">
        <button className="primary-button" onClick={() => onOpenModule(module.id)}>
          跳转到「{module.title}」产物页面 →
        </button>
        <p className="dim">{entry?.status.execution === 'succeeded' || entry?.status.execution === 'partial' ? '本运行有该步骤的结果。' : '本运行未收录该步骤结果,仍可查看说明。'}</p>
      </div>
    </aside>
  )
}
