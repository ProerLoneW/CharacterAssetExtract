// 模块文档抽屉:展示文档二该模块的六段正文(文档三 §3.1:
// 模块说明来自文档二同模块六段正文,不在前端复制改写另一版本)。

import { useState } from 'react'
import { Blocks } from './Blocks'
import type { ModuleMeta } from '../app/types'

const SECTION_ORDER = ['作用与判断', '输入与前置条件', '处理过程', '产物与消费', '异常、重跑与成本', '展示与验收']

export function ModuleDocs({ module }: { module: ModuleMeta }) {
  const [open, setOpen] = useState(false)
  return (
    <section className="module-docs">
      <button className="drawer-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? '收起本步骤说明' : '查看本步骤说明(输入、处理、异常与验收)'}
      </button>
      {open ? (
        <div className="module-docs-body">
          {SECTION_ORDER.map((name) => (
            <div className="doc-section" key={name}>
              <h4>{name}</h4>
              <Blocks blocks={module.sections[name] ?? []} />
            </div>
          ))}
          <div className="doc-section">
            <h4>实现映射(开发)</h4>
            <ul>
              <li>
                源码:{' '}
                {(module.source_refs ?? []).length ? (
                  module.source_refs.map((r) => (
                    <span className="chip" key={r}>
                      {r}
                    </span>
                  ))
                ) : (
                  <span className="dim">(未登记)</span>
                )}
              </li>
              <li>
                证据等级:{' '}
                {(module.evidence ?? []).map((e) => (
                  <span className="chip" key={e}>
                    {e}
                  </span>
                ))}
              </li>
              <li>
                渲染器 <code>{module.view.renderer}</code> · 适配器 <code>{module.view.adapter}</code>
              </li>
            </ul>
          </div>
        </div>
      ) : null}
    </section>
  )
}
