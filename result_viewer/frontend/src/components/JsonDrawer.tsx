// 开发详情抽屉:原始 JSON / 字段表 / 来源信息(文档三 §3.1 说明与证据抽屉)。

import { useState } from 'react'
import type { Item } from '../app/types'

export function JsonDrawer({ data, title = '原始 JSON' }: { data: unknown; title?: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="json-drawer">
      <button className="link-button" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? '▾' : '▸'} {title}
      </button>
      {open ? <pre className="json-view">{JSON.stringify(data, null, 2)}</pre> : null}
    </div>
  )
}

export function FieldsTable({ item, detailed }: { item: Item; detailed: boolean }) {
  const entries = Object.entries(item.fields ?? {}).filter(([k, v]) => {
    if (!detailed && (Array.isArray(v) ? v.length === 0 : v === null || v === undefined || v === '')) return false
    return true
  })
  if (!entries.length) return null
  return (
    <table className="fields-table">
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <th scope="row">{k}</th>
            <td>
              <FieldValue value={v} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function FieldValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="dim">—</span>
  if (typeof value === 'boolean') return <span className={value ? 'v-true' : 'v-false'}>{value ? '是' : '否'}</span>
  if (typeof value === 'number' || typeof value === 'string') return <>{String(value)}</>
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="dim">(空)</span>
    if (value.every((x) => typeof x === 'string' || typeof x === 'number')) {
      return (
        <span className="chip-row">
          {value.map((x, i) => (
            <span className="chip" key={i}>
              {String(x)}
            </span>
          ))}
        </span>
      )
    }
    return <NestedList value={value as unknown[]} />
  }
  return <NestedList value={value as Record<string, unknown>} />
}

function NestedList({ value }: { value: unknown[] | Record<string, unknown> }) {
  const [open, setOpen] = useState(false)
  const count = Array.isArray(value) ? `${value.length} 项` : `${Object.keys(value).length} 字段`
  return (
    <div>
      <button className="link-button" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {count}
      </button>
      {open ? <pre className="json-view small">{JSON.stringify(value, null, 2)}</pre> : null}
    </div>
  )
}
