// 系统流程图(文档三 §3.2):分层 DAG 视图。
// - 节点按"依赖深度"分列(最长上游路径),列内按分组与 order 排序;
// - 实线 = 必需依赖(depends_on),虚线 = 可选增强(optional_depends_on)——线型区分,不只靠颜色;
// - 点击节点 -> 右侧详情面板(由父组件管理选中态);
// - 支持缩放与横向滚动;不做图编辑器,只做可读的只读布局。

import { useMemo, useState } from 'react'
import type { ModuleEntry, ModuleMeta, Spec } from '../app/types'

const NODE_W = 196
const NODE_H = 44
const COL_GAP = 96
const ROW_GAP = 10
const PAD = 24

interface DiagramNode {
  module: ModuleMeta
  x: number
  y: number
  depth: number
}

interface DiagramEdge {
  fromId: string
  toId: string
  optional: boolean
}

function truncate(text: string, max = 11): string {
  return text.length > max ? text.slice(0, max) + '…' : text
}

function statusColor(execution?: string): string {
  switch (execution) {
    case 'succeeded':
      return '#1e7d43'
    case 'partial':
      return '#d9a400'
    case 'failed':
      return '#b3261e'
    default:
      return '#b9c2cf'
  }
}

export function ModuleFlowDiagram({
  spec,
  entries,
  routeId,
  showAll,
  selectedId,
  onSelect,
}: {
  spec: Spec
  entries: ModuleEntry[] | null
  routeId: string
  showAll: boolean
  selectedId: string | null
  onSelect: (moduleId: string) => void
}) {
  const [zoom, setZoom] = useState(1)

  const { nodes, edges, width, height, columns } = useMemo(() => {
    const route = spec.routes.find((r) => r.id === routeId)
    const inRoute = new Set(route?.modules ?? [])
    const groupOrder = new Map(spec.groups.map((g) => [g.id, g.order]))
    const mods = spec.modules.filter((m) => m.view.enabled && (showAll || inRoute.has(m.id)))
    const byId = new Map(mods.map((m) => [m.id, m]))

    const depsOf = (m: ModuleMeta): { dep: string; optional: boolean }[] => [
      ...m.depends_on.filter((d) => byId.has(d)).map((dep) => ({ dep, optional: false })),
      ...m.optional_depends_on.filter((d) => byId.has(d)).map((dep) => ({ dep, optional: true })),
    ]

    // 依赖深度:根节点 0,其余 = 最深上游 + 1(后端已保证无环)
    const depthMemo = new Map<string, number>()
    const depthOf = (id: string, visiting: Set<string>): number => {
      const cached = depthMemo.get(id)
      if (cached !== undefined) return cached
      if (visiting.has(id)) return 0
      visiting.add(id)
      let d = 0
      for (const { dep } of depsOf(byId.get(id)!)) {
        d = Math.max(d, depthOf(dep, visiting) + 1)
      }
      visiting.delete(id)
      depthMemo.set(id, d)
      return d
    }

    const depthList = mods.map((m) => ({ m, depth: depthOf(m.id, new Set()) }))
    const byDepth = new Map<number, { m: ModuleMeta; depth: number }[]>()
    for (const item of depthList) {
      const arr = byDepth.get(item.depth) ?? []
      arr.push(item)
      byDepth.set(item.depth, arr)
    }
    for (const arr of byDepth.values()) {
      arr.sort((a, b) => (groupOrder.get(a.m.group) ?? 0) - (groupOrder.get(b.m.group) ?? 0) || a.m.order - b.m.order)
    }

    const diagramNodes: DiagramNode[] = []
    const pos = new Map<string, { x: number; y: number }>()
    const depths = [...byDepth.keys()].sort((a, b) => a - b)
    let maxColHeight = 0
    for (const depth of depths) {
      let y = PAD
      for (const item of byDepth.get(depth)!) {
        const x = PAD + depth * (NODE_W + COL_GAP)
        diagramNodes.push({ module: item.m, x, y, depth })
        pos.set(item.m.id, { x, y })
        y += NODE_H + ROW_GAP
      }
      maxColHeight = Math.max(maxColHeight, y - ROW_GAP)
    }

    const diagramEdges: DiagramEdge[] = []
    for (const m of mods) {
      for (const { dep, optional } of depsOf(m)) {
        if (pos.has(dep) && pos.has(m.id)) {
          diagramEdges.push({ fromId: dep, toId: m.id, optional })
        }
      }
    }

    const w = PAD * 2 + (depths.length ? (depths.length - 1) * (NODE_W + COL_GAP) + NODE_W : NODE_W)
    const h = maxColHeight + PAD
    return { nodes: diagramNodes, edges: diagramEdges, width: w, height: h, columns: depths.length }
  }, [spec, routeId, showAll])

  const statusOf = (id: string) => entries?.find((e) => e.module_id === id)?.status.execution

  return (
    <div className="flow-diagram">
      <div className="flow-toolbar">
        <span className="dim flow-meta">
          {nodes.length} 个模块 · {columns} 层依赖 · 实线=必需依赖 · 虚线=可选增强
        </span>
        <div className="zoom-controls">
          <button onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.1).toFixed(2)))} aria-label="缩小">
            −
          </button>
          <button onClick={() => setZoom(1)} className="zoom-value">
            {(zoom * 100).toFixed(0)}%
          </button>
          <button onClick={() => setZoom((z) => Math.min(1.6, +(z + 0.1).toFixed(2)))} aria-label="放大">
            ＋
          </button>
        </div>
      </div>
      <div className="flow-canvas">
        <div className="flow-inner" style={{ width: width * zoom, height: height * zoom }}>
          <svg width={width * zoom} height={height * zoom} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="系统流程图">
            <defs>
              <marker id="arrow-required" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#8d99ab" />
              </marker>
              <marker id="arrow-optional" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#b3bdd0" />
              </marker>
            </defs>
            {edges.map((e, i) => {
              const from = nodes.find((n) => n.module.id === e.fromId)!
              const to = nodes.find((n) => n.module.id === e.toId)!
              const x1 = from.x + NODE_W
              const y1 = from.y + NODE_H / 2
              const x2 = to.x
              const y2 = to.y + NODE_H / 2
              const mid = (x2 - x1) / 2
              const d = `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2 - 1} ${y2}`
              const active = selectedId === e.fromId || selectedId === e.toId
              return (
                <path
                  key={i}
                  d={d}
                  className={`flow-edge ${e.optional ? 'optional' : 'required'} ${active ? 'active' : ''}`}
                  markerEnd={e.optional ? 'url(#arrow-optional)' : 'url(#arrow-required)'}
                >
                  <title>
                    {from.module.title} → {to.module.title}
                    {e.optional ? '(可选增强)' : '(必需)'}
                  </title>
                </path>
              )
            })}
            {nodes.map((n) => {
              const selected = selectedId === n.module.id
              const status = statusOf(n.module.id)
              return (
                <g
                  key={n.module.id}
                  className={`flow-node ${selected ? 'selected' : ''}`}
                  transform={`translate(${n.x}, ${n.y})`}
                  role="button"
                  tabIndex={0}
                  aria-label={`模块 ${n.module.title}`}
                  onClick={() => onSelect(n.module.id)}
                  onKeyDown={(ev) => {
                    if (ev.key === 'Enter' || ev.key === ' ') onSelect(n.module.id)
                  }}
                >
                  <rect width={NODE_W} height={NODE_H} rx={8} />
                  <circle cx={14} cy={NODE_H / 2} r={4.5} fill={statusColor(status)} />
                  <text x={26} y={NODE_H / 2 - 2}>
                    {truncate(n.module.title)}
                  </text>
                  <text className="flow-node-sub" x={26} y={NODE_H / 2 + 12}>
                    {lifecycleLabel(n.module.lifecycle)}
                  </text>
                  {n.module.lifecycle !== 'active' ? (
                    <text className="flow-node-tag" x={NODE_W - 8} y={16} textAnchor="end">
                      {tagOf(n.module.lifecycle)}
                    </text>
                  ) : null}
                  <title>{`${n.module.title}:${n.module.summary}`}</title>
                </g>
              )
            })}
          </svg>
        </div>
      </div>
      <p className="dim flow-legend">● 有结果 · ◌ 未收录 · 点击节点在右侧查看模块说明与产物,并可跳转到它的结果页</p>
    </div>
  )
}

function lifecycleLabel(l: string): string {
  const map: Record<string, string> = {
    active: '',
    optional: '可选步骤',
    legacy: '历史兼容链',
    external: '外部输入/结果',
    deprecated: '已废弃',
  }
  return map[l] ?? ''
}

function tagOf(l: string): string {
  const map: Record<string, string> = { optional: '选', legacy: '旧', external: '外', deprecated: '废' }
  return map[l] ?? ''
}
