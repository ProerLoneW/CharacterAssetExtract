// 状态徽章与指标(文档三 §3.4):执行/语义/质量/来源分维度呈现,状态非颜色表达。

import type { Metric } from '../app/types'

const EXECUTION_LABELS: Record<string, { label: string; symbol: string; cls: string }> = {
  succeeded: { label: '已完成', symbol: '●', cls: 'ok' },
  partial: { label: '部分完成', symbol: '◐', cls: 'warn' },
  failed: { label: '失败', symbol: '✕', cls: 'err' },
  running: { label: '运行中', symbol: '▷', cls: 'run' },
  not_run: { label: '未运行', symbol: '○', cls: 'dim' },
  not_applicable: { label: '本路线不适用', symbol: '—', cls: 'dim' },
  unconfigured: { label: '未收录', symbol: '◌', cls: 'dim' },
  unknown: { label: '未知', symbol: '?', cls: 'dim' },
}

const LIFECYCLE_LABELS: Record<string, string> = {
  active: '主线',
  optional: '可选',
  legacy: '历史兼容',
  deprecated: '已废弃',
  external: '外部',
}

export function ExecutionBadge({ execution, reason }: { execution: string; reason?: string | null }) {
  const info = EXECUTION_LABELS[execution] ?? EXECUTION_LABELS.unknown
  return (
    <span className={`badge exec-${info.cls}`} title={reason ?? info.label}>
      <span aria-hidden="true">{info.symbol}</span> {info.label}
      {reason ? <span className="badge-reason"> · {reason}</span> : null}
    </span>
  )
}

export function LifecycleBadge({ lifecycle }: { lifecycle: string }) {
  if (!lifecycle || lifecycle === 'active') return null
  return <span className={`badge lifecycle-${lifecycle}`}>{LIFECYCLE_LABELS[lifecycle] ?? lifecycle}</span>
}

export function MetricChips({ metrics }: { metrics: Metric[] }) {
  if (!metrics.length) return null
  return (
    <div className="metric-chips">
      {metrics.map((m) => (
        <div className="metric-chip" key={m.id} title={m.meaning ?? m.label}>
          <span className="metric-value">
            {fmtValue(m.value)}
            {m.unit ? ` ${m.unit}` : ''}
            {m.denominator != null && typeof m.denominator === 'number' ? ` / ${fmtValue(m.denominator)}` : ''}
          </span>
          <span className="metric-label">{m.label}</span>
        </div>
      ))}
    </div>
  )
}

function fmtValue(v: number | string | null): string {
  if (v === null) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  return v
}
