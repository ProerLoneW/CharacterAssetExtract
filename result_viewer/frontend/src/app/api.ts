// API 客户端:全部使用相对路径 fetch,天然匹配平台代理前缀(文档三 §9.2)。
// query 路由保证页面路径不变,相对路径始终解析到同一前缀下。

import type { Envelope, ModuleEntry, RunSummary, Spec, VersionInfo } from './types'

export class ApiRequestError extends Error {
  code: string
  status: number
  location: unknown

  constructor(code: string, message: string, status: number, location?: unknown) {
    super(message)
    this.code = code
    this.status = status
    this.location = location
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: 'application/json' } })
  if (!res.ok) {
    let code = 'HTTP_ERROR'
    let message = `请求失败: ${res.status} ${path}`
    let location: unknown
    try {
      const body = await res.json()
      if (body?.error) {
        code = body.error.code ?? code
        message = body.error.message ?? message
        location = body.error.location
      }
    } catch {
      /* 非 JSON 错误体 */
    }
    throw new ApiRequestError(code, message, res.status, location)
  }
  return (await res.json()) as T
}

export const api = {
  version: () => get<VersionInfo>('api/v1/version'),
  spec: () => get<Spec>('api/v1/spec'),
  runs: () => get<{ runs: RunSummary[] }>('api/v1/runs'),
  runModules: (runId: string) => get<{ modules: ModuleEntry[]; data_revision: string }>(`api/v1/runs/${enc(runId)}/modules`),
  items: (runId: string, moduleId: string, params: Record<string, string | null | undefined>, signal?: AbortSignal) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== '') qs.set(k, v)
    }
    return fetch(`api/v1/runs/${enc(runId)}/modules/${enc(moduleId)}/items?${qs}`, { signal }).then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new ApiRequestError(body?.error?.code ?? 'HTTP_ERROR', body?.error?.message ?? `请求失败: ${res.status}`, res.status, body?.error?.location)
      }
      return (await res.json()) as Envelope
    })
  },
  item: (runId: string, moduleId: string, entityId: string) =>
    get<Envelope>(`api/v1/runs/${enc(runId)}/modules/${enc(moduleId)}/items/${enc(entityId)}`),
}

function enc(x: string): string {
  return encodeURIComponent(x)
}
