// 与后端统一返回结构对应的类型(backend/envelope.py、module_service.py)

export interface Metric {
  id: string
  label: string
  value: number | string | null
  unit?: string
  denominator?: number | null
  granularity?: string
  meaning?: string
}

export interface Block {
  type: 'paragraph' | 'list' | 'code' | 'table' | 'subheading'
  text?: string
  items?: string[]
  ordered?: boolean
  lang?: string
  header?: string[]
  rows?: string[][]
  links?: { href: string }[]
  line?: number
}

export interface ArtifactMeta {
  id: string
  title: string
  kind: 'json' | 'jsonl' | 'structured' | 'directory' | 'media'
  paths: Record<string, string[]>
  record_key: string[]
  timebase: string
  schema: string
  required: boolean
}

export interface ModuleView {
  enabled: boolean
  renderer: string
  adapter: string
  filters: string[]
  primary_artifact: string
}

export interface ModuleMeta {
  id: string
  title: string
  order: number
  group: string
  lifecycle: 'active' | 'optional' | 'legacy' | 'deprecated' | 'external'
  implementation: 'existing' | 'external' | 'planned'
  summary: string
  depends_on: string[]
  optional_depends_on: string[]
  inputs: string[]
  outputs: ArtifactMeta[]
  view: ModuleView
  sections: Record<string, Block[]>
  source_refs: string[]
  evidence: string[]
}

export interface Group {
  id: string
  title: string
  order: number
}

export interface Route {
  id: string
  title: string
  modules: string[]
  description: string
}

export interface SystemMeta {
  doc_id: string
  schema_version: string
  content_version: string
  updated_at: string
  title: string
  summary: string
  code_baseline: string
  worktree_dirty: boolean
  evidence_section: string
  default_route: string
  groups: Group[]
  routes: Route[]
  layout_profiles: { id: string; title: string; description: string }[]
  sample_run?: {
    run_id: string
    workspace_relative_to_repo: string
    checked_at: string
    scope_ids: string[]
  }
}

export interface Spec {
  api_version: string
  doc_revision: string
  compiled_at: string
  system: SystemMeta
  module_count: number
  modules: ModuleMeta[]
  groups: Group[]
  routes: Route[]
  content_sections: Record<string, Block[]>
}

export interface MediaRef {
  media_id: string | null
  kind: 'video' | 'audio' | 'image'
  label: string
  url?: string
  mime?: string
  duration_sec?: number | null
  availability: 'available' | 'missing' | 'forbidden' | 'unconfigured'
  detail?: string
}

export interface TimeRef {
  timebase: string
  start_sec: number | null
  end_sec: number | null
  media_id?: string | null
  source_range?: number[] | null
}

export interface Relation {
  type: string
  module_id: string
  label: string
}

export interface Item {
  entity_id: string
  key: { kind: string; scope_id: string | null; local_id: string }
  title: string
  status: Record<string, unknown> & { execution?: string; quality?: string; reason?: string | null }
  summary: string
  media_refs: MediaRef[]
  time_refs: TimeRef[]
  relations: Relation[]
  fields: Record<string, unknown>
  provenance: Record<string, unknown>
}

export interface Envelope {
  api_version: string
  doc_revision: string
  data_revision: string
  adapter_version: string
  run_id: string
  module_id: string
  status: { execution: string; quality: string; reason: string | null }
  metrics: Metric[]
  items: Item[]
  page: { limit: number; next_cursor: string | null; total: number } | null
  warnings: string[]
  provenance: Record<string, unknown>
}

export interface ModuleEntry {
  module_id: string
  title: string
  order: number
  group: string
  lifecycle: string
  view_enabled: boolean
  renderer: string
  adapter: string
  filters: string[]
  primary_artifact: string
  status: { execution: string; quality: string; reason: string | null }
  metrics: Metric[]
  filters_available: Record<string, string[]>
  scopes: string[]
  warnings: string[]
  bound_artifacts: { artifact_id: string; available: boolean }[]
}

export interface RunSummary {
  run_id: string
  video_id: string | null
  title: string
  layout_profile: string
  route_id: string | null
  produced_code_revision: string | null
  created_at: string | null
  data_revision: string
  scopes: string[]
}

export interface VersionInfo {
  api_version: string
  doc_revision: string
  doc_compiled_at: string | null
  doc_ok: boolean
  last_document_errors: { code: string; message: string; line: number | null; field: string | null }[]
  ui_build_id: string | null
}
