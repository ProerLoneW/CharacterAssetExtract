"""Single-file HTML template for the asset viewer dashboard.

Mirrors the pattern used in ``review_mv_caption.py`` (one ``HTML_TEMPLATE``
constant holding inline HTML/CSS/JS).  The page is a 3-pane layout
(workspace / stage pickers, asset grid, detail drawer) rendered with
vanilla JavaScript so we don't pull in React/Vue.

The backend serialises the workspace list as JSON in a ``<script>`` tag
so the page can render before any API request completes.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Centralised Chinese strings.  All UI text rendered in the dashboard comes
# from this dictionary so localisation stays in one place.  The Python side
# ships the dictionary to the browser as part of ``__INITIAL_JSON__`` and
# the JS reads ``INITIAL.zh.*`` (mirrored as ``ZH`` below) instead of using
# hard-coded English literals.
# ---------------------------------------------------------------------------
ZH = {
    "title": "0916 Pipeline · 全流程产物",
    "app_title": "0916 Pipeline 产物",
    "workspace_label": "工作区",
    "stage_label": "阶段",
    "scope_label": "范围",
    "subject_label": "主体",
    "reset_btn_title": "重置为 URL 默认",
    "permalink": "永久链接",

    # Top-level sidebar
    "kpis_header": "工作区 KPI",
    "scope_breakdown_header": "范围明细",

    # Main pane
    "main_loading": "加载中…",
    "detail_header": "详情",
    "drawer_empty": "点击任意瓦片查看详情。",

    # KPI labels
    "kpi_character_count": "角色数",
    "kpi_tracklet_count": "轨道数",
    "kpi_assigned_tracklet_count": "已分配",
    "kpi_voice_count": "音频片段",
    "kpi_scene_count": "场景数",
    "kpi_global_clip_count": "全局片段",
    "kpi_key_object_count": "关键目标数",
    "kpi_i2i_bundle_count": "I2I 束数",
    "kpi_shot_count": "Shot 数",
    "kpi_segment_count": "Segment 数",
    "kpi_caption_count": "Caption 数",
    "kpi_artifact_count": "产物文件",

    # Stage labels (Stage dropdown)
    "stage_characters": "Stage 5 · 角色",
    "stage_alignment": "Stage 1.5 · 全局对齐",
    "stage_materialization": "Stage 1.5 · 物化",
    "stage_asset_bank": "Stage 4 · 资产库",
    "stage_tracklets": "Stage 2 · 轨道",
    "stage_voice": "音频资产",
    "stage_global_clips": "Stage 1 · 全局片段",
    "stage_shots": "Stage 1 · Shot 切片",
    "stage_captions_shot": "Stage 2 · Shot Caption",
    "stage_captions_global": "Stage 2 · Global Caption",
    "stage_captions_segment": "Stage 2 · Segment Caption",
    "stage_segments": "Stage 3 · Segment 成片",
    "stage_scene": "Stage 1.5 · 场景索引",
    "stage_scene_opt": "Stage 1.5 · 场景精排",
    "stage_key_object": "Stage 1.5 · 关键目标",
    "stage_i2i_bundles": "Stage 1.5 · I2I 束",
    "stage_i2i_test": "I2I 模型效果对比",
    "stage_tts_test": "TTS 模型效果对比",
    "stage_artifacts": "全部产物文件",

    # Buttons / chips
    "btn_pick_character": "请选择一个角色",
    "btn_pick_global_scope": "请选择一个全局范围",

    # Empty-state messages
    "empty_no_entities": "此阶段暂无实体。",
    "empty_no_characters": "暂无角色。",
    "empty_no_global_scopes": "没有可用的全局对齐范围。",
    "empty_voice_overview": "音频概览：请在 Stage 5 视图中选择一个角色。",
    "empty_no_workspace": (
        "该根目录下未检测到工作区。请通过 <code>--workspace-root</code> 指定 "
        "<code>movie_full_pipeline_*</code> 文件夹。"
    ),
    "empty_no_data": "暂无数据。",
    "empty_error_prefix": "加载失败：",

    # Section headers
    "section_overview": "概览",
    "section_tracklet": "轨道",
    "section_gallery": "画廊",
    "section_alignment": "对齐",
    "section_global_clips": "全局片段",
    "section_scene": "场景索引",
    "section_key_object": "关键目标观测",
    "section_i2i_bundles": "I2I 束",
    "section_role_character": "角色定位",
    "section_stitched_video": "拼接轨道视频 ({scope})",
    "section_per_tracklet": "逐轨道对齐",

    # Tracklet / alignment table headers
    "th_tracklet": "轨道",
    "th_subject": "主体",
    "th_tier": "层级",
    "th_confidence": "置信度",
    "th_visual_cues": "视觉线索",
    "th_reason": "原因",
    "th_shot": "镜头",
    "th_tracklets": "轨道数",
    "th_face": "人脸",
    "th_body": "身体",
    "th_rgba": "RGBA",
    "th_voice": "音频",
    "th_missing": "缺失",
    "th_scope_subject": "范围:主体",
    "th_chr": "CHR",
    "th_shots": "镜头数",
    "th_scene": "场景",
    "th_candidates": "候选数",
    "th_observations": "观测数",
    "th_crops": "裁剪图",
    "th_object": "目标",
    "th_category": "类别",
    "th_bundle_id": "束 ID",
    "th_size": "大小",
    "th_frame_count": "帧数",
    "th_source_clip": "来源片段",
    "th_no_audio": "（无音频）",

    # Confidence / status labels
    "label_confidence": "置信度",
    "label_tracklets": "轨道数",
    "label_face": "人脸",
    "label_body": "身体",
    "label_voice": "音频",

    # Badges
    "badge_face": "人脸",
    "badge_body": "身体",
    "badge_voice": "音频",
    "badge_tracklet_count": "{n} trk",

    # Drawer labels
    "drawer_shot": "镜头",
    "drawer_tracklet": "轨道",
    "drawer_time_range": "时间范围",
    "drawer_observations": "观测",
    "drawer_face_gallery": "人脸画廊",
    "drawer_body_gallery": "身体画廊",

    # Error messages
    "error_load_failed": "请求失败，请稍后再试。",
    "error_invalid_payload": "服务器返回的数据格式无效。",
    "error_missing_param": "缺少必要的参数：{params}",

    # Global clips stage
    "global_clip_meta_resolution": "分辨率",
    "global_clip_meta_duration": "时长",
    "global_clip_meta_fps": "帧率",
    "global_clip_meta_codec": "编解码",

    # Scene stage
    "scene_section_candidates": "候选预览",

    # Key object stage
    "keyobject_section_observations": "观测列表",
    "keyobject_section_crops": "裁剪画廊",

    # I2I bundle stage
    "i2i_section_payload": "完整 JSON",
}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__ZH_TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #637083;
  --line: #d8dee8;
  --accent: #0f766e;
  --accent-soft: #d8f3ee;
  --warn: #b45309;
  --bad: #b91c1c;
  --ok: #15803d;
  --shadow: 0 1px 2px rgba(31, 41, 51, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
button, input, textarea, select { font: inherit; }
button {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}
button:hover { border-color: var(--accent); }
button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
textarea, input[type="text"], input[type="number"] {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #fff;
  color: var(--ink);
}
select {
  max-width: 260px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 28px 7px 8px;
  background: #fff;
  color: var(--ink);
}
.app {
  display: grid;
  grid-template-columns: minmax(260px, 0.9fr) minmax(520px, 2.2fr) minmax(320px, 1fr);
  grid-template-rows: auto 1fr;
  height: 100vh;
  overflow: hidden;
}
.topbar {
  grid-column: 1 / -1;
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
}
.topbar h1 {
  font-size: 15px;
  margin: 0 14px 0 0;
  letter-spacing: 0.2px;
  color: var(--accent);
}
.topbar label { color: var(--muted); margin-right: 4px; }
.topbar .url-link {
  margin-left: auto;
  color: var(--muted);
  font-size: 12px;
  text-decoration: none;
}
.topbar .url-link:hover { color: var(--accent); }

.sidebar {
  min-width: 0;
  overflow: auto;
  padding: 12px;
  background: var(--panel);
  border-right: 1px solid var(--line);
}
.sidebar h2 { font-size: 13px; margin: 4px 0 6px; color: var(--muted); letter-spacing: 0.3px; text-transform: uppercase; }
.sidebar .kpi-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.kpi {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: var(--bg);
}
.kpi .num { font-size: 18px; font-weight: 600; color: var(--ink); }
.kpi .lbl { color: var(--muted); font-size: 11px; }

.main {
  min-width: 0;
  min-height: 0;
  overflow: auto;
  padding: 14px;
}
.main h2 { margin: 0 0 10px; font-size: 14px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.tile {
  position: relative;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  box-shadow: var(--shadow);
  transition: transform 0.05s ease;
}
.tile:hover { transform: translateY(-1px); border-color: var(--accent); }
.tile img, .tile video, .tile audio {
  display: block;
  width: 100%;
  height: 140px;
  object-fit: cover;
  background: #111;
}
.tile .label {
  font-size: 12px;
  padding: 6px 8px;
  border-top: 1px solid var(--line);
  background: var(--bg);
}
.tile .badge {
  position: absolute;
  top: 6px;
  left: 6px;
  background: rgba(15, 118, 110, 0.85);
  color: #fff;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 4px;
}
.tile .badge.bad { background: rgba(185, 28, 28, 0.85); }
.tile .badge.muted { background: rgba(99, 112, 131, 0.85); }

.drawer {
  min-width: 0;
  overflow: auto;
  padding: 14px;
  background: var(--panel);
  border-left: 1px solid var(--line);
}
.drawer h2 { font-size: 14px; margin: 0 0 8px; }
.drawer dl { font-size: 12px; color: var(--muted); margin: 0 0 12px; }
.drawer dt { color: var(--ink); font-weight: 600; margin-top: 6px; }
.drawer dd { margin: 0 0 6px 0; color: var(--muted); }
.drawer .video-wrap, .drawer .audio-wrap {
  margin-bottom: 12px;
  background: var(--bg);
  padding: 8px;
  border-radius: 6px;
  border: 1px solid var(--line);
}
.drawer video { width: 100%; border-radius: 4px; background: #111; }
.drawer audio { width: 100%; }

.section {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--panel);
  margin-bottom: 12px;
}
.section h3 { font-size: 12px; color: var(--muted); margin: 0 0 6px; letter-spacing: 0.3px; text-transform: uppercase; }

.bar-chart { display: flex; gap: 4px; align-items: flex-end; height: 80px; margin-top: 8px; }
.bar-chart .bar { background: var(--accent); border-radius: 3px 3px 0 0; flex: 1; min-width: 16px; position: relative; }
.bar-chart .bar.unknown { background: var(--muted); }
.bar-chart .bar .lbl { position: absolute; bottom: -16px; left: 50%; transform: translateX(-50%); font-size: 10px; color: var(--muted); white-space: nowrap; }

table.tracklets { width: 100%; border-collapse: collapse; font-size: 12px; }
table.tracklets th, table.tracklets td { padding: 4px 6px; text-align: left; border-bottom: 1px solid var(--line); }
table.tracklets th { background: var(--bg); color: var(--muted); font-weight: 500; }
table.tracklets tr:hover td { background: var(--accent-soft); cursor: pointer; }
table.tracklets .conf-bar { display: inline-block; height: 6px; background: var(--accent); border-radius: 3px; vertical-align: middle; }

.empty-state { color: var(--muted); padding: 40px 16px; text-align: center; }
.muted { color: var(--muted); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.ok { color: var(--ok); }

.toast {
  position: fixed;
  bottom: 16px;
  right: 16px;
  background: var(--ink);
  color: #fff;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  z-index: 50;
  box-shadow: var(--shadow);
}
.toast.bad { background: var(--bad); }
.toast.warn { background: var(--warn); }

.json-viewer { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.json-viewer details { border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; margin-bottom: 6px; background: var(--bg); }
.json-viewer details > summary { cursor: pointer; color: var(--accent); }
.json-viewer pre { white-space: pre-wrap; word-break: break-all; margin: 6px 0 0; color: var(--ink); }
.video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.video-grid .tile video { height: 160px; }
.crop-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
.caption-card { border-left: 3px solid var(--accent); }
.caption-card .prose, .prose { white-space: pre-wrap; line-height: 1.65; }
.caption-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.caption-meta span { background: var(--accent-soft); color: var(--accent); border-radius: 12px; padding: 2px 8px; font-size: 11px; }
.toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
.toolbar input { flex: 1; }
.file-kind { display: inline-block; min-width: 44px; text-align: center; border-radius: 4px; padding: 1px 5px; background: var(--bg); color: var(--muted); font-size: 10px; }
.drawer pre.raw { max-height: 55vh; overflow: auto; white-space: pre-wrap; word-break: break-word; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 11px; }
.json-actions { display: flex; justify-content: flex-end; margin: 6px 0; }
.json-copy-btn { color: var(--accent); padding: 4px 9px; font-size: 11px; }
.audio-result { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 8px; }
.audio-result audio { display: block; width: 100%; margin-top: 4px; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <h1>__ZH_APP_TITLE__</h1>
    <label id="ws-label">__ZH_WORKSPACE_LABEL__</label>
    <select id="ws-picker"></select>
    <label>__ZH_STAGE_LABEL__</label>
    <select id="stage-picker"></select>
    <label id="scope-label">__ZH_SCOPE_LABEL__</label>
    <select id="scope-picker"></select>
    <label id="subject-label">__ZH_SUBJECT_LABEL__</label>
    <select id="subject-picker"></select>
    <button id="reset-btn" title="__ZH_RESET_BTN_TITLE__">⟳</button>
    <a class="url-link" id="url-link" target="_blank" rel="noopener">__ZH_PERMALINK__</a>
  </div>
  <aside class="sidebar" id="sidebar">
    <h2>__ZH_KPIS_HEADER__</h2>
    <div class="kpi-grid" id="kpi-grid"></div>
    <h2 style="margin-top:14px">__ZH_SCOPE_BREAKDOWN_HEADER__</h2>
    <div id="scope-breakdown"></div>
  </aside>
  <main class="main" id="main">
    <h2 id="main-title">__ZH_MAIN_LOADING__</h2>
    <div id="main-body"></div>
  </main>
  <aside class="drawer" id="drawer">
    <h2>__ZH_DETAIL_HEADER__</h2>
    <div id="drawer-body" class="empty-state">__ZH_DRAWER_EMPTY__</div>
  </aside>
</div>

<script>
const INITIAL = __INITIAL_JSON__;
const ZH = INITIAL.zh || {};

// Choose the default workspace.  Prefer the explicit ``default_workspace``
// from the backend, fall back to ``movie_full_pipeline_20260916`` if it
// exists in the workspace list, otherwise take the first entry.
function pickDefaultWorkspace() {
  if (INITIAL.default_workspace && INITIAL.workspaces.some(w => w.name === INITIAL.default_workspace)) {
    return INITIAL.default_workspace;
  }
  const preferred = INITIAL.workspaces.find(w => w.name === 'movie_full_pipeline_20260916');
  if (preferred) return preferred.name;
  return INITIAL.workspaces[0]?.name || '';
}

const state = {
  workspace: pickDefaultWorkspace(),
  chain: '',
  stage: 'captions_global',
  scope: '',
  subject: '',
  shotId: '',
  trackletId: '',
  workspaceMap: new Map(),
  entityList: [],
};

for (const w of INITIAL.workspaces) {
  state.workspaceMap.set(w.name, w);
}

const $ = (id) => document.getElementById(id);
const api = {
  // Relative URLs keep the viewer working both at / and behind Jupyter's
  // /proxy/<port>/ prefix.
  workspaces: () => fetch('api/workspaces').then(r => r.json()),
  stages: (ws) => fetch(`api/stages?workspace=${encodeURIComponent(ws)}`).then(r => r.json()),
  entities: (ws, stage) => fetch(`api/entities?workspace=${encodeURIComponent(ws)}&stage=${encodeURIComponent(stage)}`).then(r => r.json()),
  character: (ws, scope, subject) => fetch(`api/character?workspace=${encodeURIComponent(ws)}&scope=${encodeURIComponent(scope)}&subject=${encodeURIComponent(subject)}`).then(r => r.json()),
  global: (ws, scope) => fetch(`api/global?workspace=${encodeURIComponent(ws)}&scope=${encodeURIComponent(scope)}`).then(r => r.json()),
  tracklet: (ws, shotId, tid) => fetch(`api/tracklet?workspace=${encodeURIComponent(ws)}&shot_id=${encodeURIComponent(shotId)}&tracklet_id=${encodeURIComponent(tid)}`).then(r => r.json()),
  // 列举 + 单个详情都用 /api/entities 做列表，单个详情走专门端点
  globalClip: (ws, scope) => fetch(`api/global_clip?workspace=${encodeURIComponent(ws)}&scope=${encodeURIComponent(scope)}`).then(r => r.json()),
  sceneScope: (ws, scope, opt) => fetch(`api/scene?workspace=${encodeURIComponent(ws)}&scope=${encodeURIComponent(scope)}&optimized=${opt ? 1 : 0}`).then(r => r.json()),
  keyObjectScope: (ws, scope) => fetch(`api/key_object?workspace=${encodeURIComponent(ws)}&scope=${encodeURIComponent(scope)}`).then(r => r.json()),
  i2iBundle: (ws, id) => fetch(`api/i2i_bundle?workspace=${encodeURIComponent(ws)}&id=${encodeURIComponent(id)}`).then(r => r.json()),
  detail: (ws, stage, id) => fetch(`api/detail?workspace=${encodeURIComponent(ws)}&stage=${encodeURIComponent(stage)}&id=${encodeURIComponent(id)}`).then(r => r.json()),
  artifacts: (ws, prefix = '') => fetch(`api/artifacts?workspace=${encodeURIComponent(ws)}&prefix=${encodeURIComponent(prefix)}`).then(r => r.json()),
  artifact: (ws, path) => fetch(`api/artifact?workspace=${encodeURIComponent(ws)}&path=${encodeURIComponent(path)}`).then(r => r.json()),
  media: (ws, abs) => `media/path?workspace=${encodeURIComponent(ws)}&abs=${encodeURIComponent(abs)}`,
};
const fmtBytes = (n) => {
  if (n == null) return '–';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
  return `${(n/1024/1024).toFixed(2)} MB`;
};
const fmtSeconds = (n) => {
  if (n == null) return '–';
  return `${Number(n).toFixed(2)} s`;
};
const esc = (value) => String(value ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

function toast(msg, kind = '') {
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed'; area.style.opacity = '0';
      document.body.appendChild(area); area.select();
      if (!document.execCommand('copy')) throw new Error('copy command failed');
      area.remove();
    }
    toast('JSON 已复制');
  } catch (error) {
    toast(`复制失败：${error.message || error}`, 'bad');
  }
}

function addCopyButton(container, pre, label = '复制 JSON') {
  const actions = document.createElement('div');
  actions.className = 'json-actions';
  const button = document.createElement('button');
  button.className = 'json-copy-btn';
  button.type = 'button';
  button.textContent = label;
  button.addEventListener('click', (event) => {
    event.preventDefault(); event.stopPropagation();
    copyText(pre.textContent || '');
  });
  actions.appendChild(button);
  container.insertBefore(actions, pre);
}

function setUrl() {
  const params = new URLSearchParams();
  if (state.workspace) params.set('workspace', state.workspace);
  if (state.stage) params.set('stage', state.stage);
  if (state.scope) params.set('scope', state.scope);
  if (state.subject) params.set('subject', state.subject);
  if (state.shotId) params.set('shot', state.shotId);
  if (state.trackletId) params.set('tracklet', state.trackletId);
  const qs = params.toString();
  const url = window.location.pathname + (qs ? '?' + qs : '');
  history.replaceState({}, '', url);
  $('url-link').href = url;
}

function populateWorkspacePicker() {
  const sel = $('ws-picker');
  sel.innerHTML = '';
  for (const w of INITIAL.workspaces) {
    const opt = document.createElement('option');
    opt.value = w.name;
    opt.textContent = `${w.name} · ${w.chain}`;
    if (w.name === state.workspace) opt.selected = true;
    sel.appendChild(opt);
  }
  const singleton = INITIAL.workspaces.length === 1;
  $('ws-label').style.display = singleton ? 'none' : '';
  sel.style.display = singleton ? 'none' : '';
}

async function loadStagePicker() {
  const data = await api.stages(state.workspace);
  const w = state.workspaceMap.get(state.workspace);
  state.chain = (data.chain || w?.chain || '');
  const sel = $('stage-picker');
  sel.innerHTML = '';
  const labels = {
    'characters': ZH.stage_characters,
    'alignment': ZH.stage_alignment,
    'materialization': ZH.stage_materialization,
    'asset_bank': ZH.stage_asset_bank,
    'tracklets': ZH.stage_tracklets,
    'voice': ZH.stage_voice,
    'global_clips': ZH.stage_global_clips,
    'shots': ZH.stage_shots,
    'captions_shot': ZH.stage_captions_shot,
    'captions_global': ZH.stage_captions_global,
    'captions_segment': ZH.stage_captions_segment,
    'segments': ZH.stage_segments,
    'scene': ZH.stage_scene,
    'scene_opt': ZH.stage_scene_opt,
    'key_object': ZH.stage_key_object,
    'i2i_bundles': ZH.stage_i2i_bundles,
    'i2i_test': ZH.stage_i2i_test,
    'tts_test': ZH.stage_tts_test,
    'artifacts': ZH.stage_artifacts,
  };
  const stages = data.stages || [];
  for (const s of stages) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = labels[s] || s;
    if (s === state.stage) opt.selected = true;
    sel.appendChild(opt);
  }
  if (!stages.includes(state.stage) && stages.length) {
    state.stage = stages[0];
    sel.value = state.stage;
  }
  const entityData = state.stage ? await api.entities(state.workspace, state.stage) : { entities: [] };
  state.entityList = entityData.entities || [];
  renderKPIs(data);
  renderScopeBreakdown();
  populateScopePicker();
  populateSubjectPicker();
}

function renderKPIs(data) {
  const kpis = data.kpis || {};
  const grid = $('kpi-grid');
  grid.innerHTML = '';
  const items = [
    ['character_count', ZH.kpi_character_count],
    ['tracklet_count', ZH.kpi_tracklet_count],
    ['assigned_tracklet_count', ZH.kpi_assigned_tracklet_count],
    ['voice_count', ZH.kpi_voice_count],
    ['scene_count', ZH.kpi_scene_count],
    ['global_clip_count', ZH.kpi_global_clip_count],
    ['key_object_count', ZH.kpi_key_object_count],
    ['i2i_bundle_count', ZH.kpi_i2i_bundle_count],
    ['shot_count', ZH.kpi_shot_count],
    ['segment_count', ZH.kpi_segment_count],
    ['caption_count', ZH.kpi_caption_count],
    ['artifact_count', ZH.kpi_artifact_count],
  ];
  for (const [k, lbl] of items) {
    const div = document.createElement('div');
    div.className = 'kpi';
    div.innerHTML = `<div class="num">${kpis[k] ?? '–'}</div><div class="lbl">${lbl}</div>`;
    grid.appendChild(div);
  }
}

function renderScopeBreakdown() {
  const wrap = $('scope-breakdown');
  wrap.innerHTML = '';
  const groups = state.entityList.filter(e => e.kind === 'character');
  if (!groups.length) {
    const counts = {};
    for (const entity of state.entityList) {
      const scope = entity.scope || entity.scope_id || (entity.kind === 'artifact_group' ? entity.id : '');
      if (scope) counts[scope] = (counts[scope] || 0) + (entity.file_count || 1);
    }
    if (!Object.keys(counts).length) {
      wrap.innerHTML = `<div class="muted">${ZH.empty_no_data}</div>`;
      return;
    }
    for (const [scope, count] of Object.entries(counts).sort()) {
      const button = document.createElement('button');
      button.className = 'chip-btn';
      button.textContent = `${scope} · ${count}`;
      button.style.margin = '0 4px 6px 0';
      if (stageHasScope()) button.addEventListener('click', () => {
        state.scope = scope; populateScopePicker(); setUrl(); renderMain();
      });
      wrap.appendChild(button);
    }
    return;
  }
  const byScope = {};
  for (const e of groups) {
    const [scope, subject] = e.id.split(':');
    byScope[scope] = byScope[scope] || [];
    byScope[scope].push(subject);
  }
  for (const [scope, subjects] of Object.entries(byScope).sort()) {
    const div = document.createElement('div');
    div.className = 'section';
    div.innerHTML = `<h3>${scope}</h3>`;
    const chips = subjects.sort().map(s =>
      `<button data-scope="${scope}" data-subject="${s}" class="chip-btn">${s}</button>`
    ).join(' ');
    div.insertAdjacentHTML('beforeend', `<div>${chips}</div>`);
    wrap.appendChild(div);
  }
  wrap.querySelectorAll('button.chip-btn').forEach(b => {
    b.addEventListener('click', () => {
      state.scope = b.dataset.scope;
      state.subject = b.dataset.subject;
      populateSubjectPicker();
      setUrl();
      renderMain();
    });
  });
}

// Decide whether the current stage needs a scope / subject dropdown. Stages
// without subjects (e.g. global_clips, scene) keep a single pseudo-scope
// so the existing pipeline keeps working without changing the URL shape.
function stageHasScope() {
  return ['characters', 'alignment', 'materialization', 'asset_bank',
          'global_clips', 'shots', 'captions_shot', 'captions_global',
          'captions_segment', 'segments', 'voice', 'scene', 'key_object',
          'i2i_bundles', 'tts_test'].includes(state.stage);
}
function stageHasSubject() {
  return ['characters', 'materialization'].includes(state.stage);
}

function populateScopePicker() {
  const sel = $('scope-picker');
  sel.innerHTML = '<option value="">—</option>';
  const scopes = new Set();
  for (const e of state.entityList) {
    if (e.kind === 'character') {
      const [s] = e.id.split(':');
      scopes.add(s);
    } else if (e.kind === 'global') {
      scopes.add(e.id);
    } else if (e.kind === 'global_clip') {
      scopes.add(e.scope || e.id);
    } else if (e.kind === 'scene') {
      scopes.add(e.id);
    } else if (e.kind === 'key_object') {
      scopes.add(e.id);
    } else if (e.kind === 'i2i_bundle') {
      scopes.add(e.id);
    } else if (e.scope) {
      scopes.add(e.scope);
    }
  }
  // Stages that only ever have one logical scope — synthesise a placeholder
  // so the dropdown is never empty but the value doesn't fragment the URL.
  if (!scopes.size && ['global_clips', 'scene', 'key_object', 'i2i_bundles'].includes(state.stage)) {
    scopes.add(state.stage);
  }
  for (const s of [...scopes].sort()) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    if (s === state.scope) opt.selected = true;
    sel.appendChild(opt);
  }
  // If the previous scope isn't valid for this stage, reset to the first one.
  if (sel.options.length > 1 && ![...sel.options].some(o => o.value === state.scope && o.value)) {
    state.scope = sel.options[1].value;
    sel.value = state.scope;
  }
  populateSubjectPicker();
}

function populateSubjectPicker() {
  const sel = $('subject-picker');
  sel.innerHTML = '<option value="">—</option>';
  if (!stageHasSubject()) {
    sel.disabled = true;
    state.subject = '';
    return;
  }
  sel.disabled = false;
  const subjects = state.entityList
    .filter(e => e.kind === 'character' && e.id.startsWith(state.scope + ':'))
    .map(e => e.id.split(':')[1]);
  for (const s of [...new Set(subjects)].sort()) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    if (s === state.subject) opt.selected = true;
    sel.appendChild(opt);
  }
}

function showScopePicker(visible) {
  $('scope-label').style.display = visible ? '' : 'none';
  $('scope-picker').style.display = visible ? '' : 'none';
}
function showSubjectPicker(visible) {
  $('subject-label').style.display = visible ? '' : 'none';
  $('subject-picker').style.display = visible ? '' : 'none';
}

function applyPickerVisibility() {
  showScopePicker(stageHasScope());
  showSubjectPicker(stageHasSubject());
}

async function renderMain() {
  const main = $('main-body');
  main.innerHTML = '';
  const title = $('main-title');
  title.textContent = `${state.workspace} · ${state.stage}`;
  applyPickerVisibility();
  if (!state.entityList.length) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_entities}</div>`;
    return;
  }

  if (state.stage === 'characters') {
    if (!state.scope || !state.subject) {
      renderCharacterList(main);
      return;
    }
    await renderCharacterDetail(main);
  } else if (state.stage === 'alignment') {
    await renderAlignmentOverview(main);
  } else if (state.stage === 'materialization') {
    renderMaterializationTable(main);
  } else if (state.stage === 'asset_bank') {
    renderAssetBankTable(main);
  } else if (state.stage === 'tracklets') {
    renderTrackletList(main);
  } else if (state.stage === 'voice') {
    renderVoiceOverview(main);
  } else if (state.stage === 'global_clips') {
    await renderGlobalClips(main);
  } else if (state.stage === 'shots') {
    renderMediaResults(main, 'shots');
  } else if (['captions_shot', 'captions_global', 'captions_segment'].includes(state.stage)) {
    renderCaptions(main);
  } else if (state.stage === 'segments') {
    renderMediaResults(main, 'segments');
  } else if (state.stage === 'scene' || state.stage === 'scene_opt') {
    await renderScene(main);
  } else if (state.stage === 'key_object') {
    await renderKeyObject(main);
  } else if (state.stage === 'i2i_bundles') {
    await renderI2IBundles(main);
  } else if (state.stage === 'i2i_test') {
    renderI2ITest(main);
  } else if (state.stage === 'tts_test') {
    renderTTSTest(main);
  } else if (state.stage === 'artifacts') {
    await renderArtifacts(main);
  }
}

function renderCharacterList(main) {
  const groups = state.entityList.filter(e => e.kind === 'character');
  const grid = document.createElement('div');
  grid.className = 'grid';
  for (const e of groups) {
    const div = document.createElement('div');
    div.className = 'tile';
    div.innerHTML = `
      <div class="badge">${e.tracklet_count} trk</div>
      <div class="label">
        <strong>${e.id}</strong><br>
        <span class="muted">${ZH.label_face} ${e.face_copied ?? '–'} · ${ZH.label_body} ${e.body_copied ?? '–'} · ${ZH.label_voice} ${e.voice_copied ?? '–'}</span>
      </div>`;
    div.addEventListener('click', () => {
      const [s, sj] = e.id.split(':');
      state.scope = s; state.subject = sj;
      populateScopePicker();
      setUrl();
      renderMain();
    });
    grid.appendChild(div);
  }
  main.appendChild(grid);
}

async function renderCharacterDetail(main) {
  const data = await api.character(state.workspace, state.scope, state.subject);
  if (!data.ok) {
    main.innerHTML = `<div class="empty-state bad">${data.error}</div>`;
    return;
  }
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div class="section">
      <h3>${ZH.section_role_character}</h3>
      <div><strong>${data.scope_id} : ${data.subject_id}</strong>
        &nbsp;<span class="muted">${data.role_character.schema_version || ''}</span></div>
      <div class="muted">${ZH.label_confidence}: ${(data.role_character.mean_alignment_confidence ?? 0).toFixed(3)} · ${ZH.label_tracklets}: ${data.role_character.tracklet_count}</div>
    </div>`;
  const grid = document.createElement('div');
  grid.className = 'grid';
  const faceEntries = data.faces;
  for (const f of faceEntries) {
    const t = document.createElement('div');
    t.className = 'tile';
    t.innerHTML = `
      <div class="badge">${ZH.badge_face}</div>
      <img loading="lazy" src="${api.media(state.workspace, f.path)}" alt="">
      <div class="label"><span class="muted">${f.filename}</span><br>${fmtBytes(f.size)}</div>`;
    t.addEventListener('click', () => openDrawer({
      kind: 'image',
      title: f.filename,
      url: api.media(state.workspace, f.path),
      meta: { scope: data.scope_id, subject: data.subject_id, size: fmtBytes(f.size) },
    }));
    grid.appendChild(t);
  }
  if (data.bodies.length) {
    for (const b of data.bodies.slice(0, 8)) {
      const t = document.createElement('div');
      t.className = 'tile';
      t.innerHTML = `
        <div class="badge muted">${ZH.badge_body}</div>
        <img loading="lazy" src="${api.media(state.workspace, b.path)}" alt="">
        <div class="label"><span class="muted">${b.filename}</span></div>`;
      t.addEventListener('click', () => openDrawer({
        kind: 'image', title: b.filename,
        url: api.media(state.workspace, b.path),
        meta: { scope: data.scope_id, subject: data.subject_id, size: fmtBytes(b.size) },
      }));
      grid.appendChild(t);
    }
  }
  if (data.voices.length) {
    for (const v of data.voices.slice(0, 3)) {
      const t = document.createElement('div');
      t.className = 'tile';
      t.innerHTML = `
        <div class="badge muted">${ZH.badge_voice}</div>
        <div class="label"><span class="muted">${v.filename}</span><br>${fmtBytes(v.size)}</div>
        <audio controls preload="none" src="${api.media(state.workspace, v.path)}"></audio>`;
      t.addEventListener('click', () => openDrawer({
        kind: 'audio', title: v.filename,
        url: api.media(state.workspace, v.path),
        meta: { scope: data.scope_id, subject: data.subject_id, size: fmtBytes(v.size) },
      }));
      grid.appendChild(t);
    }
  }
  wrap.appendChild(grid);
  main.appendChild(wrap);
}

async function renderAlignmentOverview(main) {
  const globals = state.entityList.filter(e => e.kind === 'global');
  if (!state.scope) state.scope = globals[0]?.id || '';
  populateScopePicker();
  if (!state.scope) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_global_scopes}</div>`;
    return;
  }
  const data = await api.global(state.workspace, state.scope);
  if (!data.ok) {
    main.innerHTML = `<div class="empty-state bad">${data.error}</div>`;
    return;
  }
  const wrap = document.createElement('div');
  const counts = {};
  for (const a of data.tracklet_alignments) {
    counts[a.subject_id] = (counts[a.subject_id] || 0) + 1;
  }
  const max = Math.max(...Object.values(counts), 1);
  const bars = Object.entries(counts).sort().map(([k, v]) =>
    `<div class="bar${k === 'unknown' ? ' unknown' : ''}" style="height:${(v/max*70).toFixed(1)}px" title="${k}: ${v}"><span class="lbl">${k}<br>${v}</span></div>`
  ).join('');
  let videoHtml = '';
  if (data.stitched_video_exists) {
    videoHtml = `
      <div class="section">
        <h3>${ZH.section_stitched_video.replace('{scope}', data.scope_id)}</h3>
        <video controls preload="metadata" src="${api.media(state.workspace, '/workspace/' + state.workspace + '/' + data.stitched_video)}"></video>
      </div>`;
  }
  wrap.innerHTML = `
    <div class="section">
      <h3>${data.scope_id} · status=${data.status} · ${data.assigned_tracklet_count}/${data.tracklet_count} ${ZH.kpi_assigned_tracklet_count}</h3>
      <div class="bar-chart">${bars}</div>
    </div>
    ${videoHtml}
    <div class="section">
      <h3>${ZH.section_per_tracklet}</h3>
      <table class="tracklets" id="alignment-table">
        <thead><tr><th>${ZH.th_tracklet}</th><th>${ZH.th_subject}</th><th>${ZH.th_tier}</th><th>${ZH.th_confidence}</th><th>${ZH.th_visual_cues}</th><th>${ZH.th_reason}</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>`;
  main.appendChild(wrap);
  const tbody = wrap.querySelector('tbody');
  for (const a of data.tracklet_alignments) {
    const tr = document.createElement('tr');
    const conf = Number(a.confidence || 0);
    tr.innerHTML = `
      <td>${a.tracklet_id}</td>
      <td>${a.subject_id}</td>
      <td>${a.importance_tier}</td>
      <td><span class="conf-bar" style="width:${(conf*60).toFixed(1)}px"></span> ${conf.toFixed(2)}</td>
      <td>${(a.visual_cues || []).join(', ')}</td>
      <td class="muted">${a.reason || ''}</td>`;
    tr.addEventListener('click', () => {
      const shotId = a.tracklet_id.replace(/_T\d+$/, '');
      openDrawerForTracklet(shotId, a.tracklet_id);
    });
    tbody.appendChild(tr);
  }
}

async function openDrawerForTracklet(shotId, tid) {
  state.shotId = shotId;
  state.trackletId = tid;
  setUrl();
  const data = await api.tracklet(state.workspace, shotId, tid);
  openDrawer({
    kind: 'tracklet',
    title: `${tid} @ ${shotId}`,
    payload: data,
  });
}

function renderMaterializationTable(main) {
  const rows = state.entityList.filter(e => e.kind === 'character');
  const table = document.createElement('table');
  table.className = 'tracklets';
  table.innerHTML = `
    <thead><tr>
      <th>${ZH.th_scope_subject}</th><th>${ZH.th_tracklets}</th><th>${ZH.th_face}</th><th>${ZH.th_body}</th><th>${ZH.th_rgba}</th><th>${ZH.th_voice}</th><th>${ZH.th_missing}</th>
    </tr></thead><tbody></tbody>`;
  const tb = table.querySelector('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.id}</td><td>${r.tracklet_count}</td><td>${r.face_copied}</td>
      <td>${r.body_copied}</td><td>${r.rgba_copied ?? '–'}</td><td>${r.voice_copied}</td>
      <td>${r.missing_files ?? '–'}</td>`;
    tr.addEventListener('click', () => {
      const [s, sj] = r.id.split(':');
      state.stage = 'characters';
      state.scope = s; state.subject = sj;
      populateScopePicker();
      setUrl();
      loadStagePicker().then(renderMain);
    });
    tb.appendChild(tr);
  }
  main.appendChild(table);
}

function renderAssetBankTable(main) {
  const rows = state.entityList.filter(e => e.kind === 'chr');
  const table = document.createElement('table');
  table.className = 'tracklets';
  table.innerHTML = `
    <thead><tr>
      <th>${ZH.th_chr}</th><th>${ZH.th_tracklets}</th><th>${ZH.th_shots}</th><th>${ZH.th_confidence}</th>
    </tr></thead><tbody></tbody>`;
  const tb = table.querySelector('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.id}</td><td>${r.tracklet_count}</td><td>${r.shot_count ?? '–'}</td>
      <td>${Number(r.identity_confidence || 0).toFixed(3)}</td>`;
    tb.appendChild(tr);
  }
  main.appendChild(table);
}

function renderTrackletList(main) {
  const rows = state.entityList.filter(e => e.kind === 'shot');
  const table = document.createElement('table');
  table.className = 'tracklets';
  table.innerHTML = `
    <thead><tr><th>${ZH.th_shot}</th><th>${ZH.th_tracklets}</th></tr></thead><tbody></tbody>`;
  const tb = table.querySelector('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${r.id}</td><td>${r.tracklet_count ?? '–'}</td>`;
    tb.appendChild(tr);
  }
  main.appendChild(table);
}

function renderVoiceOverview(main) {
  const rows = state.entityList.filter(e => e.kind === 'voice_asset' && (!state.scope || e.scope === state.scope));
  const valid = rows.filter(e => e.status === 'valid').length;
  main.innerHTML = `<div class="section"><h3>语音 / ASR 资产</h3>
    <div class="muted">${rows.length} 条 · 可用 ${valid} · 已拒绝 ${rows.length - valid}</div></div>`;
  const table = document.createElement('table');
  table.className = 'tracklets';
  table.innerHTML = `<thead><tr><th>状态</th><th>范围</th><th>说话人</th><th>ASR</th><th>文本</th><th>置信度</th></tr></thead><tbody></tbody>`;
  for (const item of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="${item.status === 'valid' ? 'ok' : 'bad'}">${esc(item.status)}</td>
      <td>${esc(item.scope)}</td><td>${esc(item.speaker_id)}</td><td>${esc(item.asr_id)}</td>
      <td>${esc(item.text)}</td><td>${esc(item.confidence || '–')}</td>`;
    tr.addEventListener('click', () => openDrawer({ kind: 'voice_result', title: `${item.asr_id} · ${item.speaker_id}`, payload: item }));
    table.querySelector('tbody').appendChild(tr);
  }
  main.appendChild(table);
}

function renderMediaResults(main, stage) {
  let rows = state.entityList.filter(e => (!state.scope || e.scope === state.scope));
  const isShot = stage === 'shots';
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${isShot ? 'Shot 切片与索引' : 'Segment 成片'}</h3>
    <div class="muted">${rows.length} 条结果，点击查看视频、caption、对话和完整 JSON。</div>`;
  main.appendChild(overview);
  if (isShot) {
    const table = document.createElement('table');
    table.className = 'tracklets';
    table.innerHTML = `<thead><tr><th>Shot</th><th>范围</th><th>时长</th><th>可见主体</th><th>Caption 摘要</th></tr></thead><tbody></tbody>`;
    for (const item of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${esc(item.id)}</td><td>${esc(item.scope)}</td><td>${fmtSeconds(item.duration_sec)}</td>
        <td>${item.visible_count ?? '–'}</td><td>${esc(item.description || '–')}</td>`;
      tr.addEventListener('click', () => openPipelineDetail(stage, item.id));
      table.querySelector('tbody').appendChild(tr);
    }
    main.appendChild(table);
  } else {
    const grid = document.createElement('div');
    grid.className = 'video-grid';
    for (const item of rows) {
      const tile = document.createElement('div');
      tile.className = 'tile caption-card';
      tile.innerHTML = `<div class="badge">${esc(item.event_phase || 'segment')}</div>
        <video controls preload="metadata" src="${api.media(state.workspace, item.media_path)}"></video>
        <div class="label"><strong>${esc(item.id)}</strong><br><span class="muted">${fmtSeconds(item.duration_sec)} · ${item.shot_count} shots · ${item.dialogue_count} 句对话</span>
        <div class="prose">${esc(item.description)}</div></div>`;
      tile.addEventListener('click', (event) => { if (event.target.tagName !== 'VIDEO') openPipelineDetail(stage, item.id); });
      grid.appendChild(tile);
    }
    main.appendChild(grid);
  }
}

function renderCaptions(main) {
  const rows = state.entityList.filter(e => (!state.scope || e.scope === state.scope));
  const level = state.stage.replace('captions_', '');
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${level.toUpperCase()} Caption</h3>
    <div class="muted">${rows.length} 条 · 包含叙事描述、场景、主体、时间线、对话/ASR 和完整模型输出。</div>`;
  main.appendChild(overview);
  if (level === 'shot') {
    const table = document.createElement('table');
    table.className = 'tracklets';
    table.innerHTML = `<thead><tr><th>Shot</th><th>范围</th><th>时长</th><th>主体</th><th>对话</th><th>Caption</th></tr></thead><tbody></tbody>`;
    for (const item of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${esc(item.id)}</td><td>${esc(item.scope)}</td><td>${fmtSeconds(item.duration_sec)}</td>
        <td>${item.subject_count}</td><td>${item.dialogue_count}</td><td>${esc(item.description || '–')}</td>`;
      tr.addEventListener('click', () => openPipelineDetail(state.stage, item.id));
      table.querySelector('tbody').appendChild(tr);
    }
    main.appendChild(table);
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'grid';
  for (const item of rows) {
    const tile = document.createElement('div');
    tile.className = 'section caption-card';
    tile.style.cursor = 'pointer';
    tile.innerHTML = `<h3>${esc(item.id)} · ${esc(item.confidence || '–')}</h3>
      <div class="prose">${esc(item.description || item.scene_description || '–')}</div>
      <div class="caption-meta"><span>${fmtSeconds(item.duration_sec)}</span><span>${item.subject_count} 主体</span>
      <span>${item.dialogue_count} 句对话</span>${item.segment_count ? `<span>${item.segment_count} segments</span>` : ''}</div>`;
    tile.addEventListener('click', () => openPipelineDetail(state.stage, item.id));
    grid.appendChild(tile);
  }
  main.appendChild(grid);
}

async function openPipelineDetail(stage, id) {
  const data = await api.detail(state.workspace, stage, id);
  openDrawer({ kind: 'pipeline_result', title: id, payload: data });
}

async function renderArtifacts(main) {
  const groups = state.entityList.filter(e => e.kind === 'artifact_group');
  const header = document.createElement('div');
  header.className = 'section';
  header.innerHTML = `<h3>全部产物文件</h3><div class="muted">${groups.reduce((n, g) => n + g.file_count, 0)} 个文件；JSON / JSONL / 日志可直接预览，媒体可播放。</div>`;
  const chips = document.createElement('div');
  chips.className = 'caption-meta';
  chips.innerHTML = `<span data-prefix="">全部</span>` + groups.map(g => `<span data-prefix="${esc(g.id === '(root)' ? '' : g.id)}">${esc(g.id)} · ${g.file_count}</span>`).join('');
  chips.querySelectorAll('span').forEach(s => { s.style.cursor = 'pointer'; s.addEventListener('click', () => loadArtifactTable(main, s.dataset.prefix || '')); });
  header.appendChild(chips);
  main.appendChild(header);
  await loadArtifactTable(main, '');
}

async function loadArtifactTable(main, prefix) {
  main.querySelector('#artifact-browser')?.remove();
  const data = await api.artifacts(state.workspace, prefix);
  const wrap = document.createElement('div');
  wrap.id = 'artifact-browser';
  const toolbar = document.createElement('div');
  toolbar.className = 'toolbar';
  toolbar.innerHTML = `<input type="text" placeholder="按路径或文件名搜索…"><span class="muted">${data.count || 0} files</span>`;
  wrap.appendChild(toolbar);
  const table = document.createElement('table');
  table.className = 'tracklets';
  table.innerHTML = `<thead><tr><th>类型</th><th>路径</th><th>大小</th></tr></thead><tbody></tbody>`;
  wrap.appendChild(table);
  const renderRows = (query = '') => {
    const items = (data.items || []).filter(x => x.path.toLowerCase().includes(query.toLowerCase()));
    const body = table.querySelector('tbody'); body.innerHTML = '';
    for (const item of items) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><span class="file-kind">${esc(item.kind)}</span></td><td>${esc(item.path)}</td><td>${fmtBytes(item.size)}</td>`;
      tr.addEventListener('click', () => openArtifact(item));
      body.appendChild(tr);
    }
    toolbar.querySelector('span').textContent = `${items.length} / ${data.count || 0} files`;
  };
  toolbar.querySelector('input').addEventListener('input', e => renderRows(e.target.value));
  renderRows();
  main.appendChild(wrap);
}

async function openArtifact(item) {
  if (['image', 'video', 'audio'].includes(item.kind)) {
    openDrawer({ kind: item.kind, title: item.path, url: api.media(state.workspace, item.path), meta: { size: fmtBytes(item.size) } });
    return;
  }
  const data = await api.artifact(state.workspace, item.path);
  openDrawer({ kind: 'artifact', title: item.path, payload: data });
}

// ---------------------------------------------------------------------------
// New render functions: global_clips, scene, key_object, i2i_bundles
// ---------------------------------------------------------------------------

async function renderGlobalClips(main) {
  main.innerHTML = '';
  // /api/entities?stage=global_clips 返回 list [{id,filename,path,size}]
  const entities = state.entityList.filter(e => e.kind === 'global_clip');
  if (!entities.length) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_entities}</div>`;
    return;
  }
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${ZH.section_global_clips} · ${ZH.section_overview}</h3>
    <div class="muted">${entities.length} ${ZH.kpi_global_clip_count}</div>`;
  main.appendChild(overview);

  const grid = document.createElement('div');
  grid.className = 'video-grid';
  for (const c of entities) {
    const t = document.createElement('div');
    t.className = 'tile';
    t.innerHTML = `
      <div class="badge">${ZH.section_global_clips}</div>
      <video controls preload="metadata" src="${api.media(state.workspace, c.path)}"></video>
      <div class="label">
        <strong>${c.id}</strong><br>
        <span class="muted">${c.filename} · ${fmtBytes(c.size)}</span>
      </div>`;
    t.addEventListener('click', () => openDrawer({
      kind: 'video',
      title: c.id,
      url: api.media(state.workspace, c.path),
      meta: { size: fmtBytes(c.size), filename: c.filename },
    }));
    grid.appendChild(t);
  }
  main.appendChild(grid);
}

async function renderScene(main) {
  main.innerHTML = '';
  const isOpt = state.stage === 'scene_opt';
  const entities = state.entityList.filter(e => e.kind === 'scene');
  if (!entities.length) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_entities}</div>`;
    return;
  }
  // 顶部概述表（列出每个 global 的 scene_index 概况）
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${ZH.section_scene}${isOpt ? ' (优化)' : ''}</h3>`;
  const tbl = document.createElement('table');
  tbl.className = 'tracklets';
  tbl.innerHTML = `<thead><tr>
    <th>${ZH.th_scene}</th><th>${ZH.th_candidates}</th><th>${ZH.th_size}</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.querySelector('tbody');
  for (const s of entities) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${s.id}</td><td>${s.scene_index_exists ? '✓' : '–'}</td><td>${fmtBytes(0)}</td>`;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => openSceneDetail(s, isOpt));
    tb.appendChild(tr);
  }
  overview.appendChild(tbl);
  main.appendChild(overview);

  // 三个 global 一行预览：每个 scope 拉一次详情 + 列出 scene_001..N 的 topk_preview
  const grid = document.createElement('div');
  grid.className = 'grid';
  for (const s of entities) {
    const data = await api.sceneScope(state.workspace, s.id, isOpt);
    if (!data || !data.ok) continue;
    const sceneCount = (data.scenes || []).length || '?';
    const cover = document.createElement('div');
    cover.className = 'tile';
    cover.innerHTML = `
      <div class="badge">${s.id}</div>
      <div class="label"><strong>${s.id}</strong><br>
        <span class="muted">${sceneCount} scenes · ${data.candidate_count || 0} ${ZH.th_candidates}</span></div>`;
    cover.addEventListener('click', () => openSceneDetail(s, isOpt));
    grid.appendChild(cover);
  }
  main.appendChild(grid);
}

async function openSceneDetail(sceneEntity, isOpt) {
  const data = await api.sceneScope(state.workspace, sceneEntity.id, isOpt);
  if (!data || !data.ok) { openDrawer({ kind: 'text', title: sceneEntity.id, meta: { error: data && data.error } }); return; }
  // 把每个 scene_001..N 列出；预览图 topk_preview.jpg 在 scene_dir/scene_NNN/
  const sceneDir = data.scene_dir;
  const sceneNames = await _listSceneSubdirs(sceneDir);
  openDrawer({
    kind: 'scene',
    title: `${sceneEntity.id} · ${isOpt ? '优化' : '原始'}`,
    payload: { ...data, scene_dir: sceneDir, scene_names: sceneNames },
  });
}

async function _listSceneSubdirs(sceneDir) {
  // 客户端不直接列 fs；通过 candidate_scores 的 scenes 字段推断 scene_001..N
  // 该函数实际不再需要——scene list 已从 server payload 提供
  return [];
}

async function renderKeyObject(main) {
  main.innerHTML = '';
  const entities = state.entityList.filter(e => e.kind === 'object_observation');
  if (!entities.length) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_entities}</div>`;
    return;
  }
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${ZH.section_key_object} · ${ZH.section_overview}</h3>
    <div class="muted">${entities.length} ${ZH.th_observations}</div>`;
  main.appendChild(overview);

  const tbl = document.createElement('table');
  tbl.className = 'tracklets';
  tbl.innerHTML = `<thead><tr>
    <th>${ZH.th_object}</th><th>${ZH.th_category}</th><th>${ZH.th_shot}</th><th>${ZH.th_frame_count}</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.querySelector('tbody');
  // 按 canonical_name 聚类
  const byName = new Map();
  for (const e of entities) {
    const k = e.canonical_name || e.id;
    if (!byName.has(k)) byName.set(k, []);
    byName.get(k).push(e);
  }
  for (const [name, items] of byName) {
    const sample = items[0];
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${sample.id}</td><td>${name}</td><td>${sample.shot_id || '–'}</td><td>${items.length}</td>`;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => openKeyObjectDetail(sample.canonical_name || sample.id));
    tb.appendChild(tr);
  }
  overview.appendChild(tbl);

  // 拉一次 payload（scope 来自 picker；如果未指定则传空字符串表示"全部"）
  // 但当前 picker 总是有值（global_0040/0041/0042），dataset 又没写 scope，所以
  // 后端会返回全部 observations + 全部 crops。
  const targetScope = state.scope || '';
  const data = await api.keyObjectScope(state.workspace, targetScope);
  if (data && data.ok && data.observations.length) {
    const sec = document.createElement('div');
    sec.className = 'section';
    sec.innerHTML = `<h3>${ZH.keyobject_section_observations} (${data.observations.length})</h3>`;
    const grid = document.createElement('div');
    grid.className = 'crop-grid';
    // group observations by canonical_name -> cover image (crops/<candidate_id>_*.jpg)
    const byCan = new Map();
    for (const o of data.observations) {
      const k = o.canonical_name || o.candidate_id;
      if (!byCan.has(k)) byCan.set(k, []);
      byCan.get(k).push(o);
    }
    for (const [can, obs] of byCan) {
      const first = obs[0];
      const cid = first.candidate_id || '';
      // 找对应 crop 图 (按 candidate_id 前缀匹配)
      const cropMatch = (data.crops || []).find(c => c.candidate_id === cid) || (data.crops || [])[0];
      const t = document.createElement('div');
      t.className = 'tile';
      t.innerHTML = `
        <div class="badge">${obs.length}</div>
        ${cropMatch ? `<img loading="lazy" src="${api.media(state.workspace, cropMatch.path)}" alt="" onerror="this.style.display='none'">` : '<div style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--muted);">无预览</div>'}
        <div class="label"><strong>${can}</strong><br>
          <span class="muted">${first.shot_id || '–'}</span></div>`;
      t.addEventListener('click', () => openKeyObjectDetail(can));
      grid.appendChild(t);
    }
    sec.appendChild(grid);
    main.appendChild(sec);
  }
}

async function openKeyObjectDetail(canonicalName) {
  // 找包含此 canonical_name 的 scope，调一次 /api/key_object
  const scope = state.scope || 'global_0040';
  const data = await api.keyObjectScope(state.workspace, scope);
  if (!data || !data.ok) return;
  const filtered = (data.observations || []).filter(o => (o.canonical_name || '') === canonicalName);
  openDrawer({
    kind: 'key_object',
    title: `${canonicalName} (${filtered.length})`,
    payload: { canonical_name: canonicalName, observations: filtered, crops: data.crops || [] },
  });
}

async function renderI2IBundles(main) {
  main.innerHTML = '';
  const entities = state.entityList.filter(e => e.kind === 'i2i_bundle');
  if (!entities.length) {
    main.innerHTML = `<div class="empty-state">${ZH.empty_no_entities}</div>`;
    return;
  }
  // 按 task_type 分组
  const byType = { scene: [], object: [] };
  for (const e of entities) {
    (byType[e.task_type] = byType[e.task_type] || []).push(e);
  }
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>${ZH.section_i2i_bundles} · ${ZH.section_overview}</h3>
    <div class="muted">${entities.length} 总数 · ${byType.scene?.length || 0} scene · ${byType.object?.length || 0} object</div>`;
  main.appendChild(overview);

  const tbl = document.createElement('table');
  tbl.className = 'tracklets';
  tbl.innerHTML = `<thead><tr>
    <th>${ZH.th_bundle_id}</th><th>${ZH.th_category}</th><th>${ZH.th_size}</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.querySelector('tbody');
  for (const b of entities) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${b.id}</td><td>${b.task_type || '–'}</td><td>${fmtBytes(b.size)}</td>`;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => openI2IBundleDetail(b.id));
    tb.appendChild(tr);
  }
  overview.appendChild(tbl);
  main.appendChild(overview);
}

async function openI2IBundleDetail(bundleId) {
  const data = await api.i2iBundle(state.workspace, bundleId);
  if (!data || !data.ok) { openDrawer({ kind: 'text', title: bundleId, meta: { error: data && data.error } }); return; }
  openDrawer({
    kind: 'i2i_bundle',
    title: bundleId,
    payload: data.bundle,
  });
}

function renderI2ITest(main) {
  main.innerHTML = '';
  const references = state.entityList.filter(e => e.kind === 'i2i_test_reference');
  const results = state.entityList.filter(e => e.kind === 'i2i_test_result');
  const models = [...new Set(results.map(e => e.model))];
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>I2I 模型效果对比</h3>
    <div class="muted">${references.length} 张输入参考图 · ${results.length} 张生成结果 · ${models.length} 个模型：${models.map(esc).join('、')}</div>`;
  main.appendChild(overview);

  if (references.length) {
    const section = document.createElement('div');
    section.className = 'section';
    section.innerHTML = '<h3>输入参考图</h3>';
    const grid = document.createElement('div'); grid.className = 'crop-grid';
    for (const item of references) {
      const tile = document.createElement('div'); tile.className = 'tile';
      tile.innerHTML = `<div class="badge muted">reference</div>
        <img loading="lazy" src="${api.media(state.workspace, item.image_path)}" alt="">
        <div class="label">${esc(item.label)} · ${fmtBytes(item.size)}</div>`;
      tile.addEventListener('click', () => openDrawer({
        kind: 'image', title: item.label,
        url: api.media(state.workspace, item.image_path), meta: { size: fmtBytes(item.size) },
      }));
      grid.appendChild(tile);
    }
    section.appendChild(grid); main.appendChild(section);
  }

  const section = document.createElement('div');
  section.className = 'section'; section.innerHTML = '<h3>生成结果</h3>';
  const grid = document.createElement('div'); grid.className = 'grid';
  for (const item of results) {
    const tile = document.createElement('div'); tile.className = 'tile';
    tile.innerHTML = `<div class="badge">${esc(item.model)}</div>
      <img loading="lazy" src="${api.media(state.workspace, item.image_path)}" alt="">
      <div class="label"><strong>${esc(item.part || item.label)}</strong><br>
        <span class="muted">${esc(item.model_version || item.model)} · ${item.total_tokens ?? '–'} tokens · ${fmtBytes(item.size)}</span></div>`;
    tile.addEventListener('click', () => openI2ITestDetail(item.id));
    grid.appendChild(tile);
  }
  section.appendChild(grid); main.appendChild(section);
}

async function openI2ITestDetail(id) {
  const data = await api.detail(state.workspace, 'i2i_test', id);
  openDrawer({ kind: 'i2i_test', title: id, payload: data });
}

function renderTTSTest(main) {
  main.innerHTML = '';
  const allRows = state.entityList.filter(e => e.kind === 'tts_test_result');
  const rows = allRows.filter(e => !state.scope || e.scope === state.scope);
  const models = [...new Set(allRows.map(e => e.model))];
  const overview = document.createElement('div');
  overview.className = 'section';
  overview.innerHTML = `<h3>TTS 模型效果对比</h3>
    <div class="muted">当前范围 ${esc(state.scope || '全部')} · ${rows.length} 组 · ${models.length} 个模型：${models.map(esc).join('、')}</div>`;
  main.appendChild(overview);
  const grid = document.createElement('div'); grid.className = 'grid';
  for (const item of rows) {
    const card = document.createElement('div'); card.className = 'section';
    card.innerHTML = `<h3>${esc(item.model)} · ${esc(item.scope)} / ${esc(item.speaker_id)}</h3>
      <div class="prose"><strong>参考文本：</strong>${esc(item.reference_text || '–')}</div>
      ${item.reference_path ? `<div class="audio-result"><span class="muted">reference_merged.wav</span><audio controls preload="none" src="${api.media(state.workspace, item.reference_path)}"></audio></div>` : ''}
      <div class="prose" style="margin-top:8px"><strong>合成文本：</strong>${esc(item.synthesis_text || '–')}</div>`;
    for (const audio of (item.output_paths || [])) {
      const row = document.createElement('div'); row.className = 'audio-result';
      row.innerHTML = `<span class="muted">${esc(audio.filename)} · ${fmtBytes(audio.size)}</span>
        <audio controls preload="none" src="${api.media(state.workspace, audio.path)}"></audio>`;
      card.appendChild(row);
    }
    const actions = document.createElement('div'); actions.className = 'json-actions';
    const button = document.createElement('button'); button.type = 'button'; button.textContent = '查看参数 JSON';
    button.addEventListener('click', () => openTTSTestDetail(item.id));
    actions.appendChild(button); card.appendChild(actions); grid.appendChild(card);
  }
  main.appendChild(grid);
}

async function openTTSTestDetail(id) {
  const data = await api.detail(state.workspace, 'tts_test', id);
  openDrawer({ kind: 'tts_test', title: id, payload: data });
}

function openDrawer({ kind, title, url, payload, meta }) {
  $('drawer-body').innerHTML = '';
  const wrap = document.createElement('div');
  const h = document.createElement('h2');
  h.textContent = title;
  wrap.appendChild(h);
  if (kind === 'image') {
    const img = document.createElement('img');
    img.src = url;
    img.style.maxWidth = '100%';
    img.style.borderRadius = '6px';
    wrap.appendChild(img);
  } else if (kind === 'audio') {
    const audio = document.createElement('audio');
    audio.src = url;
    audio.controls = true;
    audio.style.width = '100%';
    wrap.appendChild(audio);
  } else if (kind === 'video') {
    const v = document.createElement('video');
    v.src = url;
    v.controls = true;
    v.preload = 'metadata';
    v.style.width = '100%';
    wrap.appendChild(v);
  } else if (kind === 'i2i_test' && payload && payload.ok) {
    const image = document.createElement('img');
    image.src = api.media(state.workspace, payload.image_path);
    image.style.maxWidth = '100%'; image.style.borderRadius = '6px';
    wrap.appendChild(image);
    const request = payload.raw?.request || {};
    const response = payload.raw?.response || {};
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>模型</dt><dd>${esc(response.model_version || payload.model || '–')}</dd>
      <dt>任务 / 角色</dt><dd>${esc(request.task || '–')} / ${esc(request.character_label || '–')}</dd>
      <dt>Token</dt><dd>${response.usage?.totalTokenCount ?? '–'}</dd>
      <dt>结束原因</dt><dd>${esc((response.finish_reasons || []).join(', ') || '–')}</dd>`;
    wrap.appendChild(dl);
    if (request.prompt) {
      const prompt = document.createElement('div'); prompt.className = 'section';
      prompt.innerHTML = '<h3>Prompt</h3>';
      const prose = document.createElement('div'); prose.className = 'prose'; prose.textContent = request.prompt;
      prompt.appendChild(prose); wrap.appendChild(prompt);
    }
    if ((payload.references || []).length) {
      const section = document.createElement('div'); section.className = 'section';
      section.innerHTML = '<h3>输入参考图</h3>';
      const grid = document.createElement('div'); grid.className = 'crop-grid';
      for (const ref of payload.references) {
        const tile = document.createElement('div'); tile.className = 'tile';
        tile.innerHTML = `<img loading="lazy" src="${api.media(state.workspace, ref.path)}" alt="">
          <div class="label muted">${esc(ref.filename)}</div>`;
        grid.appendChild(tile);
      }
      section.appendChild(grid); wrap.appendChild(section);
    }
    const details = document.createElement('details'); details.className = 'json-viewer';
    details.innerHTML = '<summary>完整 JSON</summary>';
    const pre = document.createElement('pre'); pre.className = 'raw';
    pre.textContent = JSON.stringify(payload.raw, null, 2);
    details.appendChild(pre); addCopyButton(details, pre); wrap.appendChild(details);
  } else if (kind === 'tts_test' && payload && payload.ok) {
    const raw = payload.raw || {};
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>模型</dt><dd>${esc(raw.model || payload.model || '–')}</dd>
      <dt>参考文本</dt><dd>${esc(raw.reference_text || '–')}</dd>
      <dt>合成文本</dt><dd>${esc(raw.synthesis_text || '–')}</dd>`;
    wrap.appendChild(dl);
    if (payload.reference_path) {
      const row = document.createElement('div'); row.className = 'audio-result';
      row.innerHTML = `<strong>参考音频</strong><audio controls preload="none" src="${api.media(state.workspace, payload.reference_path)}"></audio>`;
      wrap.appendChild(row);
    }
    for (const item of (payload.outputs || [])) {
      const row = document.createElement('div'); row.className = 'audio-result';
      row.innerHTML = `<strong>${esc(item.filename)}</strong><span class="muted"> · ${fmtBytes(item.size)}</span>
        <audio controls preload="none" src="${api.media(state.workspace, item.path)}"></audio>`;
      wrap.appendChild(row);
    }
    const details = document.createElement('details'); details.className = 'json-viewer';
    details.open = true; details.innerHTML = '<summary>参数 JSON</summary>';
    const pre = document.createElement('pre'); pre.className = 'raw';
    pre.textContent = JSON.stringify(raw, null, 2);
    details.appendChild(pre); addCopyButton(details, pre); wrap.appendChild(details);
  } else if (kind === 'voice_result' && payload) {
    if (payload.media_path) {
      const audio = document.createElement('audio');
      audio.src = api.media(state.workspace, payload.media_path);
      audio.controls = true;
      audio.preload = 'metadata';
      audio.style.width = '100%';
      wrap.appendChild(audio);
    }
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>状态</dt><dd class="${payload.status === 'valid' ? 'ok' : 'bad'}">${esc(payload.status)}</dd>
      <dt>范围 / 说话人</dt><dd>${esc(payload.scope)} / ${esc(payload.speaker_id)}</dd>
      <dt>ASR 文本</dt><dd>${esc(payload.text)}</dd>
      <dt>归因证据</dt><dd>${esc(payload.evidence || '–')}</dd>
      <dt>拒绝原因</dt><dd>${esc((payload.rejection_reasons || []).join(', ') || '–')}</dd>`;
    wrap.appendChild(dl);
    const pre = document.createElement('pre');
    pre.className = 'raw'; pre.textContent = JSON.stringify(payload.signal_stats || {}, null, 2);
    wrap.appendChild(pre); addCopyButton(wrap, pre);
  } else if (kind === 'pipeline_result' && payload && payload.ok) {
    if (payload.media_path) {
      const video = document.createElement('video');
      video.src = api.media(state.workspace, payload.media_path);
      video.controls = true; video.preload = 'metadata'; video.style.width = '100%';
      wrap.appendChild(video);
    }
    const record = payload.record || {};
    const caption = payload.caption || {
      description: record.description || '', scene_description: record.scene_description || '',
      overall_soundscape: record.segment_caption?.overall_soundscape || '',
      non_diegetic_music: record.segment_caption?.non_diegetic_music || '',
      subjects: record.semantic_role_observations || [], dialogue: record.dialogue || [],
      events: [], segments: [],
    };
    const addProse = (heading, text) => {
      if (!text) return;
      const sec = document.createElement('div'); sec.className = 'section';
      const h3 = document.createElement('h3'); h3.textContent = heading;
      const p = document.createElement('div'); p.className = 'prose'; p.textContent = text;
      sec.append(h3, p); wrap.appendChild(sec);
    };
    addProse('Caption / 叙事描述', caption.description || record.description);
    addProse('场景描述', caption.scene_description || record.scene_description);
    addProse('声音环境', caption.overall_soundscape);
    addProse('非叙事音乐', caption.non_diegetic_music);
    if ((caption.subjects || []).length) {
      const sec = document.createElement('div'); sec.className = 'section'; sec.innerHTML = '<h3>主体</h3>';
      const table = document.createElement('table'); table.className = 'tracklets';
      table.innerHTML = '<thead><tr><th>ID</th><th>角色 / 层级</th><th>描述</th></tr></thead><tbody></tbody>';
      for (const subject of caption.subjects) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${esc(subject.subject_id || subject.id || '–')}</td><td>${esc(subject.role_label || subject.importance_tier || '–')}</td>
          <td>${esc(subject.identity_description || subject.description || (subject.narrative_functions || []).join(', ') || '–')}</td>`;
        table.querySelector('tbody').appendChild(tr);
      }
      sec.appendChild(table); wrap.appendChild(sec);
    }
    if ((caption.dialogue || []).length) {
      const sec = document.createElement('div'); sec.className = 'section'; sec.innerHTML = `<h3>对话 / ASR (${caption.dialogue.length})</h3>`;
      const table = document.createElement('table'); table.className = 'tracklets';
      table.innerHTML = '<thead><tr><th>时间</th><th>说话人</th><th>文本</th><th>置信度</th></tr></thead><tbody></tbody>';
      for (const line of caption.dialogue) {
        const time = line.time_range_sec || line.segment_time_range_sec || [];
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${time.length ? `${Number(time[0]).toFixed(2)}–${Number(time[1]).toFixed(2)}` : '–'}</td>
          <td>${esc(line.speaker_id || '–')}</td><td>${esc(line.text || '–')}</td><td>${esc(line.attribution_confidence || '–')}</td>`;
        table.querySelector('tbody').appendChild(tr);
      }
      sec.appendChild(table); wrap.appendChild(sec);
    }
    const timeline = (caption.segments || []).length ? caption.segments : (caption.events || []);
    if (timeline.length) {
      const sec = document.createElement('div'); sec.className = 'section'; sec.innerHTML = `<h3>事件 / Segment 时间线 (${timeline.length})</h3>`;
      for (const event of timeline) {
        const p = document.createElement('p'); p.className = 'prose';
        const range = event.time_range_sec || [];
        p.textContent = `${range.length ? `[${range[0]}–${range[1]}] ` : ''}${event.event || event.description || event.segment_caption?.segment_description || ''}`;
        sec.appendChild(p);
      }
      wrap.appendChild(sec);
    }
    const details = document.createElement('details'); details.className = 'json-viewer';
    details.innerHTML = '<summary>完整 JSON</summary>';
    const pre = document.createElement('pre'); pre.className = 'raw';
    pre.textContent = JSON.stringify(payload.raw || payload.record || payload, null, 2);
    details.appendChild(pre); addCopyButton(details, pre); wrap.appendChild(details);
  } else if (kind === 'artifact' && payload) {
    const meta = document.createElement('dl');
    meta.innerHTML = `<dt>文件</dt><dd>${esc(payload.path || '')}</dd><dt>大小</dt><dd>${fmtBytes(payload.size)}</dd>
      ${payload.row_count != null ? `<dt>JSONL 行数</dt><dd>${payload.row_count}${payload.truncated ? ' (仅预览前 100 行)' : ''}</dd>` : ''}`;
    wrap.appendChild(meta);
    const pre = document.createElement('pre'); pre.className = 'raw';
    if (payload.kind === 'json') pre.textContent = JSON.stringify(payload.payload, null, 2);
    else if (payload.kind === 'jsonl') pre.textContent = (payload.rows || []).map(x => JSON.stringify(x, null, 2)).join('\n');
    else pre.textContent = payload.text || JSON.stringify(payload, null, 2);
    wrap.appendChild(pre);
    if (payload.kind === 'json' || payload.kind === 'jsonl') addCopyButton(wrap, pre);
  } else if (kind === 'scene' && payload) {
    // 列出 scenes[] (scene_001..N 的元数据) + 全量 JSON viewer
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>${ZH.section_scene}</dt><dd>${payload.scope_id || ''}</dd>
      <dt>候选数</dt><dd>${payload.candidate_count || 0}</dd>
      <dt>目录</dt><dd class="muted">${payload.scene_dir || ''}</dd>`;
    wrap.appendChild(dl);
    // 列出每个 scene (topk_preview.jpg)
    const grid = document.createElement('div');
    grid.className = 'grid';
    for (const s of (payload.scenes || []).slice(0, 12)) {
      const previewName = s.preview_path || `${payload.scene_dir}/${s.scene_id || s.id}/topk_preview.jpg`;
      const t = document.createElement('div');
      t.className = 'tile';
      t.innerHTML = `<img loading="lazy" src="${api.media(state.workspace, previewName)}" onerror="this.style.display='none'">
        <div class="label"><strong>${s.scene_id || s.id}</strong><br>
          <span class="muted">${fmtSeconds(s.start_sec)}–${fmtSeconds(s.end_sec)}</span></div>`;
      grid.appendChild(t);
    }
    if (grid.children.length) wrap.appendChild(grid);
    // JSON viewer
    const det = document.createElement('details');
    det.className = 'json-viewer';
    det.innerHTML = `<summary>scene_index.json</summary>`;
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(payload.scene_index || {}, null, 2);
    det.appendChild(pre); addCopyButton(det, pre);
    wrap.appendChild(det);
  } else if (kind === 'key_object' && payload) {
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>${ZH.th_object}</dt><dd>${payload.canonical_name || ''}</dd>
      <dt>观测数</dt><dd>${(payload.observations || []).length}</dd>`;
    wrap.appendChild(dl);
    const tbl = document.createElement('table');
    tbl.className = 'tracklets';
    tbl.innerHTML = `<thead><tr>
      <th>${ZH.th_observations}</th><th>${ZH.th_shot}</th><th>时间(s)</th>
    </tr></thead><tbody></tbody>`;
    const tb = tbl.querySelector('tbody');
    for (const o of (payload.observations || []).slice(0, 50)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${o.candidate_id}</td><td>${o.shot_id || '–'}</td><td>${fmtSeconds(o.timestamp_sec)}</td>`;
      tb.appendChild(tr);
    }
    wrap.appendChild(tbl);
    // crops
    const crops = payload.crops || [];
    if (crops.length) {
      const grid = document.createElement('div');
      grid.className = 'crop-grid';
      for (const c of crops.slice(0, 24)) {
        const t = document.createElement('div');
        t.className = 'tile';
        t.innerHTML = `<img loading="lazy" src="${api.media(state.workspace, c.path)}">
          <div class="label muted">${c.filename}</div>`;
        grid.appendChild(t);
      }
      wrap.appendChild(grid);
    }
  } else if (kind === 'i2i_bundle' && payload) {
    const dl = document.createElement('dl');
    dl.innerHTML = `<dt>类型</dt><dd>${payload.task_type || ''}</dd>
      <dt>目标</dt><dd>${(payload.target && (payload.target.composite_key || payload.target.segment_composite_key)) || '–'}</dd>
      <dt>shot 数</dt><dd>${(payload.target && payload.target.shot_ids || []).length}</dd>`;
    wrap.appendChild(dl);
    // 全 JSON viewer（折叠）
    const det = document.createElement('details');
    det.className = 'json-viewer';
    det.open = false;
    det.innerHTML = `<summary>完整 JSON (${fmtBytes(JSON.stringify(payload).length)})</summary>`;
    const pre = document.createElement('pre');
    pre.style.maxHeight = '60vh';
    pre.style.overflow = 'auto';
    pre.textContent = JSON.stringify(payload, null, 2);
    det.appendChild(pre); addCopyButton(det, pre);
    wrap.appendChild(det);
    // references 部分
    if (payload.references && Array.isArray(payload.references.items)) {
      const grid = document.createElement('div');
      grid.className = 'crop-grid';
      for (const it of payload.references.items) {
        if (!it.path && !it.preview_path) continue;
        const p = it.path || it.preview_path;
        const t = document.createElement('div');
        t.className = 'tile';
        t.innerHTML = `<img loading="lazy" src="${api.media(state.workspace, p)}">
          <div class="label muted">${it.label || it.id || ''}</div>`;
        grid.appendChild(t);
      }
      if (grid.children.length) wrap.appendChild(grid);
    }
  } else if (kind === 'text') {
    wrap.appendChild(document.createTextNode((meta && meta.error) || ZH.empty_no_data));
  } else if (kind === 'tracklet' && payload && payload.ok) {
    if (payload.overlay_video_exists) {
      const v = document.createElement('video');
      v.src = api.media(state.workspace,
        '/' + state.workspace + '/' + payload.overlay_video_rel);
      v.controls = true;
      v.preload = 'metadata';
      v.style.width = '100%';
      wrap.appendChild(v);
    }
    const dl = document.createElement('dl');
    dl.innerHTML = `
      <dt>${ZH.drawer_shot}</dt><dd>${payload.shot_id} · ${payload.shot_dir}</dd>
      <dt>${ZH.drawer_tracklet}</dt><dd>${payload.tracklet_id}</dd>
      <dt>${ZH.drawer_time_range}</dt><dd>${payload.start_time.toFixed(2)} – ${payload.end_time.toFixed(2)} s</dd>
      <dt>${ZH.drawer_observations}</dt><dd>${payload.observation_count}</dd>
      <dt>${ZH.drawer_face_gallery}</dt><dd>${payload.face_count} (best q ${(payload.best_face_quality ?? 0).toFixed(3)})</dd>
      <dt>${ZH.drawer_body_gallery}</dt><dd>${payload.body_count} (best q ${(payload.best_body_quality ?? 0).toFixed(3)})</dd>`;
    wrap.appendChild(dl);
    const grid = document.createElement('div');
    grid.className = 'grid';
    grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(120px, 1fr))';
    for (const f of payload.faces.slice(0, 12)) {
      const t = document.createElement('div');
      t.className = 'tile';
      t.innerHTML = `<img src="${api.media(state.workspace, f.path)}"><div class="label muted">${f.filename}</div>`;
      grid.appendChild(t);
    }
    wrap.appendChild(grid);
  } else {
    wrap.appendChild(document.createTextNode(meta?.error || ZH.empty_no_data));
  }
  $('drawer-body').appendChild(wrap);
}

// event wiring
$('ws-picker').addEventListener('change', async (e) => {
  state.workspace = e.target.value;
  state.scope = ''; state.subject = ''; state.shotId = ''; state.trackletId = '';
  await loadStagePicker();
  setUrl();
  renderMain();
});
$('stage-picker').addEventListener('change', async (e) => {
  state.stage = e.target.value;
  state.scope = ''; state.subject = '';
  await loadStagePicker();
  setUrl();
  renderMain();
});
$('scope-picker').addEventListener('change', async (e) => {
  state.scope = e.target.value;
  if (state.stage === 'alignment') { setUrl(); renderMain(); return; }
  state.subject = '';
  populateSubjectPicker();
  setUrl();
  renderMain();
});
$('subject-picker').addEventListener('change', async (e) => {
  state.subject = e.target.value;
  setUrl();
  renderMain();
});
$('reset-btn').addEventListener('click', () => {
  history.replaceState({}, '', window.location.pathname);
  window.location.reload();
});

(async function init() {
  const url = new URL(window.location.href);
  const requestedWorkspace = url.searchParams.get('workspace');
  if (requestedWorkspace && state.workspaceMap.has(requestedWorkspace)) state.workspace = requestedWorkspace;
  if (url.searchParams.get('stage')) state.stage = url.searchParams.get('stage');
  state.scope = url.searchParams.get('scope') || '';
  state.subject = url.searchParams.get('subject') || '';
  state.shotId = url.searchParams.get('shot') || '';
  state.trackletId = url.searchParams.get('tracklet') || '';
  populateWorkspacePicker();
  if (!state.workspace) {
    $('main-body').innerHTML = `<div class="empty-state">${ZH.empty_no_workspace}</div>`;
    return;
  }
  await loadStagePicker();
  populateScopePicker();
  populateSubjectPicker();
  applyPickerVisibility();
  setUrl();
  renderMain();
})();
</script>
</body>
</html>
"""


def render_index(initial_payload: dict) -> str:
    """Inject the workspace list and ZH strings as JSON and return the page source."""
    import json
    payload = {
        "workspaces": initial_payload.get("workspaces", []),
        "default_workspace": initial_payload.get("default_workspace", ""),
        "zh": ZH,
    }
    json_blob = json.dumps(payload, ensure_ascii=False)
    # The template contains literal placeholders __INITIAL_JSON__ and
    # __ZH_*__ which we substitute.  Using literal string replacement avoids
    # any escape handling on the JSON itself.
    out = HTML_TEMPLATE.replace("__INITIAL_JSON__", json_blob)
    # Replace any remaining __ZH_<KEY>__ markers with their dictionary values.
    # Placeholders are uppercased versions of the ZH keys (``__ZH_TITLE__`` -> ZH['title']).
    import re
    def _repl(match: "re.Match[str]") -> str:
        key = match.group(1).lower()
        if key in ZH:
            return str(ZH[key])
        return match.group(0)
    out = re.sub(r"__ZH_([A-Z0-9_]+)__", _repl, out)
    return out
