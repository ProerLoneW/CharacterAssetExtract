# Asset Viewer · 资产展示页面

只读 web UI，用于浏览 `extract_full_pipeline/` 下两条 pipeline 链的全部产物：

- **新链** (`global_align_pipeline/global_tracklet_aligner.py`) — `global_alignment/` + `materialization_summary.json` + `characters/<scope>/<S#>/`
- **老链** (`assets/visual_person/{shot_tracklet_builder, visual_character_registry_builder, character_asset_bank, character_role_aligner}`) — `asset_bank/` + `characters/role_alignment.json` + `voice_by_role_src/valid/<scope>/<S>/`

整合后的 workspace（`movie_full_pipeline_20260916`）额外暴露以下 stage：

| Stage | 数据源 | 内容 |
|---|---|---|
| `Stage 1 · 全局片段` | `global_clips/*.mp4` | 3 个 1080P 源片段 |
| `Stage 1.5 · 场景索引` | `scene/1080P_global_XXXX/{scene_index,candidate_scores,scene_NNN}/` | DINOv3 + YOLOv8n 提取的场景候选与 anchor 图 |
| `Stage 1.5 · 场景索引 (优化)` | `scene_opt_marginal/` | marginal-gain 优化后的场景版本 |
| `Stage 1.5 · 关键目标` | `key_object/grounding/` | GroundingDINO+SAM 提取的物体观测 + crops/frames/masks/overlays/rgba |
| `Stage 1.5 · I2I 束` | `i2i_bundles/{scene,object}/*.json` | image-to-image 参考任务包 (37 个) |
| `I2I 模型效果对比` | `i2i_test/` | 输入参考图、各模型生成 PNG、prompt 与完整响应 JSON |
| `TTS 模型效果对比` | `tts_test/outputs/<model>/<scope>_<speaker>/` | 参考音频、候选音频、参考/合成文本与参数 JSON |

UI 全部使用中文标签；所有 JSON 预览均提供一键复制；默认 workspace = `movie_full_pipeline_20260916`。

当前部署使用 `--only-workspace movie_full_pipeline_20260916`，因此页面与 API
都只会暴露 0916 这一次 pipeline 运行。

## Quick start

```bash
# activate the project env first
conda activate /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annot

# launch on the Gemini platform reverse-proxy port (default)
python -m extract_full_pipeline.asset_viewer.server \
    --host 0.0.0.0 --port 8910 \
    --workspace-root /gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2/workspace \
    --default-workspace movie_full_pipeline_20260916 \
    --only-workspace movie_full_pipeline_20260916 \
    --base-url "/gemini_web/asset_viewer/~58.59.115.26~31443~/$GEMINI_JOB_NAME/"
```

启动后会打印 JSON 状态，包含：

- `internal_url` — `http://0.0.0.0:8910/` (绑定所有接口)
- `public_url` — `http://58.59.115.26:31443/gemini_web/asset_viewer/<...>/<JOB>/`
  (仅在设置了 `--base-url` 时；Gemini 平台反向代理把公网 host:port 转发到容器内的端口)
- `workspace_count`, `workspaces[]` (含 `chain`, `stages`, `scopes`)

## 公网 URL 模板 (Gemini 平台)

集群把 `http://58.59.115.26:31443/<base_url>` 反向代理到容器内的 `http://0.0.0.0:8910/`。
精确的 `<base_url>` 由平台在 job 启动时生成，匹配：

```
/gemini_web/asset_viewer/~58.59.115.26~31443~/<GEMINI_JOB_NAME>/
```

通过 `--base-url` 或环境变量 `ASSET_VIEWER_BASE_URL` 显式传入。无需 token / 密码；
授权在 proxy 边缘完成。

## 阶段选择器

顶部 4 个 `<select>` 下拉驱动所有 fetch：

1. **工作区** — 自动发现的 `movie_full_pipeline_*` 文件夹
2. **阶段** — 根据当前工作区自动列出可用阶段；缺失产物的 stage 自动隐藏
3. **范围** — `global_0040`, `global_0041`, `global_0042` 等
4. **主体** — 选定范围内的 `S1`, `S2`, …

每次变更通过 `history.replaceState` 更新 `?workspace=…&stage=…&scope=…&subject=…`，
顶部 *永久链接* 始终指向当前状态。

只有角色/物化阶段需要主体选择器；TTS 对比可按 global 范围筛选，其余无主体的阶段会自动隐藏主体选择器。

## API 端点

| 端点 | 用途 |
|---|---|
| `GET /` | 单页 HTML dashboard |
| `GET /api/workspaces` | 发现的工作区摘要 |
| `GET /api/stages?workspace=X` | 阶段列表、KPI、概览实体 |
| `GET /api/entities?workspace=X&stage=Y` | 用于填充选择器的实体 |
| `GET /api/character?workspace=X&scope=Y&subject=Z` | 角色 payload (faces/bodies/voices/role_character.json) |
| `GET /api/global?workspace=X&scope=Y` | 单个 global-clip alignment 行 |
| `GET /api/tracklet?workspace=X&shot_id=Y&tracklet_id=Z` | 单个 tracklet 画廊 + overlay mp4 路径 |
| `GET /api/global_clip?workspace=X&scope=Y` | 单个 global mp4 详情 + 元数据 |
| `GET /api/scene?workspace=X&scope=Y&optimized=0\|1` | 场景索引 payload (scene_index + candidate_scores + scenes) |
| `GET /api/key_object?workspace=X&scope=Y` | 物体观测 + crop 列表 |
| `GET /api/i2i_bundle?workspace=X&id=Y` | 单个 i2i 任务包完整 JSON |
| `GET /api/detail?workspace=X&stage=Y&id=Z` | shot / caption / segment / I2I 测试 / TTS 测试详情 |
| `GET /api/artifacts?workspace=X&prefix=Y` | 全产物文件清单 |
| `GET /api/artifact?workspace=X&path=Y` | JSON / JSONL / 文本产物预览 |
| `GET /media/path?workspace=X&abs=…` | workspace-scoped 媒体字节 |

`/media/path` 强制要求 `abs` 解析后落在所选 workspace 根目录内。
外部路径返回 `404` 并记录 WARNING 日志。
这阻止了通过 URL 读取 `/etc/passwd` 等文件。

## 添加新 stage

1. 在 `pipeline_router.py` 的 `STAGE_LABELS` 中添加规范名 (中文 label)
2. 在 `list_stages()` 中添加新产物的检测
3. 在 `list_entities()` 中枚举该阶段的条目
4. 在 `html_template.py` 的 `renderMain()` 中添加一个 `renderXxx` 分支
5. 在 `ZH` dict 中添加对应的中文 label

## 添加新链

`pipeline_router.detect_chain` 返回 `new` / `legacy` / `hybrid` / `none` 之一。
添加新分支：

1. 添加检测逻辑 (查找链专属的标记文件)
2. 在 `list_stages`, `list_entities`, `get_character_payload` 中添加链专属路径约定
3. 前端自动从 `/api/workspaces` 和 `/api/stages` 拾取链类型

## 故障排查

| 症状 | 原因 / 修复 |
|---|---|
| 工作区下拉为空 | `--workspace-root` 错误或不含 `movie_full_pipeline_*` 目录 |
| `public_url` 为 `null` | 未传 `--base-url` 且 `ASSET_VIEWER_BASE_URL` 未设置。设一个 |
| 图片 404 | `role_character.json` 中的路径指向 workspace 外。用 `ls -la <abs path>` 验证 |
| 端口占用 | 选别的 `--port` 并相应更新 proxy 前缀 |
| 公网 URL 502 / 连接拒绝 | 检查 Gemini 平台是否仍在向容器路由；重启 Jupyter kernel 并验证 `http://127.0.0.1:8888` (jupyterlab) 仍可用 |

## 文件布局

```
asset_viewer/
├── __init__.py
├── README.md          ← 本文件
├── server.py          ← CLI + ThreadingHTTPServer + handlers
├── pipeline_router.py ← 链检测 + 阶段查询 (含 5 个新 stage 的 payload 函数)
├── parsers.py         ← JSONL / role_character.json / registry.json schemas
├── html_template.py   ← 单文件 HTML_TEMPLATE 常量 + ZH 中文映射 dict
└── static/            ← 预留 (目前为空)
```

无第三方依赖；服务器仅使用 Python stdlib。
