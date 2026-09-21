# 结果展示网站(result_viewer)

按 [docs/03_结果展示网站设计.md](../docs/03_结果展示网站设计.md) 重新实现的结果展示网站。
现有 `asset_viewer/` 保持不动;本目录是独立的新实现。

## 架构(对应文档三 §2/§10)

```text
result_viewer/
├── viewer_config.json        # 运行时配置:端口/文档路径/允许的运行/产物绑定/路径重写
├── backend/                  # Python 读取服务(FastAPI + Uvicorn)
│   ├── document_parser.py    #   PipelineDoc 1.0 解析 + 严格校验(带行号错误)
│   ├── document_store.py     #   快照 + 热加载(2s 轮询/SHA-256/原子替换/坏文档回退)
│   ├── run_registry.py       #   RunDescriptor / ArtifactBinding / 来源根与路径重写
│   ├── adapters/             #   数据适配器(原始 schema → 统一 Entity/媒体引用)
│   │   ├── video_structure.py    # shots-v1 / globals-v1 / segment-media-v1
│   │   ├── semantic_captions.py  # global/shot/segment-caption-v1
│   │   ├── voice_assets.py       # voice-preview-v2
│   │   └── person_assets.py      # tracklets-v2 / characters-legacy-v1
│   ├── media_service.py      #   media_id 签发/解析 + Range(206/416)+ 路径边界
│   ├── module_service.py     #   模块状态/指标/实体/溯源聚合
│   └── app.py                #   FastAPI 组装(静态前端 + API + 媒体同源)
├── frontend/                 # React + TypeScript + Vite(免运行时 Node)
│   └── src/
│       ├── app/              #   外壳/URL query 路由/spec 轮询/页面
│       ├── renderers/        #   文档二 renderer 注册表(video-timeline 等)
│       └── components/       #   状态徽章/指标/单实例播放器/安全内容块/JSON 抽屉
├── frontend_dist/            # vite build 产物(backend 直接托管;构建后生成)
└── tests/selftest.py         # 自测:解析器坏用例 + 真实样本适配器断言
```

## 运行

```sh
# 一次性准备(本机已配置好 Node 24 + Python 3.12)
python -m pip install fastapi "uvicorn[standard]" markdown-it-py
cd frontend && npm install && npm run build && cd ..

# 启动(在 result_viewer 目录)
python -m backend.app --config viewer_config.json
# 打开 http://127.0.0.1:8930/

# 前端开发模式(另有 Vite dev server,代理 /api /media 到 8930)
cd frontend && npm run dev   # http://127.0.0.1:5173

# 自测
python tests/selftest.py
```

依赖:Python ≥ 3.10、fastapi、uvicorn、markdown-it-py;Node ≥ 18(仅构建前端需要)。

## 一键启动/部署(serve.py)

`serve.py` 是给本机和服务器部署用的统一入口(预检依赖、前端产物缺失时自动构建、打印本机/局域网访问地址),替代旧 `asset_viewer` 手敲 `python -m extract_full_pipeline.asset_viewer.server --host 0.0.0.0 --port 8910` 的方式:

```sh
# 在仓库根目录
python result_viewer/serve.py                          # 0.0.0.0:<配置端口>,可访问到本机/服务器的人都能打开
python result_viewer/serve.py --host 127.0.0.1         # 仅本机
python result_viewer/serve.py --port 8940              # 换端口(或环境变量 RESULT_VIEWER_PORT)
python result_viewer/serve.py --root-path /gemini_web/result_viewer/<代理段>/
                                                       # 平台代理"保留前缀转发"时设置(见"平台代理前缀"一节)
```

暴露给他人前确认:`viewer_config.json` 的 `allowed_runs`/`source_roots` 只登记了要展示的运行目录(服务是只读的,媒体读取被限制在登记根内);Windows 本机首次对外监听可能需放行防火墙。常驻部署建议用 systemd/supervisor 托管同一条命令,不要依赖交互式终端。

## 数据源

当前绑定本仓库内的 `movie_full_pipeline_20260912_v4`(旧链整合运行,3 个 global)。
所有绑定在 `viewer_config.json` 中显式声明(文档三 §6.1 的 custom 布局):

| 产物 | 相对路径 | 模块 |
|---|---|---|
| shots / globals | middle/*.jsonl | 镜头切分 / 全局片段聚合 |
| global_captions / attributed_asr | captions_speaker/global/ | 全局描述与说话归属 |
| shot_captions | captions_speaker/shot/ | 镜头描述 |
| segment_captions | captions_speaker/segment/ | 片段描述 |
| segment_media | segments/segments.jsonl | 片段视频导出 |
| voice_previews | voice_by_role_src/all_asr_manifest.jsonl | 角色声音候选 |
| tracklets | assets/visual_person_v2/tracklets | 人物轨迹提取 |
| role_binding / legacy_characters | characters/ | 旧链角色身份绑定 |

旧 manifest 中的绝对路径通过 `path_rewrites` 规则映射回本地受控根;无规则命中
或文件不存在时显示「源文件未收录」,不做全盘搜索。person_align、scene、object、
I2I 等本运行未收录的模块显示 `未收录` 状态而不是空列表。

## 平台代理前缀

`viewer_config.json` 的 `base_path` 配置代理前缀(如 `/gemini_web/result_viewer`)。
前端使用相对路径 fetch + `vite base:'./'`,页面/API/媒体自动走同一前缀;
启动时用 `--root-path` 匹配代理是否剥离前缀(文档三 §9.2)。

## 与设计文档的验收对照

| 文档三验收项 | 状态 |
|---|---|
| 文档即内容源,改文档二热更新 | 已实现(2s 轮询,浏览器实测:改标题→页面更新;坏 JSON→保留旧版+提示"新说明未生效";恢复→提示消失) |
| 模块易增删 | 解析器按注册能力校验;27 个模块全部解析 |
| 说明与结果一致 | 每个结果页四段布局,说明抽屉来自文档二六段正文,无第二份文案 |
| 来源可解释 | 实体带 provenance/relations;缺失媒体显示原因 |
| 状态不误导 | 候选=valid 旧分类已标注"非审核通过";91/190 类覆盖率带分母;轨迹数≠人物数 |
| 单端口承接页面/API/媒体 | FastAPI 同源;媒体支持 HEAD/Range(206/416 实测) |
| 路径边界 | media_id 服务端签发,resolve 后必须在登记来源根内(../ 越界实测 404) |
| URL 分享恢复 | query 路由 ?tab&run&module&scope&entity&view,刷新/直达实测通过 |

## 尚未覆盖(交接)

- `global-alignment-v1`(person_align)、`characters-global-v1`(新链物化)、object/scene/I2I/TTS
  等 18 个 adapter 已登记(文档校验可解析)但未实现数据解析;绑定后返回明确的
  "已注册但本版本未实现数据解析" 状态。schema 依据见文档二对应模块,接入时新增
  `adapters/*.py` 并在 `adapters/__init__.py` 注册即可。
- 0916 运行(workspace 下)未在本机;如需接入,在 viewer_config.json 增加一个 run
  并按 `integrated` 布局绑定产物。
- 旧 URL 兼容映射、SSE 推送、`.viewer_cache/docs` 快照缓存(文档三 §5.4)未实现。
- 性能目标(文档三 §1)未测量;当前样本(59 镜头/72 音频/3 global)下页面即点即开。

## 自测记录

- `python tests/selftest.py`:39 项全部通过(2026-09-20)。
  覆盖真实文档解析、16 类非法文档拒绝、适配器真实计数(58/72 候选、190/143/59 轨迹)、
  媒体重写与越界拒绝。
- 浏览器端到端(2026-09-20):系统介绍/声音候选/角色画廊/轨迹/全局描述/未收录模块
  页面渲染正常;分享链接直达;热加载与坏文档回退实测通过。
