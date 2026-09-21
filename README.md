# CharacterAssetExtract · 视频理解与资产提取

输入长视频（电影/剧集），产出**可浏览、可追溯**的结构化描述与资产：三级文字描述、人物（轨迹→角色归属→角色素材目录）、声音候选、关键物体（定位/分割/参考集）、场景参考图。只读结果网站随仓库附带，跑完即可浏览全部产物。

## 主线流程（15 阶段 + 默认统一降采样，全部最新逻辑，无旧 CHR 链）

```
视频 → [统一降采样 normalize_video(默认 480p/15fps, --no-normalize 关闭)]
     → 切镜 → 全局片段聚合(60–120s, 场景检测 SCRL|semsim|metadata)
     → 全局描述+说话归属(MiMo, 吃外部 ASR) → 镜头描述 → 固定段规划 → 片段描述 → 片段视频
     → 人物轨迹提取 → 轨迹全局归属到 S-ID → 角色声音候选 → 角色素材物化
     → 关键物体候选(来自镜头描述) → 定位与分割 → 多帧参考集
     → 场景参考图
```

- 物体候选不是独立模型步骤：镜头描述模型在 `key_object_candidates` 字段里直接给出候选（含优先级与可见时间窗），队列脚本只做过滤/编号/时间窗传递，定位分割阶段才用 SAM/grounding 模型验证。
- 旧 CHR 人物链、声音增强建模（前景分离/说话人建模）是**支线**，默认不进主线，详见 [docs/02](docs/02_全流程Pipeline说明.md) 文末“支线与历史兼容模块”。

## 2026-09-21 增强（代码已合入，服务器实验待做）

- **统一降采样**：`preprocessing/normalize_video.py`，run_full_pipeline 默认第一阶段（480p/15fps/crf20）；切镜与场景提取改吃 normalized.mp4，显著降低 MiMo 请求体积与检测推理量。注意 frame_range 语义随之变为 normalized 时间轴。
- **semsim 场景检测**：`global_clip_aggregator.py --scene-detector semsim`，免训练无 LLM——每镜头 DINOv3 自适应采样 + 上下文自相似（BaSSL/ShotCoL 取向）+ 音频响度对比（CMS 取向）+ 24s/240s 时长先验；`auto` 回落顺序 SCRL → semsim → metadata（逐级 WARNING）。SCRL/TransNetV2 权重支持 `SCRL_ENCODER_PATH`/`SCRL_BOUNDARY_PATH`/`TRANSNET_MODEL_PATH` 环境变量覆盖。
- **场景提取三选项**（默认行为不变）：`--shots-manifest`（采样吸附镜头边界）、`--max-scenes auto`（silhouette 自动定簇数）、`--anchor-selection coverage`（k-center 多样性锚点）。
- **CocktailASR 多参考多说话人**：`cocktail_asr.py --ref S1=path --ref S2=path`（官方无多说话人版本，利用目标说话人拒识做多遍合并）；`test_cocktail_asr.py --via multi` 验收。
- **切镜聚合容忍 --no-export**：镜头 clip 缺失时回退源视频按时钟采样，不再报错。
- 详见 [docs/02](docs/02_全流程Pipeline说明.md) v1.4.0 与 [TODO_ANNOTATION_SYSTEM.md](TODO_ANNOTATION_SYSTEM.md)；对比实验工具在 `experiments/division_compare/`。

## 怎么跑

### 1) 全流程一键编排（推荐入口）

```sh
python run_full_pipeline.py --input-dir <视频目录> --asr-dir <ASR目录> --dry-run   # 先看计划
python run_full_pipeline.py --input-dir <视频目录> --asr-dir <ASR目录> \
       --update-viewer-config result_viewer/viewer_config.json                    # 正式跑并并入网站
```

- 自动扫描目录下所有视频，每视频一个独立工作区（`full_pipeline_runs/<视频名>/`）。
- 断点续跑：看 `pipeline_state.json`（阶段状态）与 `logs/<阶段>.log`；成功且产物存在的阶段自动跳过；`--only` / `--force` 精细控制。
- 失败默认继续下一个视频；退出码 = 失败视频数。
- **外部前置**（编排器只预检不代配）：ffmpeg/ffprobe；VLM 端点（`OPENAI_BASE_URL`/`OPENAI_API_BASE`/`API_BASE`、`OPENAI_API_KEY`、`MODEL_NAME`）；每视频的 ASR 文件（按 `<视频名>*.jsonl/json` 自动发现，或 `--asr-map` 指定）；物体（SAM3.1/grounding）、场景（YOLO/DINOv3）、物化可选 InsightFace 的本地模型权重。
- `python run_full_pipeline.py --list-stages` 查看 15 个阶段与对应脚本。

### 2) 分步 / 局部重跑

入口总表（基础编排、Scheme A、人物归属 shell、旧 CHR 链等）与局部重跑边界见 [docs/02](docs/02_全流程Pipeline说明.md) 的“入口、共享基础设施与局部重跑”一节。

### 3) 看结果（只读网站）

```sh
python result_viewer/serve.py            # 默认 0.0.0.0:8930，同网段可访问
python result_viewer/serve.py --host 127.0.0.1    # 仅本机
```

网站内容（导航/说明/流程图）来自 `docs/02` 的热加载快照；运行结果来自 `result_viewer/viewer_config.json` 的显式产物绑定（编排器 `--update-viewer-config` 会自动追加，改配置后重启 serve.py）。详见 [result_viewer/README.md](result_viewer/README.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/00_系统模块速览](docs/00_系统模块速览.md) | 27 个模块：功能 + 代码位置（由文档二生成） |
| [docs/01_系统文档规范与Agent协作](docs/01_系统文档规范与Agent协作.md) | PipelineDoc 1.0 写作/校验协议 |
| [docs/02_全流程Pipeline说明](docs/02_全流程Pipeline说明.md) | 全流程权威说明：主线/支线、每模块六段详解、产物契约、已知差异 |
| [docs/03_结果展示网站设计](docs/03_结果展示网站设计.md) | 结果网站设计与部署（信息架构以 `movie_full_pipeline_20260912_v4` 真实产物为基准） |

## 目录速览

```
run_full_pipeline.py        全流程一键编排（15 阶段）
shot_splitter.py / global_clip_aggregator.py / mimo_segment_cutter.py   结构与切段
mimo_*captioner*.py         三级描述（最新：speaker_v2 / shot_v3 / fixed_segments_v4）
global_align_pipeline/      人物归属 + 角色物化（新链）
assets/visual_person/       轨迹提取(shot_tracklet_builder)
assets/key_object/          物体候选队列 + 定位分割(V4)
assets/scene/               场景参考图
assets/voice/               声音候选导出(主线) + 声音增强建模(支线)
segmentation/               固定段规划(three_way)
result_viewer/              只读结果网站（backend + frontend + serve.py）
asset_viewer/               旧只读网站（保留参考，不再扩展）
docs/                       四份协议文档
movie_full_pipeline_20260912_v4/   仓库内样本运行（网站当前绑定）
```

## 注意

- `prompts/` 目录（captioner 提示词）曾因 `.gitignore` 的 `*.md` 规则未进版本库；现已加 `!prompts/**` 例外，`git add prompts/` 即可纳入，同步到新机器前记得提交，否则三级描述阶段无法启动。
- 本仓库 README 同样命中 `*.md` 忽略规则，纳入版本库需 `git add -f README.md`（docs/ 已有 `!docs/*` 例外）。
- 结果网站是只读服务，媒体读取被限制在 `viewer_config.json` 登记的来源根内；对外暴露前确认 `allowed_runs` 只含要展示的运行。
