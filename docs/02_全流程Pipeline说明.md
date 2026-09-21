# 视频理解与资产提取：全流程 Pipeline 说明

本文件同时服务人类阅读和未来结果网站的内容加载。格式遵循[文档规范](01_系统文档规范与Agent协作.md)，网页实现遵循[网站设计](03_结果展示网站设计.md)。核验日期：2026-09-18；`1.1.0` 重排主线/支线、`1.2.0` 新增主线编排入口、`1.3.0 / 2026-09-21` 展示瘦身（非主线模块移至文末，未删除），均未改变模块事实。依据为当前源码工作树及实际结果目录；本次未执行模型推理或重跑系统。

<!-- section:system-index -->
## 系统索引

```json pipeline-system
{
  "doc_id": "extract-full-pipeline",
  "schema_version": "1.0.0",
  "content_version": "1.4.0",
  "updated_at": "2026-09-21",
  "title": "视频理解与资产提取系统",
  "summary": "从视频结构和语义描述出发，形成可追溯的人物、声音、物体、场景资产及生成效果对比。",
  "code_baseline": "0c5ddfd367388461d96bed16fc57e4560a70ace8",
  "worktree_dirty": true,
  "evidence_section": "evidence-baseline",
  "default_route": "fixed-global-align",
  "groups": [
    {"id":"structure","title":"视频结构","order":10},
    {"id":"semantics","title":"内容理解","order":20},
    {"id":"people","title":"人物视觉资产","order":30},
    {"id":"audio","title":"声音资产","order":40},
    {"id":"characters","title":"角色汇总","order":50},
    {"id":"objects","title":"关键物体","order":60},
    {"id":"scenes","title":"场景参考","order":70},
    {"id":"compatibility","title":"支线与扩展模块","order":80}
  ],
  "routes": [
    {"id":"fixed-global-align","title":"固定分段与全局人物归属","modules":["shot_split","global_group","segment_plan","asr_timeline","global_describe","shot_describe","segment_describe","segment_export","person_tracks","person_align","voice_preview","character_pack","object_queue","object_ground","scene_select"],"description":"主要讲解路径，与 run_full_pipeline.py 的 15 阶段编排一一对应（外部 ASR 为输入）；编排默认先执行统一降采样阶段（normalize_video，--no-normalize 可关），产物属于 source_prepare 模块。扩展与支线模块见文末章节。"},
    {"id":"basic-caption","title":"基础三级描述","modules":["source_prepare","shot_split","global_group","global_describe","shot_describe","segment_describe","segment_export"],"description":"mimo_video_pipeline.py 的六步编排；global 不含 Scheme A 的权威 ASR 归属，segment 由模型在边界约束下组织。"},
    {"id":"legacy-chr","title":"CHR 聚类与角色映射","modules":["source_prepare","shot_split","global_group","segment_plan","asr_timeline","global_describe","shot_describe","segment_describe","person_tracks","legacy_refine","legacy_identity","legacy_gallery","voice_model","voice_preview","legacy_bind"],"description":"保留逐镜头语义筛选、跨镜头身份图、CHR 画廊、投票绑定，用于兼容和对照。"},
    {"id":"audio-evidence","title":"前景声音与身份验证","modules":["source_prepare","shot_split","global_group","voice_model"],"description":"可独立运行；得到 SPK 身份和净语音证据，不自动得到视觉 S-ID。"}
  ],
  "layout_profiles": [
    {"id":"integrated","title":"0916 整合展示布局","description":"middle/、captions/、global_alignment/、characters/ 等；可能复用外部工作区媒体，部分中间产物未收录。"},
    {"id":"basic-flat","title":"基础编排扁平布局","description":"shots/、global/、captions/、segments/；由基础编排 --flat 产生。"},
    {"id":"scheme-a","title":"Scheme A 原始布局","description":"annotations/<level>/<video>/；此入口未传 --flat，结构输入来自上游。"},
    {"id":"custom","title":"显式绑定布局","description":"按模块绑定 output、tracklets、identity、asset_bank、asr_file 等真实路径；不进行全盘猜测。"}
  ],
  "sample_run": {"run_id":"movie_full_pipeline_20260916","workspace_relative_to_repo":"../workspace/movie_full_pipeline_20260916","checked_at":"2026-09-18","scope_ids":["global_0040","global_0041","global_0042"]}
}
```

这里的 route 是解释和展示路径，不是可执行 DAG 配置。可选模块、输入是否已具备以及实际 CLI 仍需检查。下面元数据中 renderer/adapter 是未来网站约定的注册名，并非现有 viewer 已实现的接口。
<!-- /section:system-index -->

<!-- section:overview -->
## 先理解整个系统

系统需要回答三类问题：视频发生了什么；画面中的人、物体、场景分别是什么；这些内容能否转化为可浏览、可复用、可追溯的素材。它先建立时间结构，再建立语义身份，最后把语义判断落实到图片、掩码、音频和视频文件。

```text
源视频 → 镜头清单 → 全局片段（60–120 秒）
                         │           ├→ 固定 segment 计划
                         │           └→ 外部 ASR → 全局描述与说话归属
                         │                            ↓
                         ├──────────────────────────→ 镜头描述 → 片段描述 → 片段视频
                         │
                         ├→ 多帧人物轨迹 → 轨迹全局归属到 S-ID ──┐
全局 ASR 归属 → 候选音频预览 ────────────────────────────→ 角色目录（裁图 + 候选声音）
                         │
                         ├→ 镜头描述 → 关键物体候选 → 定位与分割 → 物体参考集
                         └→ 全局视频 → 场景聚类与选帧 → 场景参考图
所有产物 + 本文档 → 只读结果网站

（不在上图中展开：可选预处理、I2I 生成对比等扩展模块，以及旧 CHR 人物链、声音增强建模与兼容支路，见文末“支线、扩展与外部模块”。）
```

主线只讲上图：先建时间结构与语义身份，再把语义判断落实为人物、声音、物体、场景资产；这与 run_full_pipeline.py 的 15 阶段编排一一对应（外部 ASR 为输入）。场景提取不依赖字幕或人物身份，可与语义链并行。其余模块移至文末“支线、扩展与外部模块”：可选预处理与 I2I 生成对比属于扩展；旧 CHR 人物链与轨迹直接归属的新链是替代/对照关系，不是必经步骤；声音增强链不依赖视觉人物归属，需要证据连接才能把 SPK 与 S-ID 关联。

### 身份和时间：所有页面共用的基础

| 概念 | 正确含义 | 禁止混淆 |
|---|---|---|
| source video | 当前切镜所使用的视频；可能已预处理 | 不一定是原始未裁剪影片 |
| shot | 基于切镜结果的连续画面区间 | 不等于一个动作或人物 |
| global | 通常约 60–120 秒的上下文片段 | 不等于整部电影，也不保证只含一个语义场景 |
| segment | global 内用于描述/导出的中等长度片段 | 不同分支的 `segments.jsonl` schema 不同 |
| OBS | 单帧人物观测 | 不是跨帧/跨镜头身份 |
| tracklet | 一个 shot 内连续人物轨迹 | 一个角色可能对应很多轨迹 |
| S-ID | global 内语义人物 ID | global_0040:S1 与 global_0041:S1 不可直接合并 |
| CHR | 旧链在指定输入集合上聚类的视觉身份 | 不是 S-ID，编号也不跨运行稳定 |
| SPK | 声音建模输出的说话人身份 | 不等于画面人物 ID |
| scene cluster | 根据视觉相似性聚合的场景参考组 | 不是故事情节段，也不必连续出现 |
| object candidate | 从描述提出的物体候选 | 不代表已定位正确或可用于生成 |

时间以秒为单位。清单 `shots.jsonl/global_clips.jsonl` 的 `time_range_sec` 是 source 时间；global caption 的内部事件、归属 ASR 和 segment 的 `time_range_sec` 通常是 global-local；tracklet 和物体可见窗是 shot-local。媒体本身的播放时间总从 0 开始。

关系式：`source_t = global_source_start + global_local_t`；`source_t = shot_source_start + shot_local_t`；`segment_player_t = global_local_t - segment_start_in_global`。例如样本 global_0040 从 source 3512.926083 秒开始，ASR 局部 9.04 秒对应 source 3521.966083 秒。不要拿 source 时间直接 seek global 视频。

如果预处理保留多个不连续区间，需要按 `kept_intervals` 的累计长度分段反算到原始影片时间；不能统一加一个片头偏移。源 shot 的 `frame_range` 代码写入末帧索引，不能一概解释成半开区间。网站统一的秒区间应另行约定为 `[start,end)`，保留原始字段。

### 运行布局与真实样本

```text
movie_full_pipeline_20260916/
├── middle/{shots.jsonl,global_clips.jsonl,shot_clips/}
├── global_clips/*.mp4
├── captions/{global,shot,segment}/
├── segments/{segments.jsonl,clips/,summary.json}
├── global_alignment/{requests/,stitched/,alignments/,global_tracklet_alignment.jsonl,summary.json}
├── characters/<scope>/<subject>/{visual/,voice/,role_character.json}
├── voice_by_role_src/{all_asr_manifest.jsonl,valid_manifest.jsonl,invalid_manifest.jsonl,valid/,invalid/,summary.json}
├── key_object/{grounding_queue.jsonl,grounding/}
├── scene/、scene_opt_marginal/
├── i2i_bundles/{scene,object}/、i2i_test/、tts_test/outputs/
└── materialization_summary.json
```

这是一份整合工作区：部分 manifest 保留 0912 的绝对路径，shot_clips 使用指向 0912 的软链接；此根目录未收录原始人物 tracklets、CHR registry 和完整声音增强产物。轨迹中间结果可在 0912_v4 的 `assets/visual_person_v2/tracklets` 找到，但关联前仍要校验其来源，不能靠名字偷偷跨运行拼接。

样本核验结果：3 个 global、59 个 shot、12 个导出 segment；190 条轨迹中 91 条归属 S-ID、99 条 unknown；物化 16 个 `(scope,S-ID)` 角色目录；ASR 预览 72 条，其中候选 58、非候选 14；物体候选 25 个、观测 125 行，其中 67 行 `ambiguous_instance=true`，虽然这 125 行 `status` 都是 `ok`。这些数是静态样本快照，不是准确率，不是全片统计，也不能硬编码为网页 KPI。
<!-- /section:overview -->

<!-- module:shot_split -->
## 01 · 镜头切分

```json pipeline-module
{
  "id":"shot_split","title":"镜头切分","order":10,"group":"structure","lifecycle":"active","implementation":"existing",
  "summary":"将视频变为覆盖时间轴的镜头清单和独立镜头视频。",
  "depends_on":["source_prepare"],"optional_depends_on":[],"inputs":["source_video"],
  "source_refs":["shot_splitter.py","m02_shots/teleai_src/teleai_cut_only.py","m02_shots/run_teleai_pipeline.sh"],"evidence":["code","sample"],
  "outputs":[
    {"id":"shots","title":"镜头清单","kind":"jsonl","paths":{"integrated":["middle/shots.jsonl"],"basic-flat":["shots/shots.jsonl"],"scheme-a":["shots/{video}/shots.jsonl"],"custom":["{output}/shots.jsonl"]},"record_key":["shot_id"],"timebase":"source","schema":"unversioned","required":true},
    {"id":"shot_media","title":"镜头视频与统计","kind":"directory","paths":{"integrated":["middle/shot_clips"],"basic-flat":["shots"],"scheme-a":["shots/{video}"],"custom":["{output}"]},"record_key":["shot_id"],"timebase":"shot_local","schema":"unversioned","required":false}
  ],
  "view":{"enabled":true,"renderer":"video-timeline","adapter":"shots-v1","filters":["scope","shot"],"primary_artifact":"shots"}
}
```

### 作用与判断

切镜建立后续共用的时间结构，回答“画面在哪里换镜头”。默认 `auto` 优先使用 TransNetV2；还支持 PySceneDetect 的 adaptive/content 模式，实际使用的检测器写进 summary。

### 输入与前置条件

消费 source_video；模型检测需要相应依赖和权重，视频导出需要 ffmpeg。TransNetV2 权重默认是集群绝对路径，可经 `TRANSNET_MODEL_PATH`/`TRANSNET_SOURCE_DIR` 环境变量覆盖；`auto` 因权重缺失回落 PySceneDetect 时会打 WARNING，两种检测器的切镜结果口径不可比。导出目录是否追加 `<video>/` 由 `--flat` 决定。不要把 CLI 输出根和最终清单目录混为一谈。

### 处理过程

探测视频 → 检测边界 → 修补全片覆盖 → 默认合并短于 2 秒的镜头 → 按显式最大时长可选拆长镜头 → 导出视频与清单。默认 `max-shot-sec=None`，不保证每个镜头小于 20 或 30 秒。

### 产物与消费

`shots.jsonl` 每行一个 shot：`shot_id/source_video/clip_path/time_range_sec/duration_sec/frame_range/fps/boundary_type/raw_scene_index`。普通输出还包含 `clips/<stem>_shot_<五位编号>.mp4` 与 `summary.json`。`--no-export` 时 `clip_path=null`，不能按文件存在判断整步失败。消费方：global 聚合、镜头描述、人物轨迹、物体定位、时间映射。

### 异常、重跑与成本

检测与视频导出可能很贵。基础编排只因清单存在便跳过本阶段，不验证输入是否变更或媒体是否完整。切镜变更后所有引用 shot_id 的数据需失效/迁移。

### 展示与验收

展示缩略图时间轴、时长、播放和原片定位。检查清单非空、区间有序、时间覆盖、ID 唯一、媒体可读；显式处理 no-export。不要把 `SHOT_768`、`SHOT_00768`、数字 768 当成三个镜头。
<!-- /module:shot_split -->

<!-- module:global_group -->
## 02 · 全局片段聚合

```json pipeline-module
{
  "id":"global_group","title":"全局片段聚合","order":20,"group":"structure","lifecycle":"active","implementation":"existing",
  "summary":"把相邻镜头组织为模型可理解的较长上下文片段。",
  "depends_on":["shot_split"],"optional_depends_on":[],"inputs":["shots"],
  "source_refs":["global_clip_aggregator.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"globals","title":"全局片段清单","kind":"jsonl","paths":{"integrated":["middle/global_clips.jsonl"],"basic-flat":["global/global_clips.jsonl"],"scheme-a":["global/{video}/global_clips.jsonl"],"custom":["{output}/global_clips.jsonl"]},"record_key":["global_id"],"timebase":"source","schema":"unversioned","required":true},
    {"id":"global_media","title":"全局视频与统计","kind":"directory","paths":{"integrated":["global_clips"],"basic-flat":["global"],"scheme-a":["global/{video}"],"custom":["{output}"]},"record_key":["global_id"],"timebase":"global_local","schema":"unversioned","required":true}
  ],
  "view":{"enabled":true,"renderer":"video-timeline","adapter":"globals-v1","filters":["scope"],"primary_artifact":"globals"}
}
```

### 作用与判断

global 给人物命名和长程事件理解提供上下文。场景检测可选 `scrl`（学习式 BiLSTM 边界）、`semsim`（免训练语义边界）、`metadata`、`none` 与 `auto`；算法在场景分组内按目标长度聚合。

### 输入与前置条件

消费 shots 清单，镜头视频和源视频应可读。SCRL 使用本地权重及 vendored 代码，权重路径可用 `SCRL_ENCODER_PATH`/`SCRL_BOUNDARY_PATH` 环境变量覆盖；semsim 复用场景链的本地 DINOv3 目录（`--semsim-model`，默认 `local_models/dinov3-vitl16`，可用 `SEMSIM_DINO_MODEL` 覆盖）。镜头 clip 缺失（如上游 `--no-export`）时场景检测回退到按源视频绝对时间采样，clip 与源片都不可读才报错。

### 处理过程

读取 shots → 场景分组 → 默认 min/target/max 为 60/90/120 秒进行镜头分组 → 优先从源视频区间切出 global；不能使用源片时可拼接镜头。镜头完整性、场景边界和尾段可能导致长度偏离目标，页面应展示实际长度。

`auto` 的回落顺序为 SCRL → semsim → metadata，每级回落打 WARNING。semsim 是 2026-09-21 新增的免训练检测器：每镜头按时长自适应采样（约每 2 秒 1 帧、上限 8）算 DINOv3 embedding，边界分数 = 60% 上下文窗口自相似不匹配 + 40% 相邻镜头不相似，可叠加音频响度对比（一次 ffmpeg RMS pass，`--semsim-audio-weight` 默认 0.25）；阈值默认自适应（mean+σ·std），并以 24s/240s 最小/最大场景时长先验贪心接受边界。方法取向为 BaSSL/ShotCoL 上下文自相似 + CMS 音视对比 + 时长正则，未用 LLM、未引入新权重。

### 产物与消费

`global_clips.jsonl` 字段：`global_id/source_video/clip_path/time_range_sec/duration_sec/shot_ids/shot_count/scene_id/scene_detector/source_shots_manifest/export_mode`；视频为 `global_clips/<stem>_global_<四位编号>.mp4`，另有 summary。消费方：三级描述、上游固定分段、声音/场景、人物全局对齐。

### 异常、重跑与成本

SCRL 提特征与重编码影响成本；semsim 的成本主要在 DINOv3 逐镜头采样（比 SCRL 的固定 3 帧多，长片差距更大）。基础编排同样可能仅因清单存在跳过；改变 global 边界会改变 S-ID 作用域，不能沿用旧角色键。拼接模式还要核对实际视频时间与 manifest 时长一致。

### 展示与验收

展示 global 编号、源时间、包含镜头和视频，进入后所有页面共用该 scope。样本 0040/0041/0042 时长分别为 87.378959/91.966875/89.380958 秒。验收 shot_ids 可解析、顺序正确、输出范围和播放时长相符。
<!-- /module:global_group -->

<!-- module:segment_plan -->
## 03 · 固定片段规划

```json pipeline-module
{
  "id":"segment_plan","title":"固定片段规划","order":30,"group":"structure","lifecycle":"optional","implementation":"existing",
  "summary":"在 Scheme A 中由上游决定 segment 边界，模型只补充内容。",
  "depends_on":["shot_split","global_group"],"optional_depends_on":[],"inputs":["shots","globals"],
  "source_refs":["segmentation/three_way.py","m07_segment/window.py","mimo_segment_captioner_fixed_segments_v4.py"],"evidence":["code","sample"],
  "outputs":[{"id":"segment_plan","title":"上游固定分段清单","kind":"jsonl","paths":{"custom":["{output}/segments_{impl}/segments.jsonl"]},"record_key":["global_id","segment_id"],"timebase":"mixed","schema":"unversioned","required":true}],
  "view":{"enabled":true,"renderer":"video-timeline","adapter":"segment-plan-v1","filters":["scope","segment","variant"],"primary_artifact":"segment_plan"}
}
```

### 作用与判断

隔离“如何分段”和“如何描述”。`three_way.py` 可比较 window/v1/v2 三种局部策略，默认目标 22.5 秒、最短 15 秒、最长 30 秒；与 `m07_segment` 的默认 40 秒目标不同。

### 输入与前置条件

输入 global_clips 和 shots。fixed_segments_v4 要求最终选择一份确切 `segments.jsonl`；目录下找到多份时会报错。0916 样本 caption 记录了固定计划，但该整合根没有单独保留最初规划文件，不能反推究竟来自哪一路实验。

### 处理过程

window 按长度贪心；v1 按 shot bucket 累加；v2 近似复用边界枚举的语义。均不调用 MiMo 来决定分段。输入长 shot 等边缘情况须查看实际 summary，不能把偏好长度当绝对保证。fixed_segments_v4 将上游区间裁到当前 global、求 shot 交集，并核验 shot 序列完整且只出现一次、上游段 ID 唯一。

### 产物与消费

`segments_window/segments.jsonl`、`segments_v1/segments.jsonl`、`segments_v2/segments.jsonl`；总 `summary.json` 比较分段数量及最小/平均/最大时长。关键字段 `segment_id/global_id/shot_ids/time_range_sec/source_time_range_sec/duration_sec/impl/score`。消费方为固定片段描述。单独算法规划和最终切出视频的清单不可互换。

### 异常、重跑与成本

纯规划主要为 CPU，便于先低成本比较结构。调整分段后须重做片段描述、切片与 segment 级 I2I 关联；未改 global 和 shot 时可复用它们的描述。ClipRecord 兼容支路见后文。

### 展示与验收

以同一 global 的多条横向分段条对比三种结果；明确当前采用的方案。验收覆盖、顺序、重复、跨 global 裁剪和长 shot 例外。无独立文件时显示“从描述记录读取固定计划，原始规划来源未收录”。
<!-- /module:segment_plan -->

<!-- module:asr_timeline -->
## 04 · 语音时间轴输入

```json pipeline-module
{
  "id":"asr_timeline","title":"语音时间轴输入","order":40,"group":"semantics","lifecycle":"external","implementation":"external",
  "summary":"为 Scheme A 提供不可由描述模型随意改写的 ASR 文本与时间。",
  "depends_on":["global_group"],"optional_depends_on":["voice_model"],"inputs":["globals","speaker_registry"],
  "source_refs":["mimo_segment_captioner.py","mimo_global_captioner_speaker_v2.py","assets/voice/cocktail_asr.py"],"evidence":["code","sample"],
  "outputs":[{"id":"authoritative_asr","title":"权威 ASR 输入","kind":"structured","paths":{"custom":["{asr_file}"]},"record_key":["global_id","asr_id"],"timebase":"mixed","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"transcript","adapter":"asr-v1","filters":["scope"],"primary_artifact":"authoritative_asr"}
}
```

### 作用与判断

ASR 负责“什么时候说了什么”，global 描述再判断“谁说的”。这里是输入边界，不代表 Scheme A 自带完整 ASR 生产器。原始 CLI 接收 JSON、JSONL 或目录；网页绑定到确定的文件集合，由适配器按真实格式归一化，元数据 kind=structured，不要求为展示先改写源文件。

### 输入与前置条件

必须明确 ASR 对应源片还是 global、句级边界和 `asr_id`。`--asr-timebase` 支持 auto/source/global-local；需要跨文件关联时优先显式指定。`speaker_registry` 是可选外部证据来源，不要求先跑声音增强链。

### 处理过程

读取外部 ASR → 匹配 global → 投影时间 → 建立权威条目。模型只能返回对这些 ID 的归属判断，不能添加/遗漏/重复条目。当前工作树声音 builder 已改接目标说话人 CocktailASR；自动挑一个参考声纹的输出不能当作全片所有人的完整转写。未明确分段时也不能当作逐词精确时间。

### 产物与消费

归一化输入含 `asr_id/text/start/end` 或标准化时间区间、语言和来源，主消费方为 global_describe。真正归属后输出由下一模块拥有。原始 ASR 文件路径依运行参数，不存在全系统统一固定目录。

### 异常、重跑与成本

缺 ASR 时 Scheme A global-caption 明确拒绝；可以选择基础描述路线，但不能把两者输出视为相同。核验文字与时间无需再跑 ASR。ASR 内容/切点变化会影响说话归属、音频切片及对白定位。

### 展示与验收

按时间显示文本、输入来源、时间精度；未知说话人保留未知。验收 ID 唯一、区间有效、在视频内、时基明确。目标说话人 ASR、全员 ASR 和人工文本要显示来源差异。
<!-- /module:asr_timeline -->

<!-- module:global_describe -->
## 05 · 全局描述与说话归属

```json pipeline-module
{
  "id":"global_describe","title":"全局描述与说话归属","order":50,"group":"semantics","lifecycle":"active","implementation":"existing",
  "summary":"在一个 global 内建立人物注册表、事件时间轴，并按所选路线关联权威对白。",
  "depends_on":["global_group"],"optional_depends_on":["asr_timeline"],"inputs":["globals","authoritative_asr"],
  "source_refs":["mimo_global_captioner.py","mimo_global_captioner_speaker_v2.py","mimo_caption_client.py","prompts/global_caption_prompt_en_semantic_roles_speakers_v4.md"],"evidence":["code","sample"],
  "outputs":[
    {"id":"global_captions","title":"全局描述","kind":"jsonl","paths":{"integrated":["captions/global/global_captions.jsonl"],"basic-flat":["captions/global/global_captions.jsonl"],"scheme-a":["annotations/global/{video}/global_captions.jsonl"],"custom":["{output}/global_captions.jsonl"]},"record_key":["global_id"],"timebase":"mixed","schema":"mixed","required":true},
    {"id":"attributed_asr","title":"全局说话归属","kind":"jsonl","paths":{"integrated":["captions/global/speaker_attributed_asr.jsonl"],"scheme-a":["annotations/global/{video}/speaker_attributed_asr.jsonl"],"custom":["{output}/speaker_attributed_asr.jsonl"]},"record_key":["global_id","asr_id"],"timebase":"global_local","schema":"mixed","required":false}
  ],
  "view":{"enabled":true,"renderer":"caption-detail","adapter":"global-caption-v1","filters":["scope","subject"],"primary_artifact":"global_captions"}
}
```

### 作用与判断

模型基于 global 视频建立 `subject_registry`，产生场景、声音和事件描述。Scheme A 同次请求还为权威 ASR 判断 S-ID、置信度、口型/上下文证据、画外音、重叠和候选音频资格；基础 global_captioner 不等价于此增强版本。

### 输入与前置条件

消费 globals 和视频；Scheme A 必须额外有 authoritative_asr。模型端点、密钥和模型名由客户端解析，文档不保存密钥。prompt、模型配置、实际调用入口须在运行记录中保留。

### 处理过程

上传视频与上下文 → 得到 JSON → 校验角色和 ASR ID → 以本地权威 ASR 回填文本/时间 → 保存原始响应、结构化结果。重复 ASR ID、缺失/凭空新增 ID 都会拒绝。`audio_asset_candidate` 的准入还检查直接身份线索、归属置信度、重叠状态和时长/文本密度，不能只依据 S-ID 非空。

### 产物与消费

`global_captions.jsonl` 一行一个 global，外层含 `success/model/clip_path/time_range_sec`，内层 `global_caption` 含 `subject_registry/semantic_role_summary/global_event_timeline`；内容可能还有同名嵌套 `global_caption`。单项 JSON 位于 `global_captions/global_<五位编号>.json`，调试回复在 `raw_responses/`，统计在 summary。

Scheme A 的 `speaker_attributed_asr.jsonl` 一行一个 ASR：`global_id/asr_id/speaker_id/text/time_range_sec/source_time_range_sec/attribution_confidence/attribution_evidence_type/speaker_attribution_accepted/audio_asset_candidate/audio_asset_admission_status/audio_asset_rejection_reasons`。消费方是音频预览；global_captions 还供镜头/片段描述、人物归属使用。

### 异常、重跑与成本

MiMo 调用耗时与视频体积、并发、token 上限有关。基础编排 global 默认并发 2、最大完成 32768 token、重试 3；已有记录必须全部 `success=true` 才整体跳过。这个检查不验证 prompt/input 哈希。失败/JSON 截断要展示具体错误，保留原始回复供开发查验。

### 展示与验收

先显示一句话内容、人物卡、事件条，再展开对白与证据；原始 JSON 放开发详情。验收角色 ID 只在 scope 内关联，ASR 完整回填且没有误改原文，视频点击事件时使用 local 时间。
<!-- /module:global_describe -->

<!-- module:shot_describe -->
## 06 · 镜头描述

```json pipeline-module
{
  "id":"shot_describe","title":"镜头描述","order":60,"group":"semantics","lifecycle":"active","implementation":"existing",
  "summary":"沿用全局人物身份，描述镜头中的动作、画面、声音和关键物体。",
  "depends_on":["shot_split","global_describe"],"optional_depends_on":[],"inputs":["shots","global_captions"],
  "source_refs":["mimo_shot_captioner.py","mimo_shot_captioner_v2.py","mimo_shot_captioner_v3.py","mimo_video_pipeline_scheme_a.py","prompts/shot_caption_prompt_en_semantic_roles_objects_v5.md"],"evidence":["code","sample"],
  "outputs":[{"id":"shot_captions","title":"镜头描述清单","kind":"jsonl","paths":{"integrated":["captions/shot/shot_captions.jsonl"],"basic-flat":["captions/shot/shot_captions.jsonl"],"scheme-a":["annotations/shot/{video}/shot_captions.jsonl"],"custom":["{output}/shot_captions.jsonl"]},"record_key":["shot_id"],"timebase":"mixed","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"caption-detail","adapter":"shot-caption-v1","filters":["scope","shot","subject"],"primary_artifact":"shot_captions"}
}
```

### 作用与判断

在较短视频内识别动作、主体、镜头类型、构图和关键道具。人物继承所在 global 的 S-ID，物体候选只提出“值得验证的对象”。

### 输入与前置条件

镜头清单/视频 + 所属 global caption。基础编排实际调用 `mimo_shot_captioner.py`，不是看到文件名 v3 就自动调用 v3；Scheme A 同样调用基础文件，但显式传 objects_v5 prompt。原子 v3 则复用 v2 并替换 prompt/schema。

### 处理过程

匹配全局上下文 → 模型描述 → 规范化主体引用、可见时间和候选字段 → 记录成功/失败。关键物体候选应是单个可定位实例；可见区间为 shot-relative。模型返回格式受实际 prompt 与 validator 共同约束。

### 产物与消费

`shot_captions.jsonl` 外层含 shot 元数据及 `success`；内部 `shot_caption` 含描述、`subjects/visual_events/semantic_role_observations/key_object_candidates/dialogue/on_screen_text`。候选关注 `local_object_id/canonical_name/grounding_phrases/priority/visible_time_range_sec`。另有 `shot_captions/shot_<五位编号>.json`、raw_responses、summary。被 segment_describe、人物语义对齐和 object_queue 消费。

### 异常、重跑与成本

默认 shot 并发 4；只重试失败镜头可省成本。变更 prompt 或物体提案字段须检查后续队列、可见窗、缓存。旧 sample 的 schema 可能来自旧 prompt，不能按当前脚本版本强制解释全部字段。

### 展示与验收

镜头视频旁显示人物、动作与物体候选标签；点击标签跳转对应资产/候选验证。无物体候选是合法结果。验收引用 S-ID 合法、可见窗不越界、缺字段有明确提示。
<!-- /module:shot_describe -->

<!-- module:segment_describe -->
## 07 · 片段描述

```json pipeline-module
{
  "id":"segment_describe","title":"片段描述","order":70,"group":"semantics","lifecycle":"active","implementation":"existing",
  "summary":"把多个镜头组织为可理解的片段描述，并保持人物、对白与物体来源可追溯。",
  "depends_on":["global_describe","shot_describe"],"optional_depends_on":["segment_plan","asr_timeline"],"inputs":["global_captions","shot_captions","segment_plan","authoritative_asr"],
  "source_refs":["mimo_segment_captioner.py","mimo_segment_captioner_v2.py","mimo_segment_captioner_global_speaker_v3.py","mimo_segment_captioner_fixed_segments_v4.py","segment_captioner_legacy/segment_captioner.py"],"evidence":["code","sample"],
  "outputs":[{"id":"segment_captions","title":"片段描述清单","kind":"jsonl","paths":{"integrated":["captions/segment/segment_captions.jsonl"],"basic-flat":["captions/segment/segment_captions.jsonl"],"scheme-a":["annotations/segment/{video}/segment_captions.jsonl"],"custom":["{output}/segment_captions.jsonl"]},"record_key":["global_id"],"timebase":"mixed","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"caption-detail","adapter":"segment-caption-v1","filters":["scope","segment","subject"],"primary_artifact":"segment_captions"}
}
```

### 作用与判断

输出既能讲清情节，也能连接具体片段媒体。基础 V2 会给模型可选的镜头分组；Scheme A fixed_v4 使用上游固定计划，不能让模型重切或重判说话身份。这两个路径不能合称同一个分段算法。

### 输入与前置条件

两者均需 global/shot 描述。基础 V2 可带 ASR；fixed_v4 必需 segment_plan，并读取 global 已判定的 ASR。可用 `--text-only` 只给文本上下文；这是成本/视觉核验信息的选择，不能假定输出质量完全等同。

### 处理过程

基础 V2：枚举满足 15–30 秒（容差约 0.25 秒）的连续 shot 组和完整划分，要求从合法划分拷贝；校验长度、完整性、物体来源，失败重试。默认 target 20 秒，不同于 three_way 的 22.5 秒。

fixed_v4：投影并冻结上游区间和 shot_ids；要求模型返回的 upstream_segment_id 列表与顺序精确一致；只采用允许修改的描述/对象字段；忽略无来源对象；将 global 的权威对白按区间写入段内。保留 `ignored_model_dialogue_count/ignored_unsourced_object_count/object_quality_issues` 作为审计证据。上游 plan 的机械校验不等同对所有时间重叠情形的完整证明，网站仍应校验可见区间。

### 产物与消费

`segment_captions.jsonl` 一行仍是一个 global，而不是一个 segment；真正段列表在 `segment_caption.segments[]`。每段含 `segment_id/upstream_segment_id/time_range_sec/source_time_range_sec/shot_ids/description/scene_description/dialogue/key_object_candidates`，实际字段随 schema 变化。fixed_v4 的版本位于内层 `segment_caption.schema_version=segment-caption-fixed-upstream-v4`。单项文件在 `segments/global_<五位编号>.json`，另有原始回复、summary 以及可选投影后的 `speaker_attributed_asr.jsonl`。全局归属文件仍是声音准入的主要来源，不能把两个层级 ASR 文件直接拼接计数。

消费方为切片、I2I 任务包、人物对齐上下文和网页段详情。`segment_id=1` 可以在不同 global 重复；采用 `(global_id,segment_id)` 实体键。

### 异常、重跑与成本

基础编排默认 segment 并发 1、重试 6，Scheme A 重试默认 4；不应同时把所有长视频并发推至模型。只改变段描述时不用重做 shot 提取；改变边界则需重新切片及下游关联。V1 legacy 独立入口用于历史对照，不承担 V2 物体合约。

### 展示与验收

显示片段时间条、构成镜头、故事描述、对白和道具；区分规划方案和内容来源。检查所有段引用有效、段内对白可定位、物体 refs 可追溯。旧描述与新媒体混用时提示版本来源不一致。
<!-- /module:segment_describe -->

<!-- module:segment_export -->
## 08 · 片段视频导出

```json pipeline-module
{
  "id":"segment_export","title":"片段视频导出","order":80,"group":"structure","lifecycle":"active","implementation":"existing",
  "summary":"根据已确认的片段区间切出视频，建立描述与媒体的对应关系。",
  "depends_on":["segment_describe"],"optional_depends_on":["global_group"],"inputs":["segment_captions","globals"],
  "source_refs":["mimo_segment_cutter.py"],"evidence":["code","sample"],
  "outputs":[{"id":"segment_media","title":"导出片段清单","kind":"jsonl","paths":{"integrated":["segments/segments.jsonl"],"basic-flat":["segments/segments.jsonl"],"custom":["{output}/segments.jsonl"]},"record_key":["global_id","segment_id"],"timebase":"global_local","schema":"unversioned","required":true}],
  "view":{"enabled":true,"renderer":"video-timeline","adapter":"segment-media-v1","filters":["scope","segment"],"primary_artifact":"segment_media"}
}
```

### 作用与判断

将语义段落实为可直接播放的 MP4。它不会重新决定分段或重新生成 caption。

### 输入与前置条件

明确指定 segment_captions；从记录取 global 视频，必要时用 `--global-input` 的 manifest 补齐。注意编排 wrapper 对 override 的复用存在歧义，见已知差异；逐步运行时优先直接传原子模块的确切输入。

### 处理过程

展开 global 内 segments → 校验 local 时间 → ffmpeg 裁切 → 逐行记录成功/失败。默认重编码 libx264/aac；`--copy` 属于不同精度/性能选择；默认 `--skip-existing` 复用已有文件。

### 产物与消费

`segments.jsonl` 一行一个段，关键字段 `global_id/segment_id/global_clip_path/segment_clip_path/time_range_sec/source_time_range_sec/success`，附带描述；视频在 `clips/<stem>_global_<四位>_segment_<三位>.mp4`，另有 summary。与规划步骤同名的 segments.jsonl 不同。

### 异常、重跑与成本

成本主要在转码与 I/O。已存在视频不等于与当前区间匹配；改边界应隔离输出或明确失效文件。失败行和缺媒体分别提示，不因为一个成功文件就给整个 global 显示完成。

### 展示与验收

提供段视频、相邻段切换、回到 global 的对应区间。媒体播放从 0 开始，文本时间可能仍是 global-local，必须转换。样本共 12 个段，每个 global 4 个；检查时长、音画和源引用。
<!-- /module:segment_export -->

<!-- module:person_tracks -->
## 09 · 人物轨迹提取

```json pipeline-module
{
  "id":"person_tracks","title":"人物轨迹提取","order":90,"group":"people","lifecycle":"active","implementation":"existing",
  "summary":"把镜头中多帧出现的人物连接为轨迹，并保存可复核的脸部、身体与掩码证据。",
  "depends_on":["shot_split"],"optional_depends_on":[],"inputs":["shots","shot_media"],
  "source_refs":["assets/visual_person/entity_asset_extractor.py","assets/visual_person/shot_tracklet_initializer.py","assets/visual_person/shot_tracklet_builder.py"],"evidence":["code","sample"],
  "outputs":[{"id":"tracklets","title":"人物轨迹证据目录","kind":"directory","paths":{"custom":["{tracklets}"]},"record_key":["shot_id","tracklet_id"],"timebase":"shot_local","schema":"shot-tracklets-v2","required":true}],
  "view":{"enabled":true,"renderer":"tracklet-gallery","adapter":"tracklets-v2","filters":["scope","shot","status"],"primary_artifact":"tracklets"}
}
```

### 作用与判断

单帧 OBS 只证明“这一帧这里有人”；tracklet 才把同一个镜头里的多次观测连起来。此阶段不决定跨镜头人物身份，也不直接分配 S-ID。

### 输入与前置条件

输入镜头视频目录；入口 `shot_tracklet_builder.py --shots-dir ... --output ...`。底层人物观测使用 Sa2VA、GroundingDINO、SAM3.1、InsightFace 等本地能力，相关权重、依赖和 GPU 路径须预检。`entity_asset_extractor.py` 是底层 OBS 抽取，不能当作完整人物资产链入口。

### 处理过程

选较清晰初始化帧 → 提取人物检测、mask、脸/身体裁图 → 以 SAM image mask 为周期锚点 → 稀疏光流传播 + Hungarian 匹配 → ACTIVE/LOST/FINISHED 状态机 → 双向轨迹合并与准入。默认每 8 帧重检测、至少 3 次检测/8 帧等阈值；短镜头有联合证据准入，未达标轨迹记录在 filtered_tracklets，不能悄悄忽略。

去重考虑脸/mask 证据；同一检测帧独立共现形成 cannot-link 硬负证据，纯传播重叠不等同“两个人”。可选局部放大重检测增加难帧召回。可选语义反馈文件只增加有限的检测锚点、单独输出；不能凭 caption 人名直接虚构轨迹。为了保持主依赖图无环，反馈轮次作为新的运行尝试记录，不当作一次执行中的无限回路。

### 产物与消费

根 `manifest.json` 汇总各 shot；`shots/<SHOT>/manifest.json` 记录轨迹、初始化、过滤原因、cannot_link_evidence、参数和 overlay；`tracklet_signatures.json` 含 `tracklet_signatures[]`，每条保留 ID、起止时间、observation_ids、face_gallery/body_gallery、质量和 cannot_links；`tracklet_overlay.mp4` 显示框和轨迹标签；观测子目录保存图片、mask、RGBA 等，精确路径取 manifest/gallery 字段。

消费方：全局直接归属或旧式语义筛选/身份聚类。0916 未复制这些原始文件，需显式绑定已核验的 0912_v4 tracklets 根才能看中间层。

### 异常、重跑与成本

这是 GPU 与逐帧处理的主要成本点。局部重跑只选受影响 shot，新目录保存；有 continue-on-error 时要读每个 shot 状态。变更轨迹 ID/观测会使对齐和角色素材失效；不能只更新图片而沿用旧轨迹关系。

### 展示与验收

第一层为带框视频与轨迹卡，第二层为脸/身体画廊和过滤原因。显示“轨迹数量”，不用“人物数量”替代。验收标注视频与时间一致、观测来源存在、短轨迹和过滤轨迹可追踪。
<!-- /module:person_tracks -->

<!-- module:person_align -->
## 10 · 人物轨迹归属

```json pipeline-module
{
  "id":"person_align","title":"人物轨迹归属","order":100,"group":"people","lifecycle":"active","implementation":"existing",
  "summary":"结合整段带框视频与语义上下文，把每条轨迹直接归属到 global 内的角色。",
  "depends_on":["person_tracks","global_group","global_describe","shot_describe"],"optional_depends_on":["segment_describe"],"inputs":["tracklets","globals","global_captions","shot_captions","segment_captions"],
  "source_refs":["global_align_pipeline/global_tracklet_aligner.py","global_align_pipeline/prompts/global_tracklet_alignment_prompt_en_v1.md","global_align_pipeline/run_global_align_pipeline.sh"],"evidence":["code","sample"],
  "outputs":[
    {"id":"direct_alignment","title":"全局人物归属清单","kind":"jsonl","paths":{"integrated":["global_alignment/global_tracklet_alignment.jsonl"],"custom":["{output}/global_tracklet_alignment.jsonl"]},"record_key":["global_scope_id"],"timebase":"global_local","schema":"global-tracklet-character-alignment-v2","required":true},
    {"id":"alignment_evidence","title":"归属请求与带框视频","kind":"directory","paths":{"integrated":["global_alignment"],"custom":["{output}"]},"record_key":["global_scope_id"],"timebase":"global_local","schema":"mixed","required":true}
  ],
  "view":{"enabled":true,"renderer":"alignment-evidence","adapter":"global-alignment-v1","filters":["scope","subject","status"],"primary_artifact":"direct_alignment"}
}
```

### 作用与判断

新链直接建立 `tracklet → (scope,S-ID)`，避免必须先经旧链筛选、CHR 聚类再投票才能拿到素材。这是设计路径差异，不代表已证明新链准确率高于旧链。

### 输入与前置条件

轨迹根、global_clips.jsonl、shots.jsonl 和三级 caption。CLI 部分 caption 参数可不传，但语义 ID 来源必须足够明确；本文件主路径将 global/shot 上下文作为有效归属的前提。shell 入口默认源/输出目录位于仓库下，与当前真实 `../workspace/` 布局不一致，运行时须显式指定 SRC_WS/OUT_WS。

### 处理过程

按 global 的 shot 顺序拼接各镜头 overlay → 构建精简 inventory（轨迹 ID、在 global 内的区间、观测数、脸/身体证据数和最佳质量）→ 拼接 caption 证据 → MiMo 返回归属。校验 scope、轨迹 ID 是否存在/重复、S-ID 是否在允许集合。

漏返回的轨迹补为 `unknown`、置信度 0 并记录 `missing_alignment_fail_open`。允许 `new_subject_candidate` 表示可能未注册主体；它不是新的人物唯一 ID。inventory 只上传 cannot_links 数量，当前 normalize_alignment 未逐对验证身份冲突，因此不能声称新链完整继承旧聚类的 cannot-link 约束。

### 产物与消费

`global_tracklet_alignment.jsonl` 每 global 一行；关键字段 `status/tracklet_count/assigned_tracklet_count/entity_nature_counts/tracklet_alignments/missing_alignment_tracklet_ids/unmatched_semantic_subject_ids/quality_issues`。每条归属含 `tracklet_id/subject_id/entity_nature/confidence/importance_tier/visual_cues/reason`。schema 自 v2 起增加 `entity_nature`（real_person/depicted_person/non_person/uncertain，与 S-ID 指派解耦：照片/屏幕/镜面里的人物可保留所指派的 S-ID 但标记 depicted_person；缺失或非法值降级 uncertain 不触发重试），summary 含 `entity_nature_totals`。

还写 `requests/<scope>_package.json`、`stitched/<scope>_tracklets.mp4`、`alignments/<scope>.json`、`summary.json`。行内 `stitched_video` 可能仅有文件名，适配器需按本目录 stitched/ 解析。消费方：角色物化和网页证据检查。

### 异常、重跑与成本

`--dry-run` 会拼视频、写请求包，不调用模型；它不表示归属成功。`--skip-model-call` 则输出 `skip_model_call_fail_open` 并把轨迹置为 new_subject_candidate，不能计入成功归属 KPI。默认模型并发 1、rate-limit 0.5、重试 3。返回码只要求有 ok/skip-open 等情况即可为 0，混合失败需额外读 summary/per-global 状态。

### 展示与验收

主指标分开显示已归属/未知/新主体候选/模型未运行；逐角色可看轨迹证据和原因。样本 91/190 是归属覆盖率 47.89%，不是准确率。验收每条轨迹恰好一项归属，scope 正确，降级运行有显著标记。
<!-- /module:person_align -->

<!-- module:voice_preview -->
## 11 · 角色声音候选

```json pipeline-module
{
  "id":"voice_preview","title":"角色声音候选","order":110,"group":"audio","lifecycle":"active","implementation":"existing",
  "summary":"按全局 ASR 的精确区间裁出声音预览，并保留候选、拒绝和审核证据。",
  "depends_on":["global_describe"],"optional_depends_on":["voice_model"],"inputs":["attributed_asr","speaker_registry"],
  "source_refs":["assets/voice/export_all_asr_audio_review.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"voice_previews","title":"ASR 声音预览清单","kind":"jsonl","paths":{"integrated":["voice_by_role_src/all_asr_manifest.jsonl"],"custom":["{output}/all_asr_manifest.jsonl"]},"record_key":["global_id","asr_id"],"timebase":"global_local","schema":"unversioned","required":true},
    {"id":"reviewed_voice","title":"声音身份关联与审核清单","kind":"jsonl","paths":{"custom":["{output}/voice_asset_manifest.jsonl"]},"record_key":["global_id","asr_id"],"timebase":"global_local","schema":"speaker-voice-asset-v2","required":false}
  ],
  "view":{"enabled":true,"renderer":"audio-evidence","adapter":"voice-preview-v2","filters":["scope","subject","status"],"primary_artifact":"voice_previews"}
}
```

### 作用与判断

`audio_asset_candidate=true` 只表示可进入后续质量检查。这里导出所有 ASR 预览，候选放 valid、其他放 invalid，不是最终生产准入判定。

### 输入与前置条件

`--input` 为 global 的 speaker_attributed_asr.jsonl，`global_clip_path` 必须存在；可选 voice-registry/voice-registry-root 接 consolidated registry，可选 approvals 文件给已审核键。声音 registry 必须对应同一 global clip，以 metadata.pipeline_input 匹配。

### 处理过程

按 global-local 时间从视频音轨切出 16kHz、单声道 PCM s16le WAV → 计算时长/峰值/RMS/静音比例 → 记录 cut_success。存在声音 registry 时按 turn 的时间交集计算覆盖率、主说话人纯度、扣除重叠后的净单人时长，再产生 matched/ambiguous/unmatched/insufficient_evidence/timebase_mismatch/no_registry 等结果。

### 产物与消费

根 `all_asr_manifest.jsonl/valid_manifest.jsonl/invalid_manifest.jsonl/summary.json`；WAV 在 `{valid,invalid}/<scope>/<speaker>/<asr_id>.wav`。行含原 ASR 字段、`review_audio_relative_path/cut_success/cut_error/signal_stats`。这些预览行的 `review_status=valid/invalid` 是旧分类。

只有传入声音 registries 才额外生成 `voice_asset_manifest.jsonl`，其 `review_status=pending/approved` 才表达审核状态；同时保留 `voice_link.status` 与 `clean_subintervals` 等。两种同名 review_status 不能共用一个枚举解释。消费方：候选复核、新角色目录的预览复制，以及旧绑定链的正式声音准入。

### 异常、重跑与成本

切 WAV 和计算信号统计无需模型；voice_link 不重新运行声音模型。先切片失败但 manifest 仍在的行不能播放成成功。样本 72 条切片均成功；58/14 是候选分组，不是通过/未通过人工审核。

### 展示与验收

音频卡显示原句、人物归属证据、候选原因、审核状态和播放控件；默认把 valid 翻译为“候选”。验收 ASR 文字与音频时间相符、音频格式正确、审核枚举按 schema 解释。
<!-- /module:voice_preview -->

<!-- module:character_pack -->
## 12 · 角色素材汇总

```json pipeline-module
{
  "id":"character_pack","title":"角色素材汇总","order":120,"group":"characters","lifecycle":"active","implementation":"existing",
  "summary":"按 global 与 S-ID 汇集轨迹裁图和候选声音，形成便于查看的角色目录。",
  "depends_on":["person_align","person_tracks","voice_preview"],"optional_depends_on":[],"inputs":["direct_alignment","tracklets","voice_previews"],
  "source_refs":["global_align_pipeline/global_character_materializer.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"characters","title":"角色素材目录","kind":"directory","paths":{"integrated":["characters"],"custom":["{output}/characters"]},"record_key":["semantic_scope_id","subject_id"],"timebase":"none","schema":"global-character-materialization-v2","required":true},
    {"id":"character_summary","title":"角色物化统计","kind":"json","paths":{"integrated":["materialization_summary.json"],"custom":["{output}/materialization_summary.json"]},"record_key":[],"timebase":"none","schema":"global-character-materialization-v2","required":true}
  ],
  "view":{"enabled":true,"renderer":"character-gallery","adapter":"characters-global-v1","filters":["scope","subject","status"],"primary_artifact":"characters"}
}
```

### 作用与判断

物化是“根据已有归属挑选并复制素材”，不会再次证明人物身份，也不会新增声音审核。样本 16 个角色目录是作用域内身份数，不是整部电影的 16 位不同人物。

### 输入与前置条件

归属 JSONL、原始 tracklets、voice_by_role 根与新的输出目录。当前实现启动时会 `shutil.rmtree(output/characters)` 后重建；未来 Agent 不应把已有人工整理目录直接作为输出。

### 处理过程

只接受归属行 status 为 ok 或 skip_model_call_fail_open；跳过 unknown；按 min-confidence（默认 0）过滤，按 subject 聚合轨迹；face/body 默认各选质量最高 8 张，按 observation 去重并复制 RGBA；复制 valid bucket 音频，开关可包含 invalid。

当前实现没有排除 new_subject_candidate，会把这一哨兵名称下不同未确认轨迹聚到同一目录。这是“未注册候选集合”，不能被网页渲染为已确认的一个人，也不能把复制的 valid 预览称为已审核净语音。

两个可选守护（2026-09-21，schema 升至 v2）：`--identity-check` 用 Stage3 同款 InsightFace（buffalo_l）对同一 (scope,S-ID) 内各轨迹 top-3 人脸算平均 embedding，两两 cosine 建图（阈值默认 0.40），最大连通分量（MiMo 置信度加权破平）为身份核，核外轨迹判为 outlier——默认仅剔除其 face crop（`--exclude-outlier-visual` 连 body/rgba 一起剔除），结果写 role_character.json 的 identity_check（含相似度矩阵）与根目录 identity_conflicts.json；模型/图片不可用时 fail-open 跳过，全体互不相似标 inconclusive 并保留，1v1 无法判定也进人工复核清单。`entity_nature` 门控：depicted_person/non_person 轨迹不进资产库、只按其指派 S-ID 记录在 role_character.json 的 depicted_references（该 S-ID 只有 depicted 轨迹时不建目录）；uncertain 默认照常物化，`--strict-entity-nature` 收紧。skip_model_call 兜底行无 entity_nature，按 uncertain 处理。

### 产物与消费

`characters/<scope>/<S-ID>/visual/{face,body,rgba}/`、`voice/valid/`（可选 invalid）、`role_character.json`。后者含 `semantic_scope_id/subject_id/tracklet_count/tracklet_ids/mean_alignment_confidence/entity_nature_counts/depicted_references/identity_check/visual_stats/voice_stats/source_alignment`；根 materialization_summary 给 character_count、assigned_tracklet_count、entity_nature_totals、identity_outlier_tracklet_count 和每角色复制数量，identity_conflicts.json 列出 outlier 与待复核冲突。消费方：展示、人工挑选和外部生成引用。

### 异常、重跑与成本

主要为 I/O，缺源图计 missing_files；低置信度门默认宽松。重跑会清空角色输出，必须用新目录验证再决定发布。meta 中没有逐张审核证明，不能用 mean confidence 代表所有素材可靠。

### 展示与验收

角色卡明确 scope、语义描述、轨迹证据、脸/身体/声音数量，点击返回来源；未知候选集合单独显示。验收复制数与文件数、来源可读、音频候选标签和破坏性重跑说明。
<!-- /module:character_pack -->

<!-- module:object_queue -->
## 13 · 关键物体候选

```json pipeline-module
{
  "id":"object_queue","title":"关键物体候选","order":130,"group":"objects","lifecycle":"active","implementation":"existing",
  "summary":"把镜头描述中的重要物体整理为带可见时间窗的定位候选。",
  "depends_on":["shot_describe"],"optional_depends_on":["legacy_refine"],"inputs":["shot_captions","semantic_tracklet_views"],
  "source_refs":["assets/key_object/build_object_grounding_queue_v1.py","assets/key_object/build_object_grounding_queue_v2.py","assets/visual_person/semantic_role_tracklet_refiner_v3.py","build_object_grounding_queue_from_shot_captions_v1.py"],"evidence":["code","sample"],
  "outputs":[{"id":"object_candidates","title":"物体定位队列","kind":"jsonl","paths":{"integrated":["key_object/grounding_queue.jsonl"],"custom":["{output_file}"]},"record_key":["candidate_id"],"timebase":"shot_local","schema":"object-grounding-candidates-v2","required":true}],
  "view":{"enabled":true,"renderer":"record-table","adapter":"object-queue-v2","filters":["scope","shot","status"],"primary_artifact":"object_candidates"}
}
```

### 作用与判断

定位模型需要具体对象短语和出现位置，不能直接把一段故事描述当定位指令。候选阶段保留“未验证”，不承诺图像中真的有该对象。

### 输入与前置条件

默认直接读取成功的 shot caption；按 priorities 选择 must/high（默认）。另一路来源是 legacy_refine 的物体提案，必须保留 source 字段，不能不加区分合并重复候选。

### 处理过程

将镜头号和 local_object_id 拼成 candidate_id；投影 canonical_name、grounding_phrases、属性、关联人物、状态、优先级；V2 专门保留 shot-relative visible_time_range_sec。直接队列可能 semantic_scope_id 为 null，后续通过 shot→global 清单关联。

### 产物与消费

一行一个候选，常见 ID `SHOT_772_OC1`；包含 `candidate_id/shot_id/canonical_name/grounding_phrases/priority/related_semantic_subject_ids/visual_attributes/proposal_confidence/verification_status/source/visible_time_range_sec`。`source_caption_schema_version` 保留来源。消费方为 object_ground 和 I2I bundle。

### 异常、重跑与成本

这是纯数据变换，无需 GPU。caption 失败不产候选；候选为空不代表物体模型失败。不同物体候选来源需要去重规则和证据，不能仅凭同名 canonical_name 合并不同实例。

### 展示与验收

表格显示对象、镜头、可见窗、提出原因和验证状态；跳到对应时窗播放。验收候选 ID 唯一、源 refs 正确、可见窗有效。样本 25 个候选。
<!-- /module:object_queue -->

<!-- module:object_ground -->
## 14 · 关键物体定位与分割

```json pipeline-module
{
  "id":"object_ground","title":"关键物体定位与分割","order":140,"group":"objects","lifecycle":"active","implementation":"existing",
  "summary":"在候选可见区间内验证物体实例，产出裁图、透明前景和可追踪的多帧证据。",
  "depends_on":["object_queue","shot_split"],"optional_depends_on":[],"inputs":["object_candidates","shot_media","shots"],
  "source_refs":["assets/key_object/object_grounding_segmenter_v1.py","assets/key_object/object_grounding_segmenter_v2.py","assets/key_object/object_grounding_segmenter_v3.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"object_observations","title":"物体观测清单","kind":"jsonl","paths":{"integrated":["key_object/grounding/object_observations.jsonl"],"custom":["{output}/object_observations.jsonl"]},"record_key":["candidate_id","shot_id","frame_index"],"timebase":"shot_local","schema":"mixed","required":true},
    {"id":"object_references","title":"物体多帧参考集","kind":"json","paths":{"integrated":["key_object/grounding/reference_sets.json"],"custom":["{output}/reference_sets.json"]},"record_key":["candidate_id"],"timebase":"shot_local","schema":"mixed","required":false}
  ],
  "view":{"enabled":true,"renderer":"object-evidence","adapter":"object-observations-v3-v4","filters":["scope","shot","status"],"primary_artifact":"object_observations"}
}
```

### 作用与判断

GroundingDINO 给文本对象候选框，SAM3.1 按几何提示分割；还要判断多个检测是否指向同一目标。检测成功、分割成功、实例无歧义是三个不同判断。

### 输入与前置条件

定位队列、镜头视频/清单和本地 GroundingDINO、SAM3.1 权重。精确 CLI 以 v3 文件的 parser 为准；文件名 v3 当前已包含 V4 扩展，不能从后缀推断 schema。

### 处理过程

在有效 visible_time_range_sec 内采样 → 多短语检测 → 合并重复框/短语票数 → 实例选择 → 分割 → 保存素材与诊断。当前 V4 默认在失败/歧义时于同一可见窗内追加有限清晰帧，按跨帧连续性和检测证据裁决；不改候选窗口和优先级，低一致性保留歧义。`--window-resample-max 0` 回退 V3 行为。

独立 `--build-reference-sets` 后处理读取既有观测，使用 DINOv3 前景 embedding 选有序多帧参考集，目标约 3–5 张；不重新检测/分割。不足时应报告真实数量和原因，不伪造补足。

### 产物与消费

`object_observations.jsonl` 一行一个候选/帧观测，保留 `status/ambiguous_instance/selected_bbox_xyxy/selected_grounding_score/segmentation_backend/mask_area_ratio/assets` 和检测簇/选择理由。V4 增加 sampling_pass、frame_sharpness、resample、cross_frame_support 等；样本实际为 `object-observation-v3`，当前默认实现可能写 V4。

`assets` 指向 `frames/masks/crops/rgba/overlays/` 对应文件；根 summary 汇总，选配 reference_sets.json 关联 candidate_id。消费方：物体检查、I2I 物体包和候选诊断。

### 异常、重跑与成本

在小窗口抽帧可降低 GPU 成本。禁止把 `status=ok` 直接映射成 accepted：0916 的 125 个 ok 观测中 67 个仍标注实例歧义。重复采样同帧时适配器需保留 sampling_pass/序号，避免复合键覆盖。旧输出与当前 V4 需按 schema 分支解释。

### 展示与验收

默认看对象卡与“原帧/框/掩码/透明前景”对比，醒目标出歧义、可见窗和跨帧支持；展开模型分数及失败理由。验收来源帧能定位、bbox/mask 尺寸一致、歧义不计入可用资产。
<!-- /module:object_ground -->

<!-- module:scene_select -->
## 15 · 场景参考图

```json pipeline-module
{
  "id":"scene_select","title":"场景参考图","order":150,"group":"scenes","lifecycle":"active","implementation":"existing",
  "summary":"发现视觉场景，选择遮挡较少、结构清晰且互补的参考帧。",
  "depends_on":["global_group"],"optional_depends_on":[],"inputs":["globals","global_media"],
  "source_refs":["assets/scene/important_scene_frame_extractor.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"scene_assets","title":"场景参考与评分","kind":"directory","paths":{"integrated":["scene"],"custom":["{output}"]},"record_key":["video_id","scene_name"],"timebase":"global_local","schema":"important-scene-index-v3","required":true},
    {"id":"scene_variant","title":"场景参考优化对照","kind":"directory","paths":{"integrated":["scene_opt_marginal"],"custom":["{variant_output}"]},"record_key":["video_id","scene_name"],"timebase":"global_local","schema":"important-scene-index-v3","required":false}
  ],
  "view":{"enabled":true,"renderer":"scene-gallery","adapter":"scene-index-v3","filters":["scope","variant"],"primary_artifact":"scene_assets"}
}
```

### 作用与判断

为场景生成/复核选参考图，优先保留环境结构、减少人物遮挡。纯视觉流程不依赖 MiMo caption；时间不相邻但视觉相似的帧可能属于同一场景簇。

### 输入与前置条件

视频列表/目录、输出根，DINOv3 与 YOLO 权重。通常输入 global 视频，若改为整片输入，时间基准随之改变，不能沿用 global_local 绑定。当前展示 profile 假设输入为 global。

### 处理过程

默认约 1.2 秒采样、候选数设上下限 → YOLO 检测人物并可用 tiled inference → 计算清晰度/曝光/结构遮挡等分数 → DINOv3 场景聚类 → 场景重要性排序 → 主参考和同场景辅助帧。默认每场景目标 3 帧、最多 12 场景，实际受候选和约束限制。

`aux_selection=marginal` 的辅助选择关注对已选集合的边际收益，保留 selection_strategy/max_sim_to_selected 等额外字段；与默认版本分开输出，不能把“优化”目录名称当提升质量的证明。

2026-09-21 新增三个 opt-in 选项（默认行为不变）：`--shots-manifest` 传入 shots 清单后采样时间戳吸附最近镜头边界（±0.6s，吸附计数写入 run_manifest）；`--max-scenes auto` 在 2..12 网格上按 cosine silhouette 自动定场景数（各 k 分数写入 scene_index 的 scene_discovery.selection）；`--anchor-selection coverage` 用 k-center 贪心（facility-location 次模覆盖的贪婪近似）选主/辅锚点以保证视角多样性，会覆盖 aux 选择策略。三项均未在服务器实测。

### 产物与消费

根 `run_manifest.json`；每 `<video>/` 下 `scene_index.json/candidate_scores.json/all_scenes_preview.jpg`；每 `scene_NNN/` 下 `scene_meta.json/anchor_main.jpg/anchor_aux_*.jpg/person_mask_main.png/anchor_main_debug.jpg/topk_preview.jpg`。index 的 scenes 含 sample_count、returned_frame_count、relation_constraint_satisfied、selected 的时间/分数/人物占比。消费方为 I2I 场景包和场景浏览。

### 异常、重跑与成本

GPU 特征提取与检测是主要成本；仅改变选帧策略是否能复用特征以脚本实际支持为准，不承诺未实现缓存。`--overwrite` 会影响已有输出。返回帧不足必须解释为约束/候选不足，不能显示为固定三张。

### 展示与验收

按场景展示主图和辅助图，支持默认与 marginal 同位置对比；标出时刻、遮挡比例和“辅助帧同场景约束”。验收参考图可定位、聚类数量与实际一致，不把 frame_score 当审美真值。
<!-- /module:scene_select -->

## 支线、扩展与外部模块（不在主线讲解路径中）

以下模块不属于主线：或是最新 15 阶段主线之外的可选/外部/基础设施步骤，或是明确降为支线的历史流程。它们仍被完整收录（模块元数据、六段说明与产物契约不变），供对照、复跑与兼容展示，但默认不出现在主线讲解路线与首屏导航中；网站里它们位于最末分组，需打开“显示兼容/实验模块”或在系统介绍中选择对应路线才能看到。

- 扩展与外部模块（主线之外）：可选预处理（source_prepare）、图像生成任务包（generation_bundle）、图像/语音生成效果对比（i2i_review / tts_review，外部工具）、结果文件与服务（result_browse，展示基础设施，含旧 asset_viewer）。
- 声音增强支路（voice_model）：全局视频 → 前景分离 / 重叠检测 / 说话人建模 → 净语音与审核证据。可独立运行，产出 SPK 身份；不依赖视觉人物归属，也不会自动得到 S-ID。
- 旧 CHR 人物链（legacy_refine → legacy_identity → legacy_gallery → legacy_bind）：逐镜头 LLM 语义标注 → 跨镜头身份聚类 → 视觉画廊 → 投票绑定到 S-ID。与新链（轨迹直接全局归属）是替代/对照关系，不是必经步骤。
- 兼容与实验支路（compat_structure、asset_demo）：旧 ClipRecord 工作区结构与历史 real/stub 资产演示。

对应讲解路线：legacy-chr（CHR 聚类与角色映射）、audio-evidence（前景声音与身份验证）。

<!-- module:source_prepare -->
## 16 · 输入与预处理

```json pipeline-module
{
  "id":"source_prepare","title":"输入与预处理","order":160,"group":"compatibility","lifecycle":"optional","implementation":"existing",
  "summary":"确认源视频及时间坐标，按需统一降采样、移除片头片尾低内容区间。",
  "depends_on":[],"optional_depends_on":[],"inputs":[],
  "source_refs":["preprocessing/preprocess.py","preprocessing/normalize_video.py","run_full_pipeline.py"],"evidence":["code"],
  "outputs":[
    {"id":"source_video","title":"切镜输入视频","kind":"media","paths":{"custom":["{source_video}"]},"record_key":[],"timebase":"source","schema":"unversioned","required":true},
    {"id":"preprocess_info","title":"预处理区间报告","kind":"json","paths":{"custom":["{output}/processed_info.json"]},"record_key":[],"timebase":"source","schema":"unversioned","required":false},
    {"id":"normalized_video","title":"统一降采样视频","kind":"media","paths":{"custom":["{output}/normalized/normalized.mp4"]},"record_key":[],"timebase":"source","schema":"unversioned","required":false},
    {"id":"normalize_info","title":"降采样参数报告","kind":"json","paths":{"custom":["{output}/normalized/normalized.mp4.info.json"]},"record_key":[],"timebase":"none","schema":"unversioned","required":false}
  ],
  "view":{"enabled":true,"renderer":"source-overview","adapter":"source-v1","filters":[],"primary_artifact":"source_video"}
}
```

### 作用与判断

输入登记是概念上的必需步骤，执行裁剪是可选动作。`preprocessing/preprocess.py` 用低亮度、低音量和相邻采样帧平均亮度变化判断候选丢弃区间；它不是语义片头/字幕识别器，静态剧情也可能被命中。

2026-09-21 新增统一降采样：`preprocessing/normalize_video.py`（`scale=-2:480,fps=15`、crf 20、aac 128k，写 `<output>.info.json` 记录输入/输出探测参数与命令行）。run_full_pipeline 编排默认把它作为第一阶段 `normalize_video`（`--no-normalize` 关闭、`--norm-height/--norm-fps` 可调），切镜与场景提取改吃 normalized.mp4；目的在于进入 MiMo 标注与检测前统一分辨率/帧率、显著缩小请求与推理成本。

### 输入与前置条件

输入本地视频，依赖 ffmpeg/ffprobe、numpy。先确认视频/音轨可读取，以及后续究竟使用原片还是处理后视频。裁剪链不会被两套 caption 编排的 `all` 自动执行；降采样链只被 run_full_pipeline 默认执行（基础六步编排仍直接吃原片）。

### 处理过程

默认 2 fps 稀疏采样、亮度阈值 8、静态变化阈值 1.5、音量 RMS 阈值 0.005；只接受前 180 秒、后 240 秒范围内的候选区间，合并相邻 drop，求保留区间，再切割拼接。实际 CLI 默认值优先于旧 docstring 示例。保留区间可以不止一个。

### 产物与消费

`processed_info.json` 记录 `source/duration_sec/trimmed_sec/kept_intervals/drops/processed_video`；处理视频路径以 `processed_video` 为准。没有裁剪时可直接引用原片。`source_video` 被切镜模块消费；区间报告用于原片与处理后时间互译及网页解释。降采样链产出 `normalized/normalized.mp4` 与同名 `.info.json`（输入/输出宽高、fps、时长、完整命令行），被 run_full_pipeline 的切镜与场景阶段消费；注意降采样改变帧率后 shot 的 `frame_range` 语义随 normalized 时间轴变化，与原片帧号不可混用。

### 异常、重跑与成本

这是 CPU 解码与 I/O 工作；stream-copy 切割不应被描述为逐帧精确剪辑。无音轨等输入异常需要预检。改变预处理会改变后续全部 source 时间和切片，不可继续无条件复用旧描述/资产。

### 展示与验收

展示源片信息、处理前后时长和丢弃时间条；可点击看丢弃区间。未执行显示“使用原片”，不显示失败。验收时间映射、保留区间有序不重叠和输出可读。
<!-- /module:source_prepare -->

<!-- module:generation_bundle -->
## 17 · 图像生成任务包

```json pipeline-module
{
  "id":"generation_bundle","title":"图像生成任务包","order":170,"group":"compatibility","lifecycle":"optional","implementation":"existing",
  "summary":"将片段描述、场景和物体参考整理成可复核的 I2I 输入包。",
  "depends_on":["segment_describe"],"optional_depends_on":["scene_select","object_queue","object_ground","legacy_bind","character_pack"],"inputs":["segment_captions","scene_assets","object_candidates","object_observations","object_references","role_binding","characters"],
  "source_refs":["assets/character_role_aligner.py"],"evidence":["code","sample"],
  "outputs":[
    {"id":"scene_segment_index","title":"场景与片段映射","kind":"json","paths":{"integrated":["scene/scene_segment_index.json"],"custom":["{output}/assets/scene/scene_segment_index.json"]},"record_key":["global_id","segment_id"],"timebase":"mixed","schema":"mixed","required":true},
    {"id":"i2i_bundles","title":"I2I 任务包","kind":"directory","paths":{"integrated":["i2i_bundles"],"custom":["{bundle_root}"]},"record_key":["task_id"],"timebase":"mixed","schema":"mixed","required":true}
  ],
  "view":{"enabled":true,"renderer":"generation-bundle","adapter":"i2i-bundle-v1","filters":["scope","segment","variant","status"],"primary_artifact":"i2i_bundles"}
}
```

### 作用与判断

任务包把上游产物与生成意图连接起来，包括证据是否足够；创建包不等于已经生成图片，也不等于通过质量审核。

### 输入与前置条件

`character_role_aligner.py --export-i2i-bundles <workspace>` 是独立分支，不运行前面的角色投票。其 load_i2i_inputs 仍假设特定旧布局，例如 `assets/key_object/...`、`global/global_clips.jsonl`、`assets/role_alignment.json`；0916 是整理后布局，不能直接认定此目录可作为原样重导出输入。须建立明确绑定/转换。

### 处理过程

读取 segment 描述与时间/shot → 将场景参考归到段 → 汇集人物和关键对象上下文 → 生成 scene/object 任务 prompt、references、review_assets、entity_index 和 provenance → 计算 completeness。人物/场景/物体缺失应保留缺失原因；新链 characters 是否被当前导出器完整支持需额外接入验证，当前主要关联仍是旧 role_to_chr。

### 产物与消费

`scene_segment_index.json` 关联段和场景，按 globals[].segments[] 展开后使用 global_id+segment_id，完整来源还包括视频。原子导出的真实默认位置是 `<output_root>/assets/scene/scene_segment_index.json` 与 `<output_root>/assets/i2i_bundles/{scene,object}/`；custom 的 bundle_root 应绑定后一个目录，0916 则已整理到顶层。

任务包包含 `schema_version/task_id/task_type/target/text_context/references/review_assets/entity_index/prompt/provenance/completeness`。样本 `i2i_bundles/scene/scene_global_0040_segment_001.json` 与 `object/object_SHOT_772_OC1.json` 分别代表段场景和物体实例；样本 12 个场景包、25 个物体包。消费方是外部 I2I 执行器及展示器，不由此步骤调用图像生成模型。导出汇总作为函数结果/CLI 输出提供，不应假定另有固定 summary.json 文件。

### 异常、重跑与成本

通常以 JSON 组织与图像元信息读取为主。上游更新后通过 provenance 判断包是否过期；不要沿用旧参考图却只更新 prompt。缺合适参考时保留 incomplete，比伪造图像更可追溯。

### 展示与验收

先展示生成目标、参考图和完整度，再展开 prompt 与来源。验收同一包引用同一来源范围、refs 可解析、缺失原因可见。网页不得在查看包时自动触发收费生成。
<!-- /module:generation_bundle -->

<!-- module:i2i_review -->
## 18 · 图像生成效果对比

```json pipeline-module
{
  "id":"i2i_review","title":"图像生成效果对比","order":180,"group":"compatibility","lifecycle":"external","implementation":"external",
  "summary":"对比同一参考和目标下不同图像模型的输出，保留输入与参数。",
  "depends_on":[],"optional_depends_on":["generation_bundle","legacy_gallery"],"inputs":["i2i_bundles","chr_gallery"],
  "source_refs":["asset_viewer/pipeline_router.py"],"evidence":["sample","code"],
  "outputs":[{"id":"i2i_results","title":"图像生成测试结果","kind":"directory","paths":{"integrated":["i2i_test"],"custom":["{output}"]},"record_key":["model","task_id"],"timebase":"none","schema":"unversioned","required":true}],
  "view":{"enabled":true,"renderer":"image-comparison","adapter":"i2i-test-v1","filters":["variant"],"primary_artifact":"i2i_results"}
}
```

### 作用与判断

这是系统产物的下游应用实验。外部生成工具不在本仓库核心 pipeline 内；本次核验其落盘形式和 viewer 读取逻辑，没有声称审查了所有生成模型执行代码。

### 输入与前置条件

参考图、prompt、模型参数；可能来自任务包，也可能直接来自 CHR 画廊。当前样本为 CHR004 的衣服/裤子实验，不能包装成所有场景/物体包已经执行完毕。

### 处理过程

由外部工具生成图片并保存响应元信息；展示层按任务目标分组、按模型并列。比较时必须对齐参考与 prompt；不一致时显示“条件不同”，不能直接排名。

### 产物与消费

样本 `i2i_test/<model>/*.png` 与同名 `*.png.json`，根含参考 JPG。路径/sidecar 由适配器读取。消费方仅为效果浏览和人工评估。

### 异常、重跑与成本

没有 sidecar 时显示“参数未收录”，不要猜模型和 prompt。生成成本由外部执行器控制，页面浏览不产生调用。新增模型文件夹不应要求改整页代码。

### 展示与验收

同一任务固定参考图置顶，多模型图片并列、点击放大；显示模型与输入差异。验收每个结果都能找到实际参考或明确缺失，避免跨角色错配。
<!-- /module:i2i_review -->

<!-- module:tts_review -->
## 19 · 语音合成效果对比

```json pipeline-module
{
  "id":"tts_review","title":"语音合成效果对比","order":190,"group":"compatibility","lifecycle":"external","implementation":"external",
  "summary":"对比参考声音和多个 TTS 模型的合成结果及文本。",
  "depends_on":[],"optional_depends_on":["voice_preview","voice_model","character_pack"],"inputs":["voice_previews","speaker_registry","characters"],
  "source_refs":["asset_viewer/pipeline_router.py"],"evidence":["sample","code"],
  "outputs":[{"id":"tts_results","title":"TTS 测试结果","kind":"directory","paths":{"integrated":["tts_test/outputs"],"custom":["{output}"]},"record_key":["model","scope","speaker"],"timebase":"none","schema":"unversioned","required":true}],
  "view":{"enabled":true,"renderer":"audio-comparison","adapter":"tts-test-v1","filters":["scope","subject","variant"],"primary_artifact":"tts_results"}
}
```

### 作用与判断

TTS 对比评估的是参考条件下的合成效果，不自动验证上游身份正确，也不表示候选声音已获准用于正式资产。

### 输入与前置条件

参考音频、参考文本、合成文本和模型配置；由仓库外测试工具运行。当前 viewer 按 `*/*/meta.json` 发现结果。

### 处理过程

外部执行器保存合成音频/元数据 → 适配器按 scope、speaker 和模型分组 → 浏览器提供对比播放。若参考和合成文本不同，分别展示，禁止误标成同文本复刻。

### 产物与消费

`tts_test/outputs/<model>/<scope>_<speaker>/meta.json` 及实际引用音频；样本有 voxcpm2、s2pro 目录，文件细节以 meta 和 viewer 解析为准。消费方为结果对比页面。

### 异常、重跑与成本

缺音频、合成失败、参数未记录分别显示。运行耗时/质量评分没有采集就显示“未记录”，不能虚构。浏览不触发合成。

### 展示与验收

统一的参考音频 + 模型结果列表，切换播放自动暂停上一条。验收引用正确、重复 S-ID 不跨 scope 合并、播放有文字标签，键盘可操作。
<!-- /module:tts_review -->

<!-- module:result_browse -->
## 20 · 结果文件与服务

```json pipeline-module
{
  "id":"result_browse","title":"结果文件与服务","order":200,"group":"compatibility","lifecycle":"active","implementation":"existing",
  "summary":"读取已落盘结果，为浏览器提供目录、详情和媒体播放。",
  "depends_on":[],"optional_depends_on":["segment_export","character_pack","legacy_bind","object_ground","scene_select","i2i_review","tts_review"],"inputs":["segment_media","characters","legacy_characters","object_observations","scene_assets","i2i_results","tts_results"],
  "source_refs":["asset_viewer/server.py","asset_viewer/pipeline_router.py","asset_viewer/parsers.py","asset_viewer/html_template.py"],"evidence":["code"],
  "outputs":[{"id":"workspace_artifacts","title":"可浏览产物目录","kind":"directory","paths":{"integrated":["."],"basic-flat":["."],"scheme-a":["."],"custom":["{workspace}"]},"record_key":[],"timebase":"none","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"artifact-browser","adapter":"workspace-artifacts-v1","filters":["status"],"primary_artifact":"workspace_artifacts"}
}
```

### 作用与判断

现有 asset_viewer 用标准库 ThreadingHTTPServer + Python 内嵌 HTML/JS 提供只读浏览。它只读结果，不参与产生轨迹或 caption，也没有本文档驱动导航/热加载能力。

### 输入与前置条件

工作区根、默认/唯一工作区、端口和平台 base-url。当前自动发现要求目录名前缀 `movie_full_pipeline_`，且 detect_chain 能识别新对齐或旧资产库；纯 caption 工作区可能被过滤掉。默认 CLI 端口 8910、host 0.0.0.0；实际暴露范围由启动参数决定，常量 DEFAULT_WORKSPACE 不等于 CLI 强制只暴露该目录。

### 处理过程

发现工作区 → 根据固定路径检查 stage → 枚举实体/解析详情 → 前端根据硬编码 stage 分支渲染。支持工作区、阶段、scope、subject 选择和永久链接；媒体端点实现 Range 读取。工作区注册表启动后缓存，不是实时监听文档/产物的机制。

### 产物与消费

API 包括 `/api/workspaces`、`/api/stages`、`/api/entities`、`/api/character`、`/api/global`、`/api/tracklet`、`/api/global_clip`、`/api/scene`、`/api/key_object`、`/api/i2i_bundle`、`/api/detail`、`/api/artifacts`、`/api/artifact`，媒体 `/media/path`；当前没有额外写入业务产物。

### 异常、重跑与成本

现有 README 称 resolve 后限制在 workspace 内；实际 `_safe_within_workspace` 使用 abspath 做词法范围检查以允许跨工作区软链接，二者不一致。新设计需在受控根内保留该使用场景，见文档三。读取坏 JSON/媒体缺失应显示局部错误，不能用空列表伪装成功。

### 展示与验收

沿用服务器监听和平台代理思路，升级为文档索引 + 布局适配器 + 渲染器注册制。部署和新 API 详见文档三；本次没有启动服务或验证平台网络可达。artifact-browser 仅列入已登记的结果目录，不递归暴露模型权重、日志秘密或代码配置。
<!-- /module:result_browse -->

<!-- module:voice_model -->
## 21 · 声音身份与净语音证据

```json pipeline-module
{
  "id":"voice_model","title":"声音身份与净语音证据","order":210,"group":"compatibility","lifecycle":"optional","implementation":"existing",
  "summary":"从 global 音频建立说话人时间线、重叠区间和可复核的声音参考。",
  "depends_on":["global_group"],"optional_depends_on":[],"inputs":["globals","global_media"],
  "source_refs":["assets/voice/speaker_voice_asset_pipeline.py","assets/voice/audio_foreground_separator.py","assets/voice/speaker_overlap_detector.py","assets/voice/speaker_registry_builder.py","assets/voice/speaker_turn_foreground_gate.py","assets/voice/cocktail_asr.py"],"evidence":["code"],
  "outputs":[{"id":"speaker_registry","title":"声音建模产物","kind":"directory","paths":{"custom":["{output}"]},"record_key":["speaker_id"],"timebase":"global_local","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"audio-evidence","adapter":"speaker-registry-v1","filters":["scope","subject","status"],"primary_artifact":"speaker_registry"}
}
```

### 作用与判断

音频主体是 SPK，判断依据为声音和时间，不自动知道画面中的 S-ID。增强链提供去噪/前景、说话人分段、重叠排除和参考音频身份证据，供后续音画关联。

### 输入与前置条件

入口 `speaker_voice_asset_pipeline.py run`，输入一个视频、独立输出根；需要 ClearVoice/MossFormer2_SE、说话人分离/embedding、重叠检测等本地模型。可以传 VAD/diarization/ASR JSON 复用已有结果。此链不依赖 caption，因此可与视觉并行。

### 处理过程

前景人声分离 → 重叠检测 → 在前景 stem 上建立 speech windows、局部 speaker tracklets 和 turn → CAM++ 身份特征/约束聚类 → foreground gate → 收集 speaker pools、合并证据 → 排除 simultaneous speech → 生成 consolidated 参考和审计。可选双路径验证比较增强声轨与原混音的身份；默认 shadow 记录与启用 enforce 拒绝是不同语义。

当前未提交的 builder 已用 CocktailASR 替换原 Transcriber，默认模型路径改为 `local_models/Xiaomi-CocktailASR-1`。自动参考属于目标说话人转写，不能证明覆盖所有人；需显式时间分段/可靠 ASR 输入，才能支撑句级音频。保留 `--asr-json` 的历史参数名称不表示仍运行 Whisper。

2026-09-21 起 `cocktail_asr.py` 提供多参考多说话人转写（官方无多说话人版本，利用目标说话人 ASR 的"目标缺席输出空文本"拒识特性）：`transcribe_multi_speaker`/CLI `--ref S1=path --ref S2=path`（或 `--ref-list`）。共享能量分段时间轴，每条参考音纹跑一遍目标转写，空文本段丢弃，跨说话人按时间 IoU>0.7 且文本前缀相同去重；输出行带 `speaker_id/matched_ref`，另含 dropped_duplicates 与逐参考计数。旧单参考模式行为不变；`test_cocktail_asr.py --via multi` 可验收。参考音纹建议取本链 consolidated registry 的参考音频。重叠语音下的拒识表现未实测。

### 产物与消费

增强目录的 `foreground_report.json` 及前景/原声路径；`overlap/overlap_report.json`；`stage5/registry.json`、`registry.pregate.json`、`registry.gate.json` 和 timeline、speech_windows、local_speaker_tracklets、speaker_turns、asr_timeline、global_speaker_timeline、overlap_regions、must/possible/cannot_links、unknown_turns 等 JSON。

`consolidated/` 包含 `registry.json/voice_asset_library.json/consolidation.json/speaker_turns.json/global_speaker_timeline.json/simultaneous_speech_regions.json` 等，具体参考 WAV 由 registry 的 reference_audio 等字段指向；根 `compare.json/summary.json` 汇总。消费方：voice_preview 的可选 voice_link、旧角色绑定的审核音频和声音证据页面。0916 整合目录没有这整套产物，不能自动显示为已完成。

### 异常、重跑与成本

这是较重 GPU/音频模型链；关 ASR、关重叠排除、复用 JSON 都会改变质量含义，必须展示配置。无合格语音证据允许不输出正式参考。模型/路径错误不能降格解释成“该角色没有声音”。

### 展示与验收

并排听原音、前景、净语音，画出 turn 与重叠条，显示参考录音准入原因。验收时基一致、SPK 不被误当 S-ID、被排除区间与净语音时长可追溯。
<!-- /module:voice_model -->

<!-- module:legacy_refine -->
## 22 · 逐镜头人物语义筛选

```json pipeline-module
{
  "id":"legacy_refine","title":"逐镜头人物语义筛选","order":220,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"为旧 CHR 链给轨迹附加 S-ID、叙事重要性与筛选策略。",
  "depends_on":["person_tracks","global_describe"],"optional_depends_on":["shot_describe","segment_describe"],"inputs":["tracklets","global_captions","shot_captions","segment_captions"],
  "source_refs":["assets/visual_person/semantic_role_tracklet_refiner.py","assets/visual_person/semantic_role_tracklet_refiner_v2.py","assets/visual_person/semantic_role_tracklet_refiner_v3.py"],"evidence":["code","sample"],
  "outputs":[{"id":"semantic_tracklet_views","title":"语义筛选与策略视图","kind":"directory","paths":{"custom":["{output}"]},"record_key":["shot_id","tracklet_id"],"timebase":"shot_local","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"alignment-evidence","adapter":"semantic-tracklets-v3","filters":["scope","shot","variant","status"],"primary_artifact":"semantic_tracklet_views"}
}
```

### 作用与判断

对每个 shot 让 MiMo 对齐语义角色，输出 keep_primary/keep_supporting/keep_context/reject/review 等决策；只在旧链中作为身份聚类的上游筛选，不是新全局链的必经步骤。

### 输入与前置条件

读取 tracklet 签名和 caption 上下文，保留原始轨迹证据；选择 primary、narrative 或 safe 策略时必须记录具体策略。

### 处理过程

打包镜头与轨迹 → 模型判断 → 校验角色集合/重要性/拒绝置信度 → 低证据拒绝改为 review → 输出不同保留视图。safe 是仅排除高置信背景/非人物的保守视图；primary 会显著减少进入下一步的轨迹。V2 增加物体提案，V3 保留其可见时间窗到 grounding 候选。

### 产物与消费

`alignment_results.jsonl`、`alignments/<SHOT>.json`、`semantic_tracklet_index.json`、`manifest.json`；`stage3_inputs/{primary,narrative,safe}/shots/<SHOT>/tracklet_signatures.json` 和 `semantic_identity_decisions.json`；V2/V3 另有 `object_proposals.jsonl/object_grounding_candidates.jsonl`。消费方为 legacy_identity；物体提案可以作为另一路 grounding 来源，必须标注与直接 shot queue 的区别。

### 异常、重跑与成本

此处按 shot 调用 VLM，成本随镜头数增加。失败保守保留不等于身份已确认。改变策略要单独重建其后 CHR 结果；不能混用 primary registry 和 narrative gallery。

### 展示与验收

展示输入轨迹数、各策略保留数和拒绝原因，支持看被过滤轨迹。验收每个视图的来源/策略一致，筛除数可解释。
<!-- /module:legacy_refine -->

<!-- module:legacy_identity -->
## 23 · 跨镜头身份聚类

```json pipeline-module
{
  "id":"legacy_identity","title":"跨镜头身份聚类","order":230,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"使用脸部和身体证据构建可追溯的视觉身份图，保留不确定候选。",
  "depends_on":["legacy_refine"],"optional_depends_on":[],"inputs":["semantic_tracklet_views"],
  "source_refs":["assets/visual_person/visual_character_registry_builder.py"],"evidence":["code","sample"],
  "outputs":[{"id":"visual_identity","title":"视觉身份图与 CHR 注册表","kind":"directory","paths":{"custom":["{identity}"]},"record_key":["character_id"],"timebase":"mixed","schema":"cross-shot-identity-v2","required":true}],
  "view":{"enabled":true,"renderer":"identity-graph","adapter":"visual-identity-v2","filters":["scope","status"],"primary_artifact":"visual_identity"}
}
```

### 作用与判断

把多个 shot 的轨迹聚为视觉身份 CHR。脸部证据优先，身体辅助；MUST_LINK、POSSIBLE_LINK、CANNOT_LINK 是不同证据等级，不是全部直接并入同一人。

### 输入与前置条件

选定一份 stage3_inputs 策略视图，读取轨迹 face/body 特征与共现证据；可结合 semantic_identity_decisions。所有输入应来自同一明确的运行/策略集合。

### 处理过程

抽取/整理 embedding → 建候选配对 → 检查人脸相似度和反证 → cannot-link 优先、簇间兼容的合并 → 晋升 CHR 或留 pending。检测帧共现是硬负约束，只有区间交叠而没有检测共现的情况进入 overlap_review_pairs。身份图和合并边保留，以支持以后拆分/回溯。

### 产物与消费

目录含 `registry.json`、`tracklet_features.json`、`identity_graph.json`、`merge_evidence.jsonl`、`cannot_links.json`、`possible_links.json`、`overlap_review_pairs.json`、`candidates/*.json`，以及已晋升人物目录下 `character.json/tracklets.json`。registry 中有 characters、pending_candidates、pending_links、约束违规计数和输入参数。消费方：旧画廊、旧角色绑定、身份审计页面。

### 异常、重跑与成本

无脸/低质量证据保留未决，不强行分配 CHR。不可把 CHR 数量当人物召回率；分母和真值不在该字段中。聚类参数或输入策略变更要重建身份图与下游画廊。

### 展示与验收

默认人物卡 + pending 数，开发模式展开图、合并理由和不能合并的证据。验收 cannot_link_violation_count、成员来源、身份图可追溯。
<!-- /module:legacy_identity -->

<!-- module:legacy_gallery -->
## 24 · 视觉人物画廊

```json pipeline-module
{
  "id":"legacy_gallery","title":"视觉人物画廊","order":240,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"为每个 CHR 挑选质量、纯度和多样性合适的脸部/身体参考图。",
  "depends_on":["legacy_identity"],"optional_depends_on":[],"inputs":["visual_identity"],
  "source_refs":["assets/visual_person/character_asset_bank.py"],"evidence":["code","sample"],
  "outputs":[{"id":"chr_gallery","title":"CHR 资产画廊","kind":"directory","paths":{"custom":["{asset_bank}"]},"record_key":["character_id"],"timebase":"none","schema":"mixed","required":true}],
  "view":{"enabled":true,"renderer":"character-gallery","adapter":"chr-gallery-v1","filters":["subject","status"],"primary_artifact":"chr_gallery"}
}
```

### 作用与判断

从身份簇证据中挑可用参考图；画廊整理与身份聚类是两个步骤。图多不代表身份更准确。

### 输入与前置条件

legacy_identity 的 registry 和 tracklet_features，源裁图必须可读。可选人工 reviewed 稳定描述，不能把生成描述当审核事实。

### 处理过程

按脸/身体分组，结合质量、脸部纯度、感知哈希/embedding 去重、视角及 appearance_state 选择。默认 face/body top-k 各 5、最小质量各 0.45；这些是该脚本参数，不应套到新 materializer 的 top-k 8。

### 产物与消费

根 `registry.json`；人物目录中的 `character.json/assets.json/tracklets.json/appearance_states.json` 和脸/身体/RGBA 图。字段里的资产路径是准确定位依据。被旧角色绑定以及外部人物 I2I 测试消费。

### 异常、重跑与成本

质量不足可以没有某种视角，不能补空白图伪装完整。源裁图失效时记录不可读，重建画廊无需重跑人物检测，前提是原始证据仍完整。

### 展示与验收

按脸、全身、外观状态浏览，展示来源轨迹与质量依据。默认不加载大量 embedding。验收去重、文件有效、每张图能追到来源。
<!-- /module:legacy_gallery -->

<!-- module:legacy_bind -->
## 25 · 旧链角色身份绑定

```json pipeline-module
{
  "id":"legacy_bind","title":"旧链角色身份绑定","order":250,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"把 CHR 视觉身份投票映射到 S-ID，并按审核证据决定可交付声音。",
  "depends_on":["legacy_refine","legacy_identity","legacy_gallery"],"optional_depends_on":["voice_preview"],"inputs":["semantic_tracklet_views","visual_identity","chr_gallery","reviewed_voice"],
  "source_refs":["assets/character_role_aligner.py"],"evidence":["code"],
  "outputs":[
    {"id":"role_binding","title":"CHR 与 S-ID 绑定报告","kind":"json","paths":{"custom":["{output}/role_alignment.json"]},"record_key":[],"timebase":"none","schema":"mixed","required":true},
    {"id":"legacy_characters","title":"旧链角色资产目录","kind":"directory","paths":{"custom":["{output}/characters"]},"record_key":["scope","subject_id"],"timebase":"none","schema":"mixed","required":false}
  ],
  "view":{"enabled":true,"renderer":"character-gallery","adapter":"characters-legacy-v1","filters":["scope","subject","status"],"primary_artifact":"role_binding"}
}
```

### 作用与判断

解决“视觉 CHR”与“语义/音频 S-ID”的名字不一致。与新 materializer 直接复制声音候选不同，当前旧绑定实现要求声音的明确审核和身份关联。

### 输入与前置条件

identity、同策略的 stage3-view、asset-bank；可选 voice-export。无审核清单仍可产出视觉结果，但不得悄悄从预览 WAV 补正式声音。

### 处理过程

轨迹票重为置信度乘决策系数（primary 1、supporting 0.85、context 0.5）；先在同 `(scope,shot,subject)` 取最大票，避免碎轨迹重复投票；再跨 shot 聚合。默认 min-share 0.6、min-weight 1.0、contest-margin 0.15，分别区分 confirmed/contested/weak/unresolved。冲突 winner 默认取更强者，其他记录保留。可选画廊复核和低频准入要展示开关。

### 产物与消费

`role_alignment.json` 记录 `chr_to_roles/role_to_chr/id_conflicts` 等；可选角色目录内复制 CHR 画廊，记录 role_character。声音只采用 `voice_asset_manifest.jsonl` 中 `review_status=approved` 且 `voice_link.status=matched` 的 consolidated reference_audio；无准入则记 voice_skipped_reason。输出用于旧链展示、I2I 引用及新旧对照。

### 异常、重跑与成本

投票主要为轻量计算，画廊复核可能增加特征计算。复制前可能清理旧声音目录，重跑要保留审核来源。多个 CHR 认领同 S-ID 时不能静默合并画廊来制造“一致身份”。

### 展示与验收

展示 CHR→scope:S-ID 映射、票重、冲突和未交付原因；声音卡区分审核资产与预览。验收没有批准清单时不产生“正式声音”，投票分母及同镜头去重正确。
<!-- /module:legacy_bind -->

<!-- module:compat_structure -->
## 26 · ClipRecord 兼容结构链

```json pipeline-module
{
  "id":"compat_structure","title":"ClipRecord 兼容结构链","order":260,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"保留旧工作区的镜头导入、ASR 辅助再切割和质量加权窗口聚合。",
  "depends_on":["shot_split"],"optional_depends_on":[],"inputs":["shots"],
  "source_refs":["common/workspace.py","common/clip_store.py","common/jsonl.py","schema/records.py","m02_shots/importer.py","m04_recut/run.py","m04_recut/cutter.py","m07_segment/run.py","m07_segment/window.py","m07_segment/consistency.py"],"evidence":["code"],
  "outputs":[{"id":"compat_records","title":"Clip/Segment/Global 兼容记录","kind":"directory","paths":{"custom":["{workspace}/annotations"]},"record_key":["video_id","clip_id"],"timebase":"mixed","schema":"1.0","required":true}],
  "view":{"enabled":true,"renderer":"record-table","adapter":"clip-records-v1","filters":["variant","status"],"primary_artifact":"compat_records"}
}
```

### 作用与判断

此支路源自另一套 ClipRecord→SegmentRecord→GlobalRecord 契约。`schema/records.py` 并不是所有 mimo JSONL 的统一 schema；其中 global 指整片记录，与约 90 秒 global clip 不是同一概念。

### 输入与前置条件

importer 将镜头导入相邻链接；recut 还需已有 ASR/质量记录及 config，这些并非本目录一个入口就能全部生成。状态 created/qc_done/asr_done/recut_done/speaker_done/captioned 是该 schema 的状态链，不代表相关所有生产模块都已在本仓库实现。

### 处理过程

长 clip 默认以约 20 秒限制再切：优先句间 ≥0.3 秒静音间隙，次选词边界，否则硬切；继承 ASR/质量并维护相邻关系，物理边界留帧重叠。窗口聚合默认 target 40 秒、min 为 target/2、max 为 target×1.5；可用 DINOv3 相似度约 0.6 的视觉门，直方图代理不承担同等硬阈值判断；统计质量等级权重和不合格时长比例。

### 产物与消费

`annotations/clips.jsonl` 含 ClipRecord 与 prev/next/parent 等关系；`annotations/segments.jsonl` 含 `segment_id/clip_ids/start_time/end_time/duration/unqualified_ratio/dinov3_score/continuity_ok/grade_weight/status`；`annotations/global.jsonl` 写回 segment_ids。clips/video、audio 等路径由 Workspace 约定。当前 segment_prompt 为空，注释提及 m06，但不能视为本目录已有完整 m06 编排。

### 异常、重跑与成本

recut 会更新原记录和相邻关系，按工作区隔离。fixed_v4 虽接受 clip_ids 的兼容投影，带 `002-1` 等派生 ID 不能默认与原 shot 一一对应，须验证数字归一和原 shot 覆盖；不能把任意 ClipRecord 输出直接无损接入主链。

### 展示与验收

作为高级兼容页展示原 clip→子 clip→segment 的关系、质量加权和切点原因。验收相邻链接、ASR 时间继承、覆盖以及派生 ID 转换，不把此数据当普通镜头清单解析。
<!-- /module:compat_structure -->

<!-- module:asset_demo -->
## 27 · 历史资产演示

```json pipeline-module
{
  "id":"asset_demo","title":"历史资产演示","order":270,"group":"compatibility","lifecycle":"legacy","implementation":"existing",
  "summary":"保留 segment 级 real/stub 资产演示的来源说明，防止占位结果被误当正式提取。",
  "depends_on":["source_prepare","segment_plan"],"optional_depends_on":[],"inputs":["source_video","segment_plan"],
  "source_refs":["assets_runner.py"],"evidence":["code"],
  "outputs":[{"id":"demo_assets","title":"演示资产输出","kind":"directory","paths":{"custom":["{output}"]},"record_key":["asset_id"],"timebase":"mixed","schema":"unversioned","required":true}],
  "view":{"enabled":false,"renderer":"artifact-browser","adapter":"demo-assets-v1","filters":["variant"],"primary_artifact":"demo_assets"}
}
```

### 作用与判断

`assets_runner.py` 是早期 segment 级演示编排。stub 根据时长/区间产生占位素材，real 尝试调用重模型；它不是当前统一角色链的默认入口。

### 输入与前置条件

源视频、某种 segments.jsonl、assets-root 和 mode。必须记录 `mode` 和 source_segment_impl。切片时间必须与传入视频一致。

### 处理过程

为每段分别组织人物/声音/物体示例结果；stub 的确定性数据用于结构演示，不能作为真实提取质量证据。详细调用由 assets_runner 各分支决定，不能借用其 docstring 替代主链契约。

### 产物与消费

output 下按资产类型保存媒体与记录，AssetRecord 的核心字段为 `asset_id/source_segment_id/source_segment_impl/kind/media_paths/meta`。结果仅用于历史布局查看或演示比较。

### 异常、重跑与成本

stub 仍可能调用 ffmpeg 抽帧/音频；real 会用 GPU，不能为试网页随手触发。演示结果保持独立目录。

### 展示与验收

默认隐藏，可在历史模块页开启；醒目标注 stub/real。验收演示资产永不混入正式角色/对象 KPI。
<!-- /module:asset_demo -->

<!-- section:execution-guide -->
## 入口、共享基础设施与局部重跑

| 意图 | 入口/实际覆盖 | 注意 |
|---|---|---|
| 全流程主线一键编排 | `run_full_pipeline.py --input-dir <目录> --asr-dir <ASR 目录>` | 切镜→聚合→全局描述(说话归属)→镜头描述→固定段规划→片段描述→切段→轨迹→全局归属→声音候选→物化→物体候选→物体定位分割+参考集→场景参考图，共 15 阶段，全部为最新逻辑（speaker_v2 / shot_v3 / fixed_segments_v4 / 轨迹构建器 / 全局归属 / 物化器 / grounding_queue_v2 / segmenter_v4 / 场景提取器），不含旧 CHR 链；每视频独立工作区，`pipeline_state.json` 支持断点续跑；ASR/VLM/GPU/ffmpeg 为外部前置，预检缺失会拒绝启动；跑完可 `--update-viewer-config` 并入结果网站 |
| 从视频跑基础三级描述 | `mimo_video_pipeline.py`：split-shots → aggregate-global → global-caption → shot-caption → segment-caption → cut-segments | 旧版 captioner；没有轨迹、声音、物体、场景、人物物化 |
| 固定结构的 Scheme A | `mimo_video_pipeline_scheme_a.py`：all 只含 global/shot/fixed-segment 三步描述 | 必须有 ASR 与分段；上游结构和下游物理切段另行运行 |
| 新人物归属与物化 | `global_align_pipeline/run_global_align_pipeline.sh` 的两步 | 默认路径需覆盖；此 shell 未先自动 dry-run；物化有清目录行为 |
| 声音增强建模 | `assets/voice/speaker_voice_asset_pipeline.py run` | independent global 音频链，不自动映射 S-ID |
| 声音候选预览 | `assets/voice/export_all_asr_audio_review.py` | 普通候选分类与带 registry 的审核资产是不同层次 |
| 旧 CHR 链 | builder → refiner_v3 → registry_builder → asset_bank → role_aligner | 同一策略视图贯穿，不插进新链当必经步骤 |
| I2I 包导出 | `assets/character_role_aligner.py --export-i2i-bundles` | 独立导出分支，核对 workspace 布局 |

共享模块并不意味着统一执行框架：`mimo_caption_client.py` 管理请求、视频编码、JSON 解析、manifest/prompt 辅助；`common/vlm.py` 是另一公共 VLM 能力，不能假设所有调用都经过它；`common/ffmpeg.py/jsonl.py/workspace.py/clip_store.py` 主要服务相应迁移模块；`schema/records.py` 是兼容结构 schema。`common/vlm_original.py` 为旧实现对照，变更前先查实际 import。`prompts/` 变化属于接口变化影响源，必须检查输出校验和缓存。

推荐局部重跑边界：

| 变化 | 最小应重新核验/计算的范围 |
|---|---|
| 展示文字、组件顺序 | 文档解析和网页配置；不调用模型 |
| global 角色注册表 | shot/segment 中的引用、人物归属、声音身份与角色汇总 |
| 个别 shot 描述的物体 | 对应 queue/grounding/bundle；是否影响 segment 对象索引需核验 |
| 固定 segment 边界 | segment 描述、切片、scene-segment 映射与包 |
| 某镜头轨迹 | 对应 global 对齐、角色素材；旧链还需受影响身份图/画廊 |
| 人物物化 top-k | 物化和展示，不重新调用 VLM；使用独立输出避免清旧目录 |
| 声音审核结论 | 正式绑定/展示状态，不把全部 ASR 重做 |
| 物体参考集挑选 | reference_sets 和对应任务包；不用再做定位分割 |

命令操作前确认项目 Python 环境、ffmpeg/ffprobe、GPU/模型权重路径和 VLM 连接；多个脚本依赖脚本目录 import 或父项目 third_party，不应把“源码已复制”当作完全独立可部署。本文不记录密钥、不为文档核验执行昂贵命令。外部 `MV_AnnotationCaption_Pipelines.md` 描述的音乐/上游系统、父目录图像生成与 TTS 工具，是上下游背景，不是本目录某个 all 自动覆盖的步骤。
<!-- /section:execution-guide -->

<!-- section:known-gaps -->
## 已知差异、限制和待实现项

| 编号 | 当前事实 | 对开发/展示的约束 |
|---|---|---|
| G01 | 基础编排的 shot-input/global-input 被多个阶段解释为不同类型输入 | 分阶段调用时核对原子 CLI；不要传一个目录假定同时是 shot 视频、shot caption 和 segment caption |
| G02 | Scheme A 输出在 annotations；复用的 cut-segments 默认找 captions，且 global-input override 在该阶段可能被当成 segment 输入 | Scheme A all 不切段；显式指定 cutter 的 segment manifest，不能照抄默认路径 |
| G03 | 旧说明称默认段由 MiMo 切；当前 Scheme A 已要求固定上游段 | 路线和规划来源必须按代码/内层 schema 标识 |
| G04 | 0916 是局部 global 的整合样本，manifest 保留原工作区路径；没收录 tracklets/完整 voice registry | 展示“未收录/外部引用”，不冒充全片全链完成 |
| G05 | 新物化复制 valid 候选声音；旧绑定只复制 approved+matched 的参考声音 | 使用不同 adapter，状态标签不混用 |
| G06 | 新 materializer 默认接受 new_subject_candidate，且整体重建 characters | 候选集合不能计为确认角色；重跑使用新输出 |
| G07 | 新对齐 ok 允许 unknown；返回码 0 允许部分 global 失败 | 逐项状态、覆盖率和失败数必须同时展示 |
| G08 | V4 物体算法仍在 v3 文件；0916 观测是 V3，67 个歧义行也为 ok | 适配器按 schema/字段做质量解释，不能改写旧样本语义 |
| G09 | 当前 builder ASR 是 CocktailASR 目标说话人方案，且修改未提交 | 不宣称完整多人 ASR 或精确句级时间已验证 |
| G10 | viewer 主要靠固定目录、固定 stage 分支和缓存发现 | 文档热加载、模块注册、通用运行状态均待实现 |
| G11 | viewer README 的路径隔离描述与代码软链接策略不一致 | 新网站明确配置允许的媒体根，核验 resolve 后路径 |
| G12 | `.gitignore` 忽略 *.md；旧长文档本身大多不在 git ls-files 中 | 必须按文档一显式纳入三份文档，否则 diff 驱动维护不成立 |
| G13 | 没有统一的 run manifest、输入哈希和跨所有模块的原子发布协议 | 未来索引层提供运行绑定和 provenance；现阶段不能推断产物全部由当前 commit 生成 |
| G14 | fixed_v4 容许兼容 clip_ids 的数字提取，m04 有派生子 clip | 验证派生 ID 映射，不保证无损直连 |
| G15 | I2I 导出仍偏旧布局及 role_to_chr，0916 整合布局经过整理 | 需明确 adapter/导出绑定，新链完整接入尚未核验 |
| G16 | 2026-09-21 五项增强（normalize_video 首阶段、semsim 检测器、场景提取三选项、CocktailASR 多参考多说话人、TransNet/SCRL 权重环境变量）均为仅代码核验，本机无运行环境未实测 | 服务器实验前不得把默认行为变化当作已验证效果；run_full_pipeline 默认开启降采样会改变后续所有产物口径 |
| G17 | person_align/character_pack schema 已升 v2（entity_nature、identity_check、depicted_references），文档样本仍来自 v1 运行 | 适配器解析 v2 新字段时对 v1 旧样本按缺省 uncertain 处理；新旧运行不能混在一份 characters 里 |

优先级：网站一期先解决 G04/G05/G08 的真实展示、G10 的文档解析、G11 的路径读取、G12 的文档版本管理；业务算法变化是独立开发任务。本文没有据此修改任何实现。
<!-- /section:known-gaps -->

<!-- section:evidence-baseline -->
## 证据基线与下一轮核验入口

核验源码根：`/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annotation_level3_v2/extract_full_pipeline`。

核验样本根：`../workspace/movie_full_pipeline_20260916`。同时查看了 `../workspace/movie_full_pipeline_20260912_v4/assets/visual_person_v2` 的中间结果目录。样本生产 commit 未被统一记录，本次不将当前 HEAD 当作样本的生产版本。

Git HEAD：`0c5ddfd367388461d96bed16fc57e4560a70ace8`。初始工作树有 4 个已跟踪修改文件：`.gitignore`、`assets/voice/speaker_registry_builder.py`、`global_align_pipeline/global_tracklet_aligner.py`、`global_align_pipeline/global_character_materializer.py`。未跟踪内容包括 `asset_viewer/`、`assets/voice/cocktail_asr.py`、`test_cocktail_asr.py` 和少量 pycache。本次保留这些用户改动，未提交/暂存。

关键文件内容指纹用于下一 Agent 判断是否可复用本次结论；未列文件仍按基线 diff 和模块 source_refs 核验：

| 文件 | 2026-09-18 SHA-256 |
|---|---|
| `mimo_video_pipeline.py` | `bc4c7bb0bb50746b6aa4d579ed6ac25c35fae2c0523adb54d40576590ef1e2c8` |
| `mimo_video_pipeline_scheme_a.py` | `244321c27e42a6ac9631b3b19d4d5ac4d5ab3c9396d933a73451683f75c693df` |
| `global_align_pipeline/global_tracklet_aligner.py` | `8c73d5bc80b48df75db68ab400fd1e563242b7d0d8e8473ff2f5865b7e33da2b` |
| `global_align_pipeline/global_character_materializer.py` | `62c49a5e79a28092070a3a6c93f515039ceb5550fbd43321e6a9b35e4f6a40ff` |
| `assets/voice/speaker_registry_builder.py` | `0d6c243a1dee18e7f578b1ae4de7b2ba7d9f00146ef7536bbf0b438d1b81841b` |
| `assets/voice/cocktail_asr.py` | `8b40e95e6024fb59af8e3a3a3d2d11c44e05d371e8c17d03864c654ac6c7b94d` |
| `asset_viewer/server.py` | `128c7f6dff96c2100fd2cb4a68845e058b72e1e1dd3d4ccf088a7b5e0be49c30` |
| `asset_viewer/pipeline_router.py` | `66736a2efff6efa96c7b5772f74d323365c6c80d8e18364db88225f3130ba50b` |
| `asset_viewer/html_template.py` | `5df94528a7be17fa4bc483d06c0d74c71270c9c8901ff6b2e48292de880973f5` |
| `.gitignore` | `26e5694ec51042249c409061589f87a13d894c7ec1011f67384bb4b942c91a52` |

本次证据覆盖入口编排、核心算法决策与落盘代码、已有 viewer 的读取/服务逻辑、代表 JSON/JSONL 和目录结构。没有验证模型端点可用性、GPU 运行、全量媒体内容、所有历史脚本组合、实际代理连通性或语义标注准确率。所有设计指标均留在文档三，不能反过来当作当前系统验收结果。

内容更新记录：`1.4.0 / 2026-09-21`，登记五项未运行验证的代码增强：统一降采样阶段（source_prepare 新增 normalize_video 产物与 run_full_pipeline 默认首阶段）、global_group 的 semsim 免训练检测器与 auto 回落顺序/环境变量/--no-export 容忍、scene_select 三个 opt-in 选项、voice_model 的 CocktailASR 多参考多说话人、person_align/character_pack schema v2（entity_nature/identity_check/depicted_references，此三项为更早会话实现、本次补登记）；已知差异新增 G16/G17。`1.3.0 / 2026-09-21`，展示瘦身：主线收敛为与 run_full_pipeline.py 15 阶段编排对应的 15 个模块，source_prepare、generation_bundle、i2i_review、tts_review、result_browse 移入文末“支线、扩展与外部模块”章节与最末分组（移动而非删除）；默认路线与总览图同步。`1.2.0 / 2026-09-21`，新增全流程主线一键编排入口 run_full_pipeline.py（15 阶段、最新逻辑、断点续跑、可并入结果网站），入口表与主线路线说明同步。`1.1.0 / 2026-09-20`，主线/支线重排：主线模块（视频结构、内容理解、人物资产、物体、场景、生成对比、结果服务）前置并按流程重编号；旧 CHR 人物链、声音增强建模与兼容支路移入文末“支线与历史兼容模块”章节与最末分组；总览图改为只画主线。模块事实、依赖与产物契约未变。`1.0.0 / 2026-09-18`，建立 27 个业务模块的统一说明及网站读取元数据，区分现有实现、外部结果、兼容分支和待建展示层。
<!-- /section:evidence-baseline -->
