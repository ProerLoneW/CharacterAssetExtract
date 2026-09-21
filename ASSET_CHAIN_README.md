# Asset Extraction Pipelines

本文档梳理 `extract_full_pipeline/` 下当前可用的资产提取链路，覆盖角色视觉资产、角色声音资产、关键物体资产、关键场景资产，以及角色视觉 ID 与语义角色 ID 的对齐和物化。

文档结构参考 [`MV_AnnotationCaption_Pipelines.md`](./MV_AnnotationCaption_Pipelines.md)：先给出总览，再按阶段说明入口、模型、输入、输出、关键规则、默认阈值、串联顺序和校验方法。角色链的概念导读可配合 [`CHARACTER_ASSET_CHAIN_README.md`](./CHARACTER_ASSET_CHAIN_README.md) 阅读；若文档描述与代码不一致，以当前脚本的 docstring、`argparse` 默认值和实际产物为准。

> 范围说明：切镜、global/shot/segment 三级 caption 和 ASR 属于资产链上游，本文件只说明它们与资产阶段的接口。完整 caption 链见 [`PIPELINE_README.md`](./PIPELINE_README.md) 和 [`MV_AnnotationCaption_Pipelines.md`](./MV_AnnotationCaption_Pipelines.md)。

> **2026-09-21 增量**（本节为背景索引快注，权威说明见 [`docs/02`](./docs/02_全流程Pipeline说明.md) v1.4.0；均未服务器实测）：
> 1. 主线编排新增默认首阶段 `normalize_video`（480p/15fps 统一降采样，`--no-normalize` 关闭），下游 shot/scene 输入随之改变口径；
> 2. 全局聚合新增 `--scene-detector semsim`（免训练 DINOv3 上下文自相似 + 音频对比，BaSSL/ShotCoL/CMS 取向），`auto` 回落顺序 SCRL → semsim → metadata；SCRL/TransNetV2 权重可经 `SCRL_ENCODER_PATH`/`SCRL_BOUNDARY_PATH`/`TRANSNET_MODEL_PATH` 覆盖；
> 3. 场景链 Stage S0 新增三个 opt-in 选项：`--shots-manifest`（采样吸附镜头边界）、`--max-scenes auto`（silhouette 定簇数）、`--anchor-selection coverage`（k-center 锚点），默认行为不变；
> 4. 声音链 ASR 引擎 CocktailASR-1 新增多参考多说话人模式（`cocktail_asr.py --ref S1=path --ref S2=path`，目标说话人拒识多遍合并）；官方无多说话人版本，重叠语音表现待实测；
> 5. 角色链 Stage5 替代入口（`global_align_pipeline/`）schema 已升 v2：`entity_nature`（画中人防护）、materializer `--identity-check`（人脸一致性过滤）与 `depicted_references`，详见 [`global_align_pipeline/README.md`](./global_align_pipeline/README.md)。

## 0. Pipeline 概览

当前资产链不是一条完全串行的流水线，而是四条分支：

```text
视频 + shot/global clip + ASR + Scheme A 三级 caption
  │
  ├─ 角色视觉
  │    shot_tracklet_builder
  │      └─ 内部逐帧调用 entity_asset_extractor 形成 OBS
  │    → semantic_role_tracklet_refiner_v3
  │    → visual_character_registry_builder（CHR）
  │    → character_asset_bank
  │    → character_role_aligner（CHR → S-ID）
  │    → <output>/characters/<scope>/<S-ID>/{visual,voice,role_character.json}
  │
  ├─ 角色声音
  │    mimo_global_captioner_speaker_v2（ASR → S-ID）
  │    → speaker_attributed_asr.jsonl
  │    → export_all_asr_audio_review（句级预览 wav）
  │    → 可选 consolidated speaker registry + 人工 approvals
  │    → character_role_aligner 选择正式 voice 资产
  │
  ├─ 关键物体
  │    semantic_role_tracklet_refiner_v3 生成候选队列
  │      或 build_object_grounding_queue_v2 直接投影 shot caption（评测适配器）
  │    → object_grounding_segmenter_v3（默认执行 V4 行为）
  │    → object_observations.jsonl + frame/mask/crop/rgba/overlay
  │    → 可选 reference_sets.json
  │
  └─ 关键场景
       important_scene_frame_extractor
       → DINOv3 场景聚类 + YOLOv8n/OpenCV 可用性排序
       → scene_index.json + scene_###/anchor_*.jpg
```

角色视觉链与角色声音链最终在 `character_role_aligner.py` 汇合。关键物体和关键场景保留独立目录；只有启用可选的 I2I bridge export 时，`character_role_aligner.py` 才会读取它们并生成 scene/object bundle，不会把它们写进角色目录。

| 阶段 | 入口 | 做什么 | 关键输入 | 关键输出 | 主要默认值 |
| --- | --- | --- | --- | --- | --- |
| 角色 Stage1 | `assets/visual_person/shot_tracklet_builder.py` | shot 内人物检测、分割、追踪 | `shot_clips/*.mp4` | `tracklets/shots/<SHOT>/manifest.json`、`tracklet_signatures.json`、overlay | 重检测间隔 `8` 帧；最多丢失 `12` 帧；最少 `3` 次检测 / `8` 帧 |
| 角色 Stage1.5 | `assets/visual_person/semantic_role_tracklet_refiner_v3.py` | 用 MiMo 和 caption 证据将 Tracklet 对齐到 S-ID，同时产出物体候选 | Tracklet + global/shot/segment caption | `alignments/`、`stage3_inputs/{primary,narrative,safe}/`、`object_grounding_candidates.jsonl` | concurrency `1`；rate `0.5/s`；retry `3` |
| 角色 Stage3 | `assets/visual_person/visual_character_registry_builder.py` | 跨 shot 合并视觉身份 | Stage1.5 view + semantic decisions | `identity/registry.json`、身份图、候选与 CHR 目录 | CHR 最低置信度 `0.60`；cannot-link 优先 |
| 角色 Stage4 | `assets/visual_person/character_asset_bank.py` | 对已晋升 CHR 做质量、纯度、去重和槽位竞争 | Stage3 identity | `asset_bank/characters/CHRxxx/` | face/body top-k 均为 `5`；质量阈值均为 `0.45` |
| 声音 A | `mimo_global_captioner_speaker_v2.py` | global caption 与 ASR 说话人归属一次完成 | global clip manifest + ASR | `global_captions.jsonl`、`speaker_attributed_asr.jsonl` | concurrency `2`；retry `3`；temperature `0.1` |
| 声音 B | `assets/voice/export_all_asr_audio_review.py` | 按 ASR 句级切预览 wav，可选关联正式声纹 registry | `speaker_attributed_asr.jsonl` | `valid/`、`invalid/`、三份 manifest；可选 `voice_asset_manifest.jsonl` | PCM s16le、16 kHz、mono |
| 角色 Stage5 | `assets/character_role_aligner.py` | CHR 对 S-ID 投票并物化角色资产 | identity + Stage1.5 view + bank + voice export | `role_alignment.json`、`characters/<scope>/<S-ID>/` | share `0.6`；weight `1.0`；contest margin `0.15` |
| 物体 O0 | Stage1.5 或 `build_object_grounding_queue_v2.py` | 产生带可见时间窗的候选 | shot caption / Stage1.5 结果 | `object-grounding-candidates-v2` JSONL | 直接投影适配器默认只保留 `must/high` |
| 物体 O1 | `assets/key_object/object_grounding_segmenter_v3.py` | 时窗内 GroundingDINO + SAM 3.1 | candidate JSONL + `shots.jsonl` | `object_observations.jsonl` + 图像资产 | 每 shot `5` 帧；box `0.20`；默认最多补采 `4` 帧 |
| 物体 O2 | 同一脚本的 `--build-reference-sets` 模式 | 从已有 observation 选择 3–5 张参考图 | grounding 输出目录 | `reference_sets.json` | 全帧占比上限 `0.8`；近重复相似度 `0.95`；最多 `5` 张 |
| 场景 S0 | `assets/scene/important_scene_frame_extractor.py` | 纯视觉发现重要场景并选锚点 | 完整视频或 global clip | `<video_stem>/scene_index.json`、`scene_###/` | 每 `1.2s` 采样；最多 `12` 个场景；每场景最多 `3` 帧 |

最容易混淆的几点：

- `shot_tracklet_builder.py` 已经在内部调用 `entity_asset_extractor.py` 生成逐帧 OBS；通常不需要先独立运行一次 Stage0。
- `semantic_role_tracklet_refiner_v3.py` 是当前角色语义对齐入口；v1/v2 是实现历史，v3 复用 v2 主体并补上 `visible_time_range_sec` 与 v3 schema。
- `object_grounding_segmenter_v3.py` 文件名虽是 v3，默认 `--window-resample-max=4` 时写 v4 observation；设为 `0` 才回到精确 v3 行为。
- 声音 `valid/*.wav` 是候选预览，不等于正式角色声音。最终复制到角色目录还需要 `voice_asset_manifest.jsonl` 中同时满足 `review_status=approved` 和 `voice_link.status=matched`。
- `CHRxxx` 是跨 shot 的视觉身份；`S1/S2/...` 是 global scope 内的语义角色；二者不是天然一一对应。

## 1. 工作区、输入和 ID 约定

### 1.1 典型工作区

脚本都通过 CLI 接受路径，没有强制唯一工作区结构。仓库现有 `run_v2_asset_pipeline.sh` 使用的典型布局如下：

```text
<workspace>/
  middle/
    shots.jsonl
    shot_clips/*.mp4
  global/
    global_clips.jsonl
    global_clips/*.mp4
  segments/segments.jsonl
  captions_speaker/
    global/
    shot/
    segment/
  assets/
    visual_person_v2/
      tracklets/
      alignment/
      identity/
      asset_bank/
    voice_by_role/
    key_object/
    scene/
  characters/                 # 取决于 character_role_aligner --output
```

不同试验工作区也可能使用 `shots/shots.jsonl` 和 `shots/clips/`。资产脚本不依赖固定名字，传入的 manifest 内 `clip_path` / `video_path` 和 CLI 路径才是实际数据源。

### 1.2 运行环境

下文命令统一假设当前目录是 `general_video_annotation_level3_v2` 根目录。这样 Stage3 的 `local_models/...` 默认相对路径可以正确解析：

```bash
ROOT=/path/to/general_video_annotation_level3_v2
EFP="$ROOT/extract_full_pipeline"
PY=python

cd "$ROOT"
export PYTHONPATH="$ROOT/third_party/sam3_src:${PYTHONPATH:-}"
```

实际生产环境可将 `PY` 换成项目 Python 解释器。若直接在 `extract_full_pipeline/` 内执行，需显式传绝对模型路径。

### 1.3 ID 体系

| ID | 作用域 | 由谁产生 | 含义 |
| --- | --- | --- | --- |
| `OBS` | 单帧 | `entity_asset_extractor` | 一个 frame-local 人物观测，不表示跨帧身份 |
| Tracklet ID | 单 shot | `shot_tracklet_builder` | 同一物理镜头内连续出现的同一个人 |
| `CHR001...` | 多 shot 视觉域 | `visual_character_registry_builder` | 基于 face/body evidence 聚合的视觉身份 |
| `S1/S2/...` | 单 global scope | Scheme A caption | 剧情语义角色或说话人；完整键是 `<scope>:<S-ID>` |
| `SHOT_xxx_OCn` | 单 shot 候选 | shot caption / Stage1.5 | 关键物体候选 ID，不是跨 shot 物体身份 |
| `scene_001...` | 单输入视频 | scene extractor | DINOv3 外观聚类后选出的重要场景簇 |

`S1` 在 `global_0040` 与 `global_0041` 中是两个局部编号。任何跨 global 的消费都应使用 `global_0040:S1` 这样的 scope-qualified key。

## 2. 角色视觉资产链

### 2.1 Stage1：shot 内 OBS + Tracklet

入口：

```bash
"$PY" "$EFP/assets/visual_person/shot_tracklet_builder.py" \
  --shots-dir <workspace>/middle/shot_clips \
  --output <workspace>/assets/visual_person_v2/tracklets \
  --continue-on-error
```

#### 模型和处理逻辑

每个 shot 内部按以下顺序工作：

```text
挑选初始化帧和周期性检测锚帧
  → Sa2VA 发现主要人物区域
  → GroundingDINO person box
  → 困难帧局部放大重检测
  → InsightFace 检脸并绑定到人物
  → SAM 3.1 实例 mask
  → 稀疏光流在锚帧间传播
  → Hungarian matching 关联观测
  → ACTIVE / LOST / FINISHED 状态机
  → 过滤低证据轨迹并导出 TrackletSignature
```

本阶段只建立 shot-local Tracklet，`identity_id` 仍为 `None`。同一帧共同出现的不同 Tracklet 会形成 cannot-link evidence，供 Stage3 阻止错误聚类。

#### 输出

```text
tracklets/
  manifest.json
  shots/<SHOT_ID>/
    manifest.json
    tracklet_signatures.json
    tracklet_overlay.mp4
    observations/frame_####/OBS##/...
```

`manifest.json` 内保存完整 `tracklets[]`、`filtered_tracklets[]`、`cannot_links[]`、检测帧和运行参数；`tracklet_signatures.json` 是后续 Stage1.5/Stage3 的紧凑输入。

#### 关键默认值

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--redetect-interval` | `8` 帧 | 周期性重新检测 |
| `--max-missed-frames` | `12` 帧 | Tracklet 进入结束状态前允许丢失的帧数 |
| `--min-match-score` | `0.25` | observation 与现有轨迹的最小关联分数 |
| `--min-flow-confidence` | `0.25` | 光流传播最低置信度 |
| `--redetect-cooldown` | `4` 帧 | 冷启动/补检测冷却 |
| `--min-track-detections` | `3` | 常规轨迹最少独立检测次数 |
| `--min-track-frames` | `8` | 常规轨迹最少帧数 |
| `--short-track-min-confidence` | `0.5` | 短 shot 联合证据准入分数 |
| `--short-track-min-coverage` | `0.5` | 短 shot 轨迹覆盖率 |
| `--selection-window-sec` | `1.5` | 初始化帧选择窗口 |
| `--candidate-frames` | `12` | 初始化候选帧数 |
| `--max-local-rois` | `3` | 困难帧最多局部重检测区域 |

`--disable-local-redetect`、`--disable-dedup-instance-evidence` 和 `--disable-sam` 是回滚/消融开关，常规生产跑法保持默认开启的本地重检测、实例证据去重和 SAM。

#### `entity_asset_extractor.py` 的位置

该脚本可独立运行，用于检查单个视频的逐帧人物发现和 OBS 资产：

```bash
"$PY" "$EFP/assets/visual_person/entity_asset_extractor.py" \
  --video <video.mp4> \
  --output <obs_debug_dir> \
  --num-samples 5
```

它是 `shot_tracklet_builder.py` 的底层 observation backend，也是调试工具；独立输出没有 Tracklet 和 CHR，不能直接当角色资产库。

### 2.2 Stage1.5：Tracklet → S-ID 语义对齐

入口：

```bash
"$PY" "$EFP/assets/visual_person/semantic_role_tracklet_refiner_v3.py" \
  --tracklets <workspace>/assets/visual_person_v2/tracklets \
  --global-captions <workspace>/captions_speaker/global \
  --shot-captions <workspace>/captions_speaker/shot \
  --segment-captions <workspace>/captions_speaker/segment \
  --output <workspace>/assets/visual_person_v2/alignment \
  --concurrency 2 \
  --skip-existing
```

#### 模型实际输入

- 当前 shot 的 `tracklet_overlay.mp4`，画面中带稳定 Tracklet ID。
- Tracklet inventory 和时间范围。
- global/segment/shot caption 中的 subject registry、角色描述、对话、可见性和物体候选证据。
- `tracklet_semantic_object_alignment_prompt_en_v2.md`。

MiMo 返回人物/非人物判断、Tracklet 对 S-ID 的归属、`primary/supporting/contextual/background` 等语义分类，并绑定或补充关键物体候选。

#### 输出

| 路径 | 用途 |
| --- | --- |
| `alignments/*.json` | 每个 shot 的 Tracklet 语义判断 |
| `manifest.json` | 成功/失败计数、策略路径和可选 feedback 汇总 |
| `semantic_tracklet_index.json` | 全局 Tracklet 语义索引 |
| `stage3_inputs/primary/` | 只保留高置信主角色；默认 Stage3 输入 |
| `stage3_inputs/narrative/` | 主角色 + supporting 角色，召回更高 |
| `stage3_inputs/safe/` | fail-open 视图，只移除高置信背景/非人物 |
| `*/semantic_identity_decisions.json` | 同一语义 S-ID 的跨 shot 身份提示，供 Stage3 审核 |
| `object_grounding_candidates.jsonl` | 经过 Stage1.5 绑定的物体候选，保留 `visible_time_range_sec` |
| `object_proposals.jsonl` | 物体 proposal 审计结果 |

当前 schema 为 `semantic-role-tracklet-object-alignment-v3`，物体队列 schema 为 `object-grounding-candidates-v2`。

#### 关键默认值和开关

| 参数 | 默认值 |
| --- | --- |
| concurrency | `1` |
| rate limit | `0.5` request/s |
| retries | `3` |
| timeout | `1200s` |
| max completion tokens | `4096` |
| temperature | `0.1` |
| reject confidence | `0.85` |
| semantic link confidence | `0.90` |
| object proposal confidence | `0.50` |

可选 `--enable-semantic-feedback` 会为“caption 明确可见但 Tracklet 未召回”的 shot 写 `feedback_requests.json`。随后可用 `shot_tracklet_builder.py --semantic-feedback-requests ...` 在独立输出目录补检测一次，再通过 `--feedback-tracklets` 重新对齐。反馈最多一轮，且不应覆盖原始 Tracklet 目录。

### 2.3 Stage3：跨 shot 视觉身份聚类

入口：

```bash
"$PY" "$EFP/assets/visual_person/visual_character_registry_builder.py" \
  --inputs <workspace>/assets/visual_person_v2/alignment/stage3_inputs/primary \
  --vlm-decisions <workspace>/assets/visual_person_v2/alignment/stage3_inputs/primary/semantic_identity_decisions.json \
  --output <workspace>/assets/visual_person_v2/identity
```

#### 处理逻辑

Stage3 为每条 Tracklet 计算/读取 face 与 body embedding，构造可回滚身份图：

```text
TrackletSignature
  → 人脸质量分层 + body 特征
  → 两两产生 MUST_LINK / POSSIBLE_LINK / CANNOT_LINK / NO_EVIDENCE
  → 同帧检测共现形成 CANNOT_LINK
  → semantic_identity_decisions 只审核视觉上可成立的候选边
  → 约束感知聚类
  → 达到 CHR 置信度的候选晋升为 CHRxxx
```

cannot-link 是硬约束。语义 S-ID 只能帮助 `POSSIBLE_LINK`，默认不能绕过明确的人脸反证或同帧共现证据。

#### 输出

```text
identity/
  registry.json
  identity_graph.json
  tracklet_features.json
  merge_evidence.jsonl
  cannot_links.json
  possible_links.json
  overlap_review_pairs.json
  candidates/CAND_CHR###.json
  characters/CHR###/{character.json,tracklets.json}
```

`registry.json.characters[]` 是已晋升 CHR；`pending_candidates[]` 保留证据不足的候选，二者不要混用。

#### 关键默认值

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--min-face-det-score` | `0.75` | 高质量人脸检测门槛 |
| `--min-face-side` | `48` | 最小人脸边长 |
| `--min-face-sharpness` | `30.0` | 最小人脸清晰度 |
| `--must-face-max / --must-face-mean` | `0.55 / 0.50` | 人脸 MUST_LINK 证据 |
| `--possible-face-mean` | `0.35` | 人脸 POSSIBLE_LINK 证据 |
| `CONTRADICTION_FACE_MEAN` | `0.25` | 双方都有高质量脸时的人脸反证线 |
| `--cluster-compatibility-mean` | `0.40` | cluster 合并兼容线 |
| `--must-body-mean` | `0.90` | body 强证据线；body-only MUST_LINK 默认关闭 |
| `--possible-body-mean` | `0.78` | body 候选边门槛 |
| `--appearance-state-threshold` | `0.82` | 外观状态聚合阈值 |
| `--chr-min-confidence` | `0.60` | 候选晋升 CHR 的最低置信度 |

`--allow-body-must-link`、`--allow-medium-face-linking` 和 `--allow-vlm-edge-creation` 都会放宽自动合并，默认关闭。

### 2.4 Stage4：角色视觉资产库

入口：

```bash
"$PY" "$EFP/assets/visual_person/character_asset_bank.py" \
  --identity <workspace>/assets/visual_person_v2/identity \
  --output <workspace>/assets/visual_person_v2/asset_bank
```

Stage4 只消费 Stage3 identity，不改变身份成员关系。每张候选图依次通过质量打分、face purity、视角/身体完整度分类、感知哈希去重和有限槽位竞争。

#### 输出

```text
asset_bank/
  registry.json
  characters/CHR001/
    character.json
    tracklets.json
    appearance_states.json
    assets.json
    representative/primary_face.jpg
    representative/primary_appearance.jpg
    face/<frontal|three_quarter|profile|other>/*.jpg
    body/<view>/<full|partial>/<appearance_state>/*.jpg
```

`assets.json` 同时记录被接受资产、来源 shot/tracklet/OBS、质量分解和 `rejected_assets`，适合定位“为什么某张图没有进 bank”。当高质量跨 Tracklet 人脸不足时，资产可标记为 `identity_unverified` 后保留，而不是伪造纯度结论。

| 参数 | 默认值 |
| --- | --- |
| `--face-top-k` | `5` |
| `--body-top-k` | `5` |
| `--min-face-quality` | `0.45` |
| `--min-body-quality` | `0.45` |
| `--phash-distance` | `5` |
| `--embedding-duplicate` | `0.97` |
| `--face-purity-threshold` | `0.35` |

### 2.5 Stage5：CHR → S-ID 对齐与角色物化

入口：

```bash
"$PY" "$EFP/assets/character_role_aligner.py" \
  --identity <workspace>/assets/visual_person_v2/identity \
  --stage3-view <workspace>/assets/visual_person_v2/alignment/stage3_inputs/primary \
  --asset-bank <workspace>/assets/visual_person_v2/asset_bank \
  --voice-export <workspace>/assets/voice_by_role \
  --output <角色对齐输出根>
```

#### 投票规则

每个 CHR 的 Tracklet 根据 Stage1.5 结果投票：

```text
vote_weight = confidence × decision_factor

keep_primary    = 1.00
keep_supporting = 0.85
keep_context    = 0.50
其他 decision  = 0
```

同一 `(scope, shot, subject_id)` 的多个 Tracklet 碎片只取最大一票，再跨 shot 求和，避免同一镜头碎轨重复放大证据。

| 状态 | 判定 | 是否进入物化竞争 |
| --- | --- | --- |
| `contested` | 至少两个候选，前两名 share 差 `< 0.15` | 否 |
| `confirmed` | 第一名 share `>= 0.6` 且总有效权重 `>= 1.0` | 是 |
| `weak` | 有第一名但未达到 confirmed 门槛 | 默认否 |
| `unresolved` | 无有效 S-ID 票 | 否 |

同一 `(scope, S-ID)` 若被多个 confirmed CHR 认领，票重最高的 CHR 获得物化目录，其余写入 `id_conflicts[]` 并继续保留在 Stage4 bank，不会被静默合并。

#### 输出路径语义

脚本固定写：

```text
<output>/role_alignment.json
<output>/characters/<scope>/<S-ID>/
  role_character.json
  visual/...
  voice/...
```

因此传 `--output <workspace>/characters` 会得到 `<workspace>/characters/characters/...`。如果目标是 `<workspace>/characters/<scope>/<S-ID>`，应把 `--output` 设为 `<workspace>`；如果目标是 `<workspace>/assets/characters/...` 和 `<workspace>/assets/role_alignment.json`，则设为 `<workspace>/assets`。

#### 视觉和声音复制规则

- `visual/` 从 `asset_bank/characters/<CHR>` 整目录复制。
- 每次重跑会先清空该角色旧的 `voice/`，避免残留。
- `voice/` 只读取 `<voice-export>/voice_asset_manifest.jsonl`。
- 一行必须同时满足：`review_status=approved`、`voice_link.status=matched`、`global_id` 对应当前 scope、`asr_speaker_id` 等于当前 S-ID。
- 实际复制的是 `voice_link.reference_audio[]` 指向的 consolidated clean voice，而不是 `valid/.../*.wav` 预览。

可选 `--multi-chr-gallery-review` 会比较冲突 CHR 的 Stage3 人脸 embedding，并在报告中标注 `compatible/inconclusive/contradicted`；它不更换 winner，也不合并资产。可选 `--allow-low-frequency-admission` 可让满足严格条件且权重 `>=0.9` 的 weak 单角色进入低频准入。

## 3. 角色声音资产链

### 3.1 Stage A：Scheme A global caption + ASR 归属

入口：`mimo_global_captioner_speaker_v2.py`。它对每个 global clip 发起一次 MiMo 请求，同时生成 global caption、`subject_registry` 和每条权威 ASR 的 S-ID 归属。

```bash
"$PY" "$EFP/mimo_global_captioner_speaker_v2.py" \
  --input <global clip manifest 或目录> \
  --output-dir <caption 输出根> \
  --asr-input <权威 ASR JSON/JSONL> \
  --asr-timebase auto \
  --concurrency 2 \
  --skip-existing
```

模型不会改写 ASR 文本和时间戳。响应校验通过后，脚本把模型判断与原始 ASR 做本地 join，生成：

```text
<output>/<manifest_parent>/
  global_captions/global_00040.json
  global_captions.jsonl
  speaker_attributed_asr.jsonl
  raw_responses/
  summary.json
```

使用 `--flat` 时不再追加 `<manifest_parent>/`。`speaker_attributed_asr.jsonl` 每行包含 `global_id`、`global_clip_path`、`asr_id`、`speaker_id`（值为 `S1/S2/...`）、`time_range_sec`、归属证据、`audio_asset_candidate` 和拒绝原因。

| 参数 | 默认值 |
| --- | --- |
| `--asr-timebase` | `auto` |
| `--concurrency` | `2` |
| `--timeout` | `1800s` |
| `--retries` | `3` |
| `--max-completion-tokens` | `32768` |
| `--temperature` | `0.1` |
| `--thinking` | 关闭 |

### 3.2 Stage B：ASR 句级声音预览

入口：

```bash
"$PY" "$EFP/assets/voice/export_all_asr_audio_review.py" \
  --input <speaker_attributed_asr.jsonl> \
  --output-dir <workspace>/assets/voice_by_role
```

脚本从每行 `global_clip_path` 按 `time_range_sec` 裁剪音频，统一为 PCM s16le、16 kHz、mono，并按 `audio_asset_candidate` 分为：

```text
voice_by_role/
  valid/global_####/S#/ASR#####.wav
  invalid/global_####/S#/ASR#####.wav
  all_asr_manifest.jsonl
  valid_manifest.jsonl
  invalid_manifest.jsonl
  summary.json
```

这里的 `valid` 只表示 `audio_asset_candidate=true` 的初筛候选，`invalid` 表示该字段为 false。两类都裁成预览 wav，方便人工审听；它们都不会自动晋升为正式 voice asset。

### 3.3 正式 voice asset 的可选 registry + approval 门槛

若额外传入 `--voice-registry` 或 `--voice-registry-root`，Stage B 会把 ASR 区间与 `speaker_voice_asset_pipeline.py` 生成的 `consolidated/registry.json` 做时间交集，并写 `voice_asset_manifest.jsonl`：

```bash
"$PY" "$EFP/assets/voice/export_all_asr_audio_review.py" \
  --input <speaker_attributed_asr.jsonl> \
  --output-dir <workspace>/assets/voice_by_role \
  --voice-registry-root <consolidated registry 根目录> \
  --approvals <approved_asr_rows.jsonl>
```

approval 文件每行格式为：

```json
{"global_id": 40, "asr_id": "ASR00001"}
```

| 门槛 | 默认值 | 含义 |
| --- | --- | --- |
| `--min-voice-coverage` | `0.5` | registry turn 覆盖 ASR 区间的最低比例 |
| `--min-voice-purity` | `0.6` | 主导声纹在已归属语音中的最低占比 |
| `--min-voice-net-seconds` | `2.0s` | 去掉 simultaneous speech 后的最低单人语音时长 |
| timebase tolerance | `0.5s` | 超出 registry 音频时长后的时间基不匹配容差 |

`speaker_voice_asset_pipeline.py` 是较重的本地质量链：MossFormer2_SE 前景分离 → Stage5/CAM++ 建库 → foreground gate → pyannote overlap 检测 → pooled consolidation。简单预览链不依赖它；但按当前 `character_role_aligner.py` 实现，要向最终角色目录复制正式 voice，就必须有它的 consolidated registry、Stage B 生成的 `voice_asset_manifest.jsonl` 和人工 approval。

## 4. 关键物体资产链

### 4.1 Stage O0：候选队列

推荐输入是 Stage1.5 写出的：

```text
<alignment>/object_grounding_candidates.jsonl
schema_version = object-grounding-candidates-v2
```

每条候选包含 shot-local `candidate_id`、`shot_id`、`canonical_name`、grounding phrases、priority 和 `visible_time_range_sec`。时间窗是 shot-relative，后续只在对象明确可见的区间采帧。

也可以直接从 shot caption 投影候选：

```bash
"$PY" "$EFP/assets/key_object/build_object_grounding_queue_v2.py" \
  --shot-captions <shot_captions.jsonl> \
  --output <object_grounding_candidates.jsonl> \
  --priorities must high
```

该脚本 docstring 将自身定义为“未经过 Stage1.5 校验的评测适配器”。它适合消融或独立物体评测；完整链优先使用 Stage1.5 输出。

### 4.2 Stage O1：Grounding + SAM 3.1 分割

入口：

```bash
"$PY" "$EFP/assets/key_object/object_grounding_segmenter_v3.py" \
  --candidates <object_grounding_candidates.jsonl> \
  --shots <shots.jsonl> \
  --output <workspace>/assets/key_object/grounding
```

`shots.jsonl` 必须能按 `shot_id` 找到对应 `clip_path` 或 `video_path`。

#### 处理逻辑

```text
在 visible_time_range_sec 内均匀采帧
  → GroundingDINO 对多个 phrase 检测
  → 按 IoU 聚类同一实例 proposal
  → 首轮失败/歧义时在同一时间窗内优先补采清晰帧
  → 跨帧 instance track 仲裁
  → SAM 3.1 根据最终 bbox 出 mask
  → 写 frame/mask/crop/rgba/overlay
```

补采不会扩大时间窗、改变 caption 候选或提升候选 priority。

#### 输出

```text
grounding/
  object_observations.jsonl
  summary.json
  frames/*.jpg
  masks/*.png
  crops/*.jpg
  rgba/*.png
  overlays/*.jpg
```

observation 的常见 `status` 包括 `ok`、`no_detection`、`no_mask`、`ambiguous_instance`、`instance_track_absent` 和 `shot_missing`。只有 `status=ok` 的行才是后续 reference set 的有效来源。

#### 关键默认值

| 参数 | 默认值 |
| --- | --- |
| `--samples-per-shot` | `5` |
| `--box-threshold` | `0.20` |
| `--text-threshold` | `0.15` |
| `--cluster-iou` | `0.75` |
| `--window-resample-max` | `4` |
| `--window-resample-scan` | `64` |
| `--track-link-iou` | `0.30` |
| `--min-track-support-frames` | `2` |
| `--track-ambiguity-tolerance` | `0.05` |

默认写 `object-observation-v4`；`--window-resample-max 0` 会关闭补采和多帧选择，恢复 v3 schema/行为。

### 4.3 Stage O2：参考图集合

该模式只读已经完成的 grounding 输出，不重跑检测和分割：

```bash
"$PY" "$EFP/assets/key_object/object_grounding_segmenter_v3.py" \
  --build-reference-sets \
  --grounding-dir <workspace>/assets/key_object/grounding \
  --candidates <object_grounding_candidates.jsonl>
```

默认写 `<grounding-dir>/reference_sets.json`。每个候选目标选择 3–5 张有序参考图，实际不足时标记为 `reduced` 或 `empty`。

选择规则：

- 只从 `status=ok`、非 `ambiguous_instance`、非 mask 面积离群的帧中选。
- `mask_area_ratio > 0.8` 视为全帧主导离群；同时应用组内 Tukey IQR fence。
- 第一张优先选择对同候选其他帧最具代表性的前景 DINOv3 embedding。
- 后续使用 greedy marginal gain 增加视角信息。
- 前景 cosine similarity `>0.95` 视为近重复并降低增益。
- `--max-references` 默认 `5`。

## 5. 关键场景资产链

入口：

```bash
"$PY" "$EFP/assets/scene/important_scene_frame_extractor.py" \
  <global_clip_or_video.mp4> \
  --output-dir <workspace>/assets/scene \
  --max-scenes 12 \
  --frames-per-scene 3
```

也支持 `--input-dir <dir> --pattern '*.mp4'` 批量处理。每个输入视频写入独立 `<output-dir>/<video_stem>/`。

### 模型和处理逻辑

本链完全本地、纯视觉，不调用 LLM/VLM/远程 API，也不读取 shot/segment/global caption：

```text
按时间均匀采样候选帧
  → YOLOv8n 检测人物与保守的 COCO 结构类
  → 可选 tiled inference 补小人物
  → 抑制人物区域后提取 DINOv3 patch embedding
  → average-linkage cosine 场景聚类
  → 清晰度、曝光、构图、结构遮挡、代表性、稳定性、视差代理加权打分
  → 每个重要场景选 main + aux anchors
```

默认权重：

| 分量 | 权重 |
| --- | --- |
| person clear | `0.22` |
| structure clear | `0.22` |
| sharpness | `0.16` |
| representativeness | `0.15` |
| exposure | `0.10` |
| composition | `0.10` |
| stability | `0.03` |
| parallax proxy | `0.02` |

### 输出

```text
scene/<video_stem>/
  all_scenes_preview.jpg
  candidate_scores.json
  scene_index.json
  scene_001/
    anchor_main.jpg
    anchor_aux_01.jpg
    anchor_aux_02.jpg
    person_mask_main.png
    anchor_main_debug.jpg
    topk_preview.jpg
    scene_meta.json
```

`scene_index.json` 当前 schema 为 `important-scene-index-v3`，记录模型策略、采样信息、打分公式、场景聚类阈值、每张锚点的时间戳与 DINO 相似度。

### 关键默认值

| 参数 | 默认值 |
| --- | --- |
| YOLO person confidence | `0.18` |
| structural class confidence | `0.30` |
| image size | `1280` |
| batch size | `12` |
| tile overlap | `0.18` |
| sample interval | `1.2s` |
| min / max candidates | `12 / 600` |
| scene similarity | `0.65` |
| aux-to-main similarity | `0.78` |
| min scene frames | `2` |
| min scene quality | `0.25` |
| max scenes | `12` |
| frames per scene | `3`，包含 main 在内的目标总数 |
| minimum gap | `2.0s` |

`person_mask_main.png` 是人物 bbox union 的保守 proxy，不是实例分割 mask；结构物体只覆盖脚本列出的 COCO 类；`parallax_potential_score` 也是外观变化代理，不是相机标定结果。

## 6. 推荐串联顺序

### 6.1 依赖关系

```text
切镜 / global 聚合 / ASR
  → Scheme A global + shot + segment caption
      ├─ 角色 Stage1.5
      │    ├─ 角色 Stage3 → Stage4 → Stage5
      │    └─ 物体 Stage O1 → 可选 O2
      └─ 声音 Stage B → 可选 registry/approval → Stage5 voice

完整视频或 global clip
  └─ 场景 Stage S0（可与角色/物体链并行）
```

角色 Stage1 只依赖 shot clips，可以在 caption 生成期间提前运行。Stage1.5 必须等待 Tracklet 和 caption；Stage3 必须使用与 Stage5 `--stage3-view` 相同的 policy view。

### 6.2 复用既有 caption 的角色链

仓库内可直接参考：

```text
workspace/movie_full_pipeline_20260912_v2/run_v2_asset_pipeline.sh
```

它复用 `middle/`、`segments/`、`captions_speaker/` 和声音预览，只重跑：

```text
Stage1 Tracklet
  → Stage1.5 MiMo 对齐
  → Stage3 CHR
  → Stage4 bank
  → Stage5 CHR↔S-ID
```

该脚本目前把 `--output` 设为 `$WS/characters`，所以实际角色目录是 `$WS/characters/characters/...`；这是参数与脚本内部固定 `characters/` 子目录叠加的结果。

### 6.3 物体与场景可以独立补跑

- 已有 Stage1.5 时，直接使用 `alignment/object_grounding_candidates.jsonl` 重跑物体 O1/O2。
- 只想评测 shot caption 物体召回时，用 `build_object_grounding_queue_v2.py` 生成独立队列。
- 场景链只需要视频，可完全独立运行；改变 caption 或角色聚类不要求重跑场景。

## 7. 可选 I2I bridge export

`character_role_aligner.py` 还提供独立的只读导出模式：

```bash
"$PY" "$EFP/assets/character_role_aligner.py" \
  --export-i2i-bundles <workspace> \
  --i2i-output-root <output_root>
```

它读取 workspace 已有的 scene、key object、segment/shot/global caption 和 `assets/role_alignment.json`，新增：

```text
<output_root>/assets/scene/scene_segment_index.json
<output_root>/assets/i2i_bundles/scene/*.json
<output_root>/assets/i2i_bundles/object/*.json
```

该模式不会重跑资产提取，也不会触发 CHR→S-ID 投票。省略 `--i2i-output-root` 时默认写回传入 workspace。也可以在正常角色对齐命令中同时传 `--export-i2i-bundles`，对齐结束后追加导出。

## 8. 跑后校验

建议按层检查，不要只看最终目录是否存在。

| 层 | 检查文件 | 关键不变量 / 异常信号 |
| --- | --- | --- |
| Tracklet | `tracklets/manifest.json` | `status=ok`；`shots_completed` 与输入 shot 数匹配；检查 `errors[]` 和 `filtered_tracklet_count` |
| 语义对齐 | `alignment/manifest.json` | 查看 `status_counts`、失败 shot；确认目标 policy view 下确有 `tracklet_signatures.json` |
| CHR | `identity/registry.json` | `cannot_link_violation_count` 应为 `0`；分别查看 `characters[]` 与 `pending_candidates[]` |
| 视觉 bank | `asset_bank/registry.json` | character 数应与已晋升 CHR 对得上；逐 CHR 检查 `assets.json.rejected_assets` |
| 角色对齐 | `role_alignment.json` | 查看 `chr_to_roles` 的 confirmed/weak/contested、`id_conflicts`、`unassigned_visual_identities` |
| 声音预览 | `voice_by_role/summary.json` | `cut_failed_count=0`；valid/invalid 数与源 ASR 行数之和一致 |
| 正式声音 | `voice_asset_manifest.jsonl` | 最终需要 approved + matched；仅存在 `valid/*.wav` 时角色 `voice/` 为空是当前设计行为 |
| 物体 | `grounding/summary.json` | `successful_observation_count>0`；重点检查 ambiguous/no_detection/no_mask 分布 |
| 物体参考集 | `reference_sets.json` | 检查每个候选的 completeness=`full/reduced/empty` 与离群原因 |
| 场景 | `scene_index.json` | `returned_scene_count>0`；锚点文件存在；`relation_constraint_satisfied=true` |

角色侧还应检查这三个关系：

1. Stage3 与 Stage5 使用同一份 Stage1.5 policy view。
2. 每个 `role_to_chr` 指向的 CHR 在 Stage4 `asset_bank/characters/` 中存在。
3. 最终目录路径按 `<output>/characters/...` 解释，避免把双层 `characters/characters/` 误判为缺产物。

## 9. 已知约束与常见坑

1. **没有统一的一键 all-assets orchestrator。** `mimo_video_pipeline_scheme_a.py --stage all` 只跑三层 caption；现有 `run_v2_asset_pipeline.sh` 只编排角色视觉与对齐。物体、场景和正式 voice approval 需要按本文件单独串联。
2. **Stage 编号是历史命名。** 角色链有 Stage1、Stage1.5、Stage3、Stage4、Stage5，没有独立 Stage2；单帧 OBS 有时被文档称为 Stage0/Stage1，但生产 builder 已将其内联。
3. **文件版本名与运行 schema 不完全一致。** `semantic_role_tracklet_refiner_v3.py` 复用 v2；`object_grounding_segmenter_v3.py` 默认输出 v4。
4. **物体候选默认是 shot-local。** 当前没有自动把不同 shot 的 `OC` 合并成一个全局物体身份。
5. **场景是外观聚类。** 它不理解剧情地点名，也不使用 shot boundary；相似布景可能被合并，快速闪现且短于采样间隔的场景可能漏掉。
6. **CHR 与 S-ID 粒度不同。** 同一语义角色可能因背影、换装、遮挡被拆成多个 CHR；同一演员的相近外观也可能让不同剧情角色接近。Stage5 的报告应保留这些冲突，不应直接改 JSON 强行合并。
7. **声音预览与正式资产是两层。** `valid_manifest.jsonl` 不是 approval manifest；没有 `voice_asset_manifest.jsonl` 时 Stage5 不复制声音。
8. **输出目录由“参数根 + 固定子目录”组成。** Stage4 固定追加 `characters/CHR...`，Stage5 固定追加 `characters/<scope>/<S-ID>`。写运行脚本时先展开最终路径。
9. **相对模型路径依赖 cwd。** 现有跑法先 `cd` 到 `general_video_annotation_level3_v2` 根目录，并把 `third_party/sam3_src` 放进 `PYTHONPATH`。

## 10. 下游依赖速查

| 下游需要什么 | 应读取的产物 |
| --- | --- |
| shot 内人物轨迹和可视化 | `tracklets/shots/<SHOT>/manifest.json`、`tracklet_overlay.mp4` |
| Tracklet 的语义角色 | `alignment/alignments/*.json` 或 `semantic_tracklet_index.json` |
| 跨 shot 视觉身份 | `identity/registry.json` |
| 可直接选图的角色画廊 | `asset_bank/characters/CHRxxx/` |
| 语义角色到视觉身份映射 | `role_alignment.json::role_to_chr` |
| 统一角色目录 | `<output>/characters/<scope>/<S-ID>/role_character.json` |
| ASR 句级试听 | `voice_by_role/{valid,invalid}/.../*.wav` |
| 可进入最终角色目录的声音 | `voice_asset_manifest.jsonl` 中 approved + matched 行指向的 `reference_audio` |
| 物体 mask / RGBA / crop | `grounding/object_observations.jsonl` 中 `status=ok` 行 |
| 物体多图参考 | `grounding/reference_sets.json` |
| 场景锚点 | `scene/<video_stem>/scene_index.json` 和 `scene_###/anchor_*.jpg` |
| I2I scene/object 组装输入 | `assets/scene/scene_segment_index.json`、`assets/i2i_bundles/{scene,object}/` |

## 11. 关联文档

| 文档 | 适合解决的问题 |
| --- | --- |
| [`CHARACTER_ASSET_CHAIN_README.md`](./CHARACTER_ASSET_CHAIN_README.md) | 快速理解角色 CHR、S-ID、Tracklet 和最终目录 |
| [`PIPELINE_README.md`](./PIPELINE_README.md) | 从切镜到三级 caption 再到资产的完整仓库说明 |
| [`MV_AnnotationCaption_Pipelines.md`](./MV_AnnotationCaption_Pipelines.md) | MV caption/ASR 上游输入及其格式 |
| [`ASSET_EXTRACTION_QUICKSTART.md`](./ASSET_EXTRACTION_QUICKSTART.md) | 资产相关关键文件的短导读 |
| [`PIPELINE_COMPARISON.md`](./PIPELINE_COMPARISON.md) | Scheme A/B 和不同 segmentation 方案的区别 |
| [`I2I_ASSET_OPTIMIZATION_PLAN.md`](./I2I_ASSET_OPTIMIZATION_PLAN.md) | scene/object/character 资产向 I2I 下游交付的优化设计 |

一句话概括：角色链先在 shot 内找到人并形成 Tracklet，再借 caption 对齐 S-ID、跨 shot 聚成 CHR、筛选画廊，最后用加权投票物化角色；物体链从带可见时间窗的 caption 候选出发做 GroundingDINO + SAM 3.1；场景链独立用 DINOv3 + YOLOv8n/OpenCV 选场景锚点；声音预览只有经过 consolidated registry 关联和人工 approval 后才会进入最终角色目录。
