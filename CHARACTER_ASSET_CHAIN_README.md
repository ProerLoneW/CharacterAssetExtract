# 角色资产提取链路导读

> 范围**只**覆盖"角色资产"——也就是视觉的 CHR（Character ID）与音频的 S-ID（Speaker ID），以及两者如何对齐。
>
> 不覆盖：场景锚点、关键物体、segment/shot/global 三级 caption、scene/object 资产结构（这些见 [`ASSET_EXTRACTION_QUICKSTART.md`](./ASSET_EXTRACTION_QUICKSTART.md) 与 [`PIPELINE_README.md`](./PIPELINE_README.md)）。
>
> 读完这份文档 + 第 1 节列的 8 个文件，能讲清楚"一段视频进来怎么变成 `characters/<scope>/<S_ID>/{visual,voice}/` 下的角色资产"。

---

## 0. 30 秒建立全景

角色资产链是 **Scheme A 的核心**：caption 模型（MiMo）一次性把"画面里在说话的人"和"ASR 时间戳"统一到同一个 S-ID 命名空间，**视觉走 tracklet → embedding 聚类**得到 CHR，**音频走 ASR 句级切片**得到 wav，最后用 `character_role_aligner` 把 CHR 投票映射到 S-ID，物化成统一目录。

```
视频 + ASR
   │
   ├──[caption 链]──► MiMo 三级 caption（global/shot/segment）──► 给出 S-ID 与对话归属
   │                  （详见 caption 链路文档，不在本导读范围）
   │
   ├──[视觉链]─────► shot_tracklet_builder ──► semantic_role_tracklet_refiner_v3
   │                                                │
   │                                                ▼
   │                              visual_character_registry_builder (CHR001/002…)
   │                                                │
   │                                                ▼
   │                                  character_asset_bank (face/body 画廊)
   │
   ├──[音频链]─────► export_all_asr_audio_review ──► assets/voice_by_role/valid/<scope>/<S_ID>/*.wav
   │
   └──[对齐]───────► character_role_aligner ──► characters/<scope>/<S_ID>/{visual,voice,role_character.json}
```

**两个关键 ID 体系**：
- **CHR**（Character ID，`CHR001/CHR002/…`）：视觉身份的"外观"聚类。同一张脸/同一套衣服 = 同一个 CHR，**不区分戏里戏外、不区分场景**。
- **S-ID**（Subject ID，`S1/S2/…`）：caption 模型给出的"剧情中的角色"。同一个 S-ID 可能跨越多个 CHR（正脸 vs 背影）。

`character_role_aligner` 的工作就是**把 CHR 投票映射到 S-ID**，然后物化统一目录。

---

## 1. 必读的 8 个文件

按"先骨架后细节"顺序读；每个文件只需读**顶部 docstring**（通常 30-80 行）就够建立认知。

| # | 路径 | 读完能讲清什么 | 关键概念 |
|---|---|---|---|
| 1 | [`assets/visual_person/entity_asset_extractor.py`](./assets/visual_person/entity_asset_extractor.py) | 单帧画面里的"人"怎么被检测/识别/打码 | Sa2VA + GroundingDINO + InsightFace + SAM 3.1 的最小组合，输出 frame-local `OBS` |
| 2 | [`assets/visual_person/shot_tracklet_builder.py`](./assets/visual_person/shot_tracklet_builder.py) | 单镜头内多个 OBS 怎么连成 tracklet | ACTIVE / LOST / FINISHED 状态机；tracklet = 同一个人的多帧追踪 |
| 3 | [`assets/visual_person/semantic_role_tracklet_refiner_v3.py`](./assets/visual_person/semantic_role_tracklet_refiner_v3.py) | 每条 tracklet 怎么被 MiMo 决策对齐到 S-ID | overlay 视频 + caption 证据 → tracklet→S-ID 对齐（v3 是生产版，v1/v2 是迭代历史） |
| 4 | [`assets/visual_person/visual_character_registry_builder.py`](./assets/visual_person/visual_character_registry_builder.py) | 多个 tracklet 怎么合并成 CHR | face + body embedding 相似度 + cannot-link 优先 complete-link 聚类；输出 `identity/registry.json` |
| 5 | [`assets/visual_person/character_asset_bank.py`](./assets/visual_person/character_asset_bank.py) | CHR 的视觉资产怎么落盘 | face/body × 视角 × appearance_state 画廊 + 纯度/去重/竞争；最终目录 `asset_bank/CHRxxx/` |
| 6 | [`assets/voice/export_all_asr_audio_review.py`](./assets/voice/export_all_asr_audio_review.py) | 音频怎么按 ASR 句级切 wav | 16kHz mono wav → `voice_by_role/valid/<scope>/<S_ID>/*.wav`；生成三份 manifest 供人工复核 |
| 7 | [`assets/character_role_aligner.py`](./assets/character_role_aligner.py) | CHR 与 S-ID 怎么对齐 + 怎么物化统一目录 | 加权投票（按 tracklet/shot）+ id_conflict 仲裁 + 物化 `characters/<scope>/<S_ID>/{visual,voice,role_character.json}` |
| 8 | [`mimo_global_captioner_speaker_v2.py`](./mimo_global_captioner_speaker_v2.py) | "音频来源"怎么产生 S-ID | Scheme A 一次性 global caption + ASR speaker attribution；是 `export_all_asr_audio_review` 输入端 |

> **如果只读 3 个文件**：第 3（视觉决策）+ 第 6（音频切片）+ 第 7（对齐物化）——这三步就是角色资产的"产线"。

---

## 2. 数据流（每一步的产物形态）

下面每一行 = 一个 stage，输入输出都列出，方便定位"这个产物在哪个阶段出来"。

### 视觉链（4 个 stage）

| Stage | 脚本 | 输入 | 输出 | 关键 schema / 文件 |
|---|---|---|---|---|
| **Stage0** 单帧 OBS | `entity_asset_extractor.py` | 视频帧 | `observations/`（frame-local entity 检测结果） | `OBS` 字典（person bbox + face embedding + mask） |
| **Stage1** tracklet | `shot_tracklet_builder.py` | `observations/` + 视频帧 | `assets/visual_person_v2/tracklets/shots/SHOT_*/` | 每 shot 多个 `tracklet_NNN.json`（ACTIVE/LOST/FINISHED 状态机） |
| **Stage1.5** semantic align | `semantic_role_tracklet_refiner_v3.py` | tracklet + global/shot/segment caption | `assets/visual_person_v2/alignment/` | `alignments/*.json`（每条 tracklet → S-ID 决策）+ `manifest.json`（ok/失败计数） |
| **Stage3** CHR 聚类 | `visual_character_registry_builder.py` | alignment 输出 + face/body embedding | `assets/visual_person_v2/identity/` | `registry.json`（CHR 列表 + cannot-link 一票否决）+ `candidates/` |
| **Stage4** 资产库 | `character_asset_bank.py` | identity + alignment | `assets/visual_person_v2/asset_bank/CHRxxx/{face,body}/<view>/` | face/body × view × state 画廊 |

### 音频链（2 个 stage）

| Stage | 脚本 | 输入 | 输出 |
|---|---|---|---|
| **StageA** S-ID 归属 | `mimo_global_captioner_speaker_v2.py` | video + ASR | `captions_speaker/global/*.json`（含 `subject_id`、`speaker_attributions`） |
| **StageB** wav 切片 | `export_all_asr_audio_review.py` | ASR + caption 中的 subject_id | `assets/voice_by_role/valid/<scope>/<S_ID>/*.wav`（16kHz mono） + 三份 manifest |

> **legacy 音频链**（不推荐）：`speaker_voice_asset_pipeline.py`（MossFormer2_SE 前景分离 → Stage5 CAM++ 声纹 → pool）——只读 docstring 了解其存在即可，**生产链走 Scheme A 路径**。

### 对齐链（1 个 stage）

| Stage | 脚本 | 输入 | 输出 |
|---|---|---|---|
| **Stage5** CHR↔S-ID | `character_role_aligner.py` | identity + alignment + asset_bank + voice_by_role | `characters/<scope>/<S_ID>/{visual,voice,role_character.json}` + `role_alignment.json` |

---

## 3. 三件容易搞混的事

### 3.1 CHR vs S-ID 不是 1-to-1

- **CHR 粒度更细**：同一 S-ID 在不同光照/姿态/遮挡下可能被聚成多个 CHR（如 v2-new 跑出 global_0042 三个 CHR003/004/005 都映射 S1）。
- **CHR 不区分场景**：同一个演员演两个角色，v2 不会自动拆分（除非 cannot-link 触发）；当前 cannot_link_count 通常为 0，所以**几乎都是按"外观"聚**。
- **CHR 不一定都能物化**：单 CHR 单 shot 通常 `status=weak`，不会被物化到统一目录（参见 v1 vs v2 对比报告里 global_0040 的 CHR001 退化案例）。

### 3.2 三个 `semantic_role_tracklet_refiner` 版本

仓库里有 v1 / v2 / v3 三个版本：
- **v1**：`semantic_role_tracklet_refiner.py`（最老，line 数 966）
- **v2**：`semantic_role_tracklet_refiner_v2.py`（迭代版本，line 数 983）
- **v3**：`semantic_role_tracklet_refiner_v3.py`（**生产版**，`run_v2_asset_pipeline.sh:69` 用的是这个）

**不要直接编辑 v1/v2**——它们是历史快照；改 v3。

### 3.3 物化目录的 `characters/` vs `assets/characters/`

- `assets/characters/`：历史物化路径（v1 原版用），角色按 `<scope>/<S_ID>/` 组织。
- `characters/`：v2-new 输出路径（**注意当前 v2 跑出来的实际是 `characters/characters/<scope>/<S_ID>/`，多嵌了一层**——这是 `character_role_aligner.py` 内部路径 bug，下游消费 `role_alignment.json` 不受影响）。

---

## 4. 怎么验证角色资产跑对了？

跑完一次后，对照下面 4 个不变量检查产物（推荐写成 smoke script）：

| 不变量 | 怎么查 | 异常信号 |
|---|---|---|
| 1. 每个 CHR 至少 1 个 tracklet → 1 个 S-ID | `cat identity/registry.json \| jq '.characters[].tracklet_count'` | 有 CHR 的 tracklet_count=0 |
| 2. 每个物化的 role 至少 1 个 wav | `find characters/<scope>/<S_ID>/voice -name '*.wav' \| wc -l` | wav 数 = 0 |
| 3. id_conflicts 应当非空且被仲裁 | `jq '.id_conflicts' role_alignment.json` | 冲突 > 5 且没有 `materialized` 字段 |
| 4. CHR 数 ≥ role 数 | 比较 `registry.character_count` 与 `role_alignment.character_count` | 反过来 = 物化时丢了角色 |

---

## 5. 想"重跑角色资产但复用 caption/ASR"？

参考 v2 workspace 的搭建方法：`workspace/movie_full_pipeline_20260912_v2/run_v2_asset_pipeline.sh`。

**核心 trick**：把上游已完成的中间产物（`middle/`、`captions_speaker/`、`segments/`、`assets/voice_by_role/`）做成**软链**指向原工作区，只重跑 Stage1 → 1.5 → 3 → 4 → 5。

```bash
# 软链上游产物
ln -s ../../movie_full_pipeline_20260912/middle middle
ln -s ../../movie_full_pipeline_20260912/captions_speaker captions_speaker
ln -s ../../movie_full_pipeline_20260912/segments segments
ln -s ../../movie_full_pipeline_20260912/assets/voice_by_role voice_by_role_src

# 只跑 5 个 stage
python assets/visual_person/shot_tracklet_builder.py \
  --shots-dir middle/shot_clips --output assets/visual_person_v2/tracklets --continue-on-error
python assets/visual_person/semantic_role_tracklet_refiner_v3.py \
  --tracklets assets/visual_person_v2/tracklets \
  --global-captions captions_speaker/global \
  --shot-captions captions_speaker/shot \
  --segment-captions captions_speaker/segment \
  --output assets/visual_person_v2/alignment --concurrency 2 --skip-existing
python assets/visual_person/visual_character_registry_builder.py \
  --inputs assets/visual_person_v2/alignment/stage3_inputs/primary \
  --vlm-decisions assets/visual_person_v2/alignment/stage3_inputs/primary/semantic_identity_decisions.json \
  --output assets/visual_person_v2/identity
python assets/visual_person/character_asset_bank.py \
  --identity assets/visual_person_v2/identity --output assets/visual_person_v2/asset_bank
python assets/character_role_aligner.py \
  --identity assets/visual_person_v2/identity \
  --stage3-view assets/visual_person_v2/alignment/stage3_inputs/primary \
  --asset-bank assets/visual_person_v2/asset_bank \
  --voice-export voice_by_role_src \
  --output characters
```

典型耗时（59 shot 一次）：47-50 分钟（其中 Stage1 占 35-40 分钟，其余 ~10 分钟）。

---

## 6. 关联文档

| 想了解什么 | 看哪份 |
|---|---|
| 整个仓库（caption + 资产 + 切镜）全流程 | [`PIPELINE_README.md`](./PIPELINE_README.md)（§0-§14） |
| 10 文件全景导读（含 caption/物体/场景） | [`ASSET_EXTRACTION_QUICKSTART.md`](./ASSET_EXTRACTION_QUICKSTART.md) |
| Scheme A vs Scheme B / MiMo-cut vs window-cut 差异 | [`PIPELINE_COMPARISON.md`](./PIPELINE_COMPARISON.md) |
| 角色资产面向 I2I 下游的优化方向 | [`PROMPT_i2i_asset_optimization.md`](./PROMPT_i2i_asset_optimization.md) |
| 视觉/音频资产优化规划 prompt（通用） | [`PROMPT_asset_optimization_planning.md`](./PROMPT_asset_optimization_planning.md) |

---

## 一句话总结

> 角色资产链 = **视觉（shot_tracklet → semantic_role_tracklet_refiner_v3 → visual_character_registry_builder → character_asset_bank）+ 音频（mimo_global_captioner_speaker_v2 → export_all_asr_audio_review）+ 对齐（character_role_aligner）**。视觉给 CHR，音频给 S-ID，`character_role_aligner` 把 CHR 投票映射到 S-ID 物化出 `characters/<scope>/<S_ID>/{visual,voice,role_character.json}`。改任何一环时，**先看顶部 docstring 再看代码**——大多数设计意图都写在 docstring 里。
