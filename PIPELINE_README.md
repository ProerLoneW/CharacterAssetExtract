# extract_full_pipeline —— 视频资产提取与三级 Caption 流水线导览

> **路径**位置**：**`general_video_annotation_level3_v2/extract_full_pipeline/`
> **目的**：把根目录下分散的子流程整合成一个模块化的"视频 → 切镜 → 聚合 → 三级 caption → 资产提取"端到端系统。
> **读者**：需要复现、调试或扩展该流水线的研究员 / 工程师。

本文档分两部分：

- **第 1 ~ 10 章**给出端到端流程图、文件-产物映射表、每一步详细逻辑、默认阈值与边界规则；
- **第 11 ~ 14 章**说明资产侧（视觉/音频/关键物体/CHR↔S-ID 对齐）的设计思想与运行顺序。

精简版另见 [`ASSET_EXTRACTION_QUICKSTART.md`](./ASSET_EXTRACTION_QUICKSTART.md)，只列出"读哪些文件能快速上手"。
"MiMo 切 vs 窗口切"两套 segment pipeline 的对比见 [`PIPELINE_COMPARISON.md`](./PIPELINE_COMPARISON.md)（什么时候用哪条、怎么接到一起）。

---

## 0. 主链路（一图流）

```text
原始视频 (raw.mp4)
   │
   │ ① 预处理（可选）：去黑场/静默/低动态
   │      → preprocessing/preprocess.py
   ▼
processed.mp4
   │
   │ ② 镜头切分
   │      → shot_splitter.py  (TransNetV2 / PySceneDetect)
   ▼
<work_dir>/shots/              shots.jsonl + clips/<video>_shot_NNNNN.mp4 + summary.json
   │
   │ ③ 聚合为 60-120s 的 global clip（保证音视频流连续）
   │      → global_clip_aggregator.py  (可选 SCRL 场景边界)
   ▼
<work_dir>/global/             global_clips.jsonl + global_clips/<video>_global_NNNN.mp4 + summary.json
   │
   │ ④ Global Caption（MiMo 看 1-2 min global clip）
   │      → mimo_global_captioner.py     (V3 默认 prompt, subject_registry / event_timeline)
   │      → mimo_global_captioner_speaker_v2.py  (Scheme A：同时做 ASR → S-ID 归属)
   ▼
<work_dir>/captions/global/    global_captions.jsonl + global_captions/global_NNNNN.json + raw_responses/
   │
   │ ⑤ Shot Caption（MiMo 看单个 shot 片段，复用 global 的 S-ID）
   │      → mimo_shot_captioner.py        (V1 调度)
   │      → mimo_shot_captioner_v2.py     (V2：增加 key_object_candidates 合约)
   │      → mimo_shot_captioner_v3.py     (V3 = V2 prompt 替换 + schema_version 升级)
   ▼
<work_dir>/captions/shot/      shot_captions.jsonl + shot_captions/shot_NNNNN.json + raw_responses/
   │
   │ ⑥ Segment Caption（MiMo 把一个 global 内的 shot 切分成 15-30s segment + 描述）
   │      → mimo_segment_captioner.py         (V1 基底，被 V2/V3/V4 monkey-patch)
   │      → mimo_segment_captioner_v2.py      (V2 当前默认：增加 duration_contract + object_candidate_inventory)
   │      → mimo_segment_captioner_global_speaker_v3.py  (Scheme A 中段：归属由 global 决定)
   │      → mimo_segment_captioner_fixed_segments_v4.py  (Scheme A 末段：段边界由上游固定)
   ▼
<work_dir>/captions/segment/   segment_captions.jsonl + segments/global_NNNN.json + raw_responses/
   │
   │ ⑦ 按 segment 时间窗切出细粒度 mp4
   │      → mimo_segment_cutter.py
   ▼
<work_dir>/segments/           segments.jsonl + clips/<video>_global_NNNN_segment_NNN.mp4 + summary.json

────────────────────────────────────────────────────────────────────────
   │
   │ ⑧ 资产提取（依赖 caption 产物，**可与主链路并行**）
   ▼
   8.1 关键物体 ── assets/key_object/{build_object_grounding_queue_v2, object_grounding_segmenter_v3}.py
   8.2 人物视觉（推荐链）── assets/visual_person/{shot_tracklet_builder, semantic_role_tracklet_refiner_v3,
                          visual_character_registry_builder, character_asset_bank}.py
   8.3 人物音频（Scheme A 推荐）── mimo_global_captioner_speaker_v2.py + assets/voice/export_all_asr_audio_review.py
   8.4 旧式音频池 ── assets/voice/speaker_voice_asset_pipeline.py  (MossFormer2_SE → Stage5 → pool)
   8.5 ID 统一 ── assets/character_role_aligner.py  (CHR → caption S-ID 投票 + 物化角色目录)
```

编排器（任选其一）：

| 编排器 | 入口 | 适用场景 |
|---|---|---|
| **主编排**（** | `mimo_video_pipeline.py` | 单视频跑 split → aggregate → global → shot → segment → cut 六步 |
| Scheme A 编排 | `mimo_video_pipeline_scheme_a.py` | 结构（shots/segments）已在上游生成，只跑 caption 三级 + 说话人归属 |
| 单步 CLI | 各 stage 脚本 | 单独重跑或 debug |
| legacy runner | `assets_runner.py` | segment 级 stub 演示，**不是**当前生产推荐（见 ARCHITECTURE.md 4.13/4.14） |

---

## 1. 工作区目录约定

`common/workspace.py` 给出 `< root >/<video_id>/` 风格（也可加 `--flat` 走扁平布局）：

```text
<workspace>/
├── source/            源片与 25fps 重采样版
├── shots/             ② 产物（JSONl + 切镜 mp4 + summary.json）
├── clips/             最终 clip 宇宙（视频/音频文件）
├── faces/             ① 抽帧、人脸、embedding、代表脸图集
├── annotations/
│   ├── clips.jsonl      Schema 一级
│   ├── segments.jsonl   Schema 二级
│   ├── global.jsonl     Schema 三级（单行）
│   └── qc_report.json   质检汇总
├── logs/              StageTimer 累计日志 + timings.json
└── captions/          三级 caption 产物（mimo_video_pipeline 的扁平约定）
```

主链路用 **`--flat`** 后的约定（`mimo_video_pipeline.py` 默认）：

```text
<work_dir>/
├── shots/                shots.jsonl + clips/*.mp4 + summary.json
├── global/               global_clips.jsonl + global_clips/*.mp4 + summary.json
├── captions/
│   ├── global/           global_captions.jsonl + global_captions/*.json + raw_responses/
│   ├── shot/             shot_captions.jsonl + shot_captions/*.json + raw_responses/
│   └── segment/          segment_captions.jsonl + segments/*.json + raw_responses/
└── segments/             segments.jsonl + clips/*.mp4 + summary.json
```

注意：传统 `Workspace` 路径约定（`annotations/clips.jsonl` 等）由 `m02_shots/m04_recut/m07_segment` 沿用；新主链路用扁平目录。两套可独立运行，不互相耦合。

---

## 2. 文件 ↔ 产物映射表

| 步骤 | 入口脚本 | 输入 | 主产物 | 关键中间产物 |
|---|---|---|---|---|
| ① 预处理 | `preprocessing/preprocess.py` | 原始 mp4 | `processed_info.json` + 裁后 mp4 | — |
| ② 镜头切分 | `shot_splitter.py` | mp4 | `shots.jsonl` + `clips/*.mp4` + `summary.json` | `shots/<video>_shot_NNNNN.mp4` |
| ②' 切镜导入 | `m02_shots/importer.py` | 镜头清单/目录 | `<ws>/annotations/clips.jsonl` | 邻接表骨架 |
| ③ 聚合 global | `global_clip_aggregator.py` | `shots.jsonl` + 源片 | `global_clips.jsonl` + `global_clips/*.mp4` + `summary.json` | 优先从**源视频**切；兜底 shot concat |
| ④ Global caption | `mimo_global_captioner.py` | `global_clips.jsonl` | `global_captions.jsonl` + `global_captions/global_NNNNN.json` | `raw_responses/global_NNNNN.txt` |
| ④' Global + ASR 归属 | `mimo_global_captioner_speaker_v2.py` | `global_clips.jsonl` + ASR 旁路 | 同上 + `speaker_attributed_asr.jsonl` | `schema_version=global-caption-speaker-attribution-v2` |
| ⑤ Shot caption | `mimo_shot_captioner.py` (默认 v3 prompt) | `shots.jsonl` + global | `shot_captions.jsonl` + `shot_captions/shot_NNNNN.json` | `raw_responses/shot_NNNNN.txt` |
| ⑥ Segment caption | `mimo_segment_captioner_v2.py` (当前默认) | global + shot | `segment_captions.jsonl` + `segments/global_NNNN.json` | `object_candidate_index` + `duration_contract` |
| ⑥' Scheme A 中段 | `mimo_segment_captioner_global_speaker_v3.py` | global_with_speaker + shot | 同上 | 说话人由 global 决定，`dialogue: []` 由本地写入 |
| ⑥'' Scheme A 末段 | `mimo_segment_captioner_fixed_segments_v4.py` | global_with_speaker + shot + 上游 segments.jsonl | 同上 | 段边界由上游固定，MiMo 只填内容 |
| ⑥-A 窗口切（纯算法，可选） | `m07_segment/run.py::run_segment` | `clips.jsonl` + 邻接表 | `<ws>/annotations/segments.jsonl`（Schema 二级） | 含 QualityCard 加权（grade_weight）+ DINOv3 score + unqualified_ratio；**不调 caption** |
| ⑥-B 三路对比（纯算法，可选） | `segmentation/three_way.py` | `global_clips.jsonl` + `shots.jsonl` | `segments_{window,v1,v2}/segments.jsonl` + `summary.json` | 一次性产出 window / v1 / v2 三路；**不调 caption** |
| ⑦ 切 segment | `mimo_segment_cutter.py` | `segment_captions.jsonl` + `global_clips.jsonl` | `segments.jsonl` + `clips/*.mp4` + `summary.json` | `global_NNNN_segment_NNN.mp4`；输入 caption stage 走哪条都行 |
| 8.1 关键物体队列 | `assets/key_object/build_object_grounding_queue_v2.py` | `shot_captions.jsonl` | `object_grounding_queue.jsonl` | 保留 `visible_time_range_sec` |
| 8.1 关键物体分割 | `assets/key_object/object_grounding_segmenter_v3.py` | queue + mp4 | `object_observations.jsonl` + mask/png/bbox.png | GroundingDINO + SAM 3.1 |
| 8.2 视觉 tracklet | `assets/visual_person/shot_tracklet_builder.py` | shots/*.mp + entity_asset_extractor | `tracklet_signatures.json` + `tracklet_overlay.mp4` | ACTIVE/LOST/FINISHED 状态机 |
| 8.2 语义对齐 | `assets/visual_person/semantic_role_tracklet_refiner_v3.py` | tracklet + global + shot + segment caption | `semantic_identity_decisions.json` + `stage3_inputs/{primary,narrative,safe}/` | S-ID 白白名单校验，decision=keep_primary/keep_supporting/keep_context/reject_background/reject_non_person/review |
| 8.2 跨 shot 聚类 | `assets/visual_person/visual_character_registry_builder.py` | tracklet_signatures.json | `registry.json` (CHR001/CHR002…) | face embedding + body embedding + cannot-link 优先 complete-link |
| 8.2 画廊 | `assets/visual_person/character_asset_bank.py` | registry + tracklet features | `asset_bank/<CHR>/<face/body>/<view>/...` | 纯度校验 + pHash/embedding 去重 + top-K |
| 8.3 单帧视觉 | `assets/visual_person/entity_asset_extractor.py` | 抽帧 | `OBS00`…` 资产 + manifest | Sa2VA → GroundingDINO → SAM 3.1 → InsightFace（**仅作为 Stage1 底层**，不复用进资产库） |
| 8.4 音频池（旧式） | `assets/voice/speaker_voice_asset_pipeline.py` | mp4/wav | `consolidated/registry.json` | MossFormer2_SE → Stage5 → overlap → pool |
| 8.4' ASR 音频切 | `assets/voice/export_all_asr_audio_review.py` | global mp4 + speaker_attributed_asr | `valid\|invalid/<global>/<S_ID>/<asr_id>.wav` | 信号统计 + 三份 manifest（人工复核） |
| 8.5 ID 统一 | `assets/character_role_aligner.py` | Stage1.5 + Stage3 + Stage4 + export | `role_alignment.json` + `characters/<scope>/<S_ID>/{visual,voice}/` | CHR→S-ID 投票 |

---

## 3. 详细步骤说明

### 3.1 预处理（可选） `preprocessing/preprocess.py`

**目的**：识别并裁掉片头/片尾中与电影主体无关的片段（黑场/静默/单色/低内容/静态 logo）。

**输入**：原始 mp4
**输出**：

```text
processed_info.json: {source, duration_sec, trimmed_sec, kept_intervals, drops[], processed_video}
```

**关键信号**：

| 信号 | 实现 |
|---|---|
| 黑场 | 连续帧亮度（mean luma）极低 |
| 静默 | 音频 RMS 在连续 N 秒内低于阈值 |
| 单色/低内容 | 帧间亮度方差极低（演职员表/纯色背景） |
| 静态 logo | 前后帧差异极小且持续时间长 |

**调用**：

```bash
python -m extract_full_pipeline.preprocessing.preprocess \
    --input /path/to/source.mp4 \
    --output-dir workspace/processed \
    [--max-head-trim-sec 120] [--max-tail-trim-sec 180]
```

---

### 3.2 镜头切分 `shot_splitter.py`

**目的**：从单视频检测 shot 边界，切出 mp4 clip，落 `shots.jsonl`。

**Pipeline**：

```text
raw_video.mp4
  -> detector (transnetv2 | adaptive | content)
  -> coverage-aware interval post-processing
  -> merge_short_scenes / split_long_scenes
  -> ffmpeg clip export
  -> shots.jsonl
```

**关键参数（默认值）**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--detector` | `auto` | `auto` 优先 TransNetV2，权重缺失时回退 PySceneDetect adaptive；可强制 `content` |
| `--threshold` | `3.0` | PySceneDetect 检测阈值 |
| `--detector-min-scene-sec` | `0.5` | 最小镜头时长（秒） |
| `--min-shot-sec` | `2.0` | 合并阈值（短于此则合并相邻镜头） |
| `--max-shot-sec` | `None` | 长于此强制切分（默认关闭） |
| `--merge-short` | `True` | 是否合并短镜头 |
| `--crf` | `18` | x264 CRF |
| `--transnet-model` | `…/transnetv2-pytorch-weights.pth` | TransNetV2 权重 |
| `--transnet-source-dir` | `m02_shots/teleai_src/`（vendored） | **不** import teleAI-T2V 仓库代码 |
| `--transnet-device` | `auto` | `cuda:0` if available else `cpu` |

**关键函数**：

- `probe_video(video_path, ffprobe_bin)`：ffprobe 拿 `duration_sec` / `fps` / `width` / `height` / `has_audio`。
- `detect_pyscenedetect_scenes(...)`：PySceneDetect `AdaptiveDetector` 或 `ContentDetector`，min_scene_len = `max(1, round(min_scene_sec * fps))`。
- `detect_transnetv2_scenes(...)`：从 vendored `m02_shots/teleai_src/video_split.py` 动态 `importlib` 加载，CPU 模式下给 `torch.load` 注入 `map_location='cpu'`。
- `ensure_coverage(raw_scenes, duration_sec)`：检测器只给 cut 点 → 归一为连续区间（首尾 padding）。
- `merge_short_scenes(scenes, min_sec, max_sec)`：相邻短镜头合并（带 `max_sec` 上限）。
- `split_long_scenes(scenes, max_sec)`：超 `max_sec` 镜头按 `max_sec` 强切（boundary_type=`forced_max_duration``）。
- `export_clip(...)`：ffmpeg `-ss` 在 `-i` 之前（input seeking），`libx264 -crf 18 -preset veryfast`，`+faststart`。

**产物**：

```text
<output>/shots.jsonl        每行一条：shot_id, source_video, clip_path,
                             time_range_sec, duration_sec, frame_range,
                             fps, boundary_type, raw_scene_index
<output>/clips/<video>_shot_NNNNN.mp4
<output>/summary.json       detector/threshold/min/max/shot_count 等
```

**`shots.jsonl` 关键字段**：

```json
{
  "shot_id": 1,
  "source_video": "/abs/path/source.mp4",
  "clip_path": "/abs/path/clips/<video>_shot_00001.mp4",
  "time_range_sec": [0.0, 2.96],
  "duration_sec": 2.96,
  "frame_range": [0, 74],
  "fps": 25.0,
  "boundary_type": "detected_cut" | "merged_short_scene" | "forced_max_duration" | "coverage_padding",
  "raw_scene_index": 0
}
```

**常见问题**：

- 切镜的 `frame_range` 是 `[start_frame, end_frame_inclusive]`（左闭右闭），与 segment cutter 的 `frame_range = [start, end_exclusive-1]` 不混。
- 检测器失败回退顺序：`transnetv2` → `pysSceneDetect adaptive` → `pysSceneDetect content`（如强制 `--detector transnetv2` 失败则直接抛错不回退）。

---

### 3.3 切镜导入（传统邻接表路径） `m02_shots/importer.py`

> 这是 **level3 风格**的导入入口。**主链路（`mimo_video_pipeline`）不依赖它**；只有当你想把外部切镜产物喂给 `Workspace` 邻接表模型（→ `m04_recut`、`m07_segment`、`clip_store.split_clip` 等）时才用。

**两种适配器**：

1. `from_manifest(video_id, manifest_path, ...)`：读镜头清单 JSON `[{"shot_id","start","end","start_frame","end_frame","video_path"}, ...]`，落 `Clips.jsonl`。
2. `from_scandir(video_id, shot_dir, ...)`：扫描切好镜头目录，按文件名自然序 + ffprobe 探测累计偏移推回源片时间轴。

**邻接表规则**：

```text
- 首镜头：prev.clip_id=None, continuous=False
- 末镜头：next.clip_id=None, continuous=False
- 相邻原始镜头（不同 parent）：continuous=False
- ④ 再切割产生的相邻子 clip（同 parent_clip_id）：continuous=True
```

由 `common/clip_store.rebuild_links` 统一重挂指针，避免手工改表出错。

**配套**：

- `m02_shots/teleai_src/`：teleAI-T2V 切镜链路的最小自包含副本（与上游代码仅三处本地适配：`save_cut_with_timeout` 用 `sys.executable`；子进程显式传 `PYTHONPATH`；`teleai_cut_only.py::main_split_fast` 跳过 `main_split_only` 的二次 TransNet 精修）。**运行期不 import teleAI-T2V 仓库**。
- `m02_shots/run_teleai_pipeline.sh`：默认只跑本地 `teleai_src/teleai_cut_only.py`（TransNetV2 单次切镜，不含 audio/base_filter）；加 `--full` 回退 teleAI-T2V 原仓库 `cut_v2,audio,base_filter_v2` 全流水线。

---

### 3.4 聚合 global clip `global_clip_aggregator.py`

**目的**：把 `shots.jsonl` 中的连续 shot 贪心打包为 60~120 秒的 group（保证音视频流连续），可选用 SCRL 做场景边界。

**Pipeline**：

```text
shots.jsonl
  -> SCRL scene boundaries (or metadata / none)
  -> choose_groups(min/target/max_sec)
  -> export from source video via ffmpeg   ← 优先源视频切
  -> global_clips.jsonl
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--min-sec` | `60.0` | 每组最小秒数 |
| `--target-sec` | `90.0` | 目标秒数 |
| `--max-sec` | `120.0` | 每组最大秒数；约束 `0 < min <= target <= max` |
| `--scene-detector` | `auto` | `scrl` / `metadata` / `none` |
| `--scrl-seq-len` | `16` | 4 的正整数倍 |
| `--scrl-threshold` | `0.5` | 边界概率阈值 |
| `--scrl-device` | `auto` | `cuda:0` if available else `cpu` |

**`choose_groups` 算法**：从 start 起向后扩展，累计时长落入 `[min, max]` 的所有候选里选最接近 `target_sec` 的；超出 `max` 终止。最终再尝试把最后一组向前一组合并（如合并后不超过 `max` 且最后一组短于 `min`）。

**`export_mode`**：

- `source_slice`（默认）：从原始视频切，保证音视频流连续。
- `concat_fallback`：当 `shots.jsonl` 中 `source_video` 缺失或找不到文件时，`concat_clips` 流复制失败后回退重编码。

**产物**：

```json
{
  "global_id": 1,
  "source_video": "/abs/source.mp4",
  "clip_path": "/abs/global_clips/<video>_global_0001.mp4",
  "time_range_sec": [0.0, 87.3],
  "duration_sec": 87.3,
  "shot_ids": [1,2,3,4,5,6],
  "shot_count": 6,
  "scene_id": 1,
  "scene_detector": "scrl" | "metadata" | "none",
  "source_shots_manifest": "/abs/shots.jsonl",
  "export_mode": "source_slice" | "concat_fallback"
}
```

`summary.json` 同时记录 `global_clip_count / source_slice_count / concat_fallback_count / scene_group_count`。

---

### 3.5 Global Caption（MiMo 1-2 分钟）

#### 3.5.1 `mimo_global_captioner.py`（V3 默认）

**输入**：`global_clips.jsonl`
**输出**：

```text
<output>/<video>/global_captions/global_NNNNN.json
<output>/<video>/raw_responses/global_NNNNN.txt
<output>/<video>/global_captions.jsonl
<output>/<video>/summary.json
```

**请求内容**（`annotate_one`）：

```text
1. text: Global 主 Prompt（默认 prompts/global_caption_prompt_en_semantic_roles_v3.md，可经 --prompt-file 覆盖）
2. video_url: 当前 global clip 的 base64 data URL
3. text: "\nThis is global clip {id}. Its source-video time range is {range}, and it contains shot IDs {ids}.\n"
```

**Schema 校验（`validate_global_payload`）**：

| 顶层字段 | 要求 |
|---|---|
| `subject_registry` | list，每条必须有 `subject_id = "S[1-9][0-9]*"`，`subject_type=person` |
| `subject_event_timeline` | list，每条 `subject_ids`/`speaker_id` 必须是 `unknown` 或已注册的 S-ID |
| `global_event_timeline` | list，必须含 `time_range_sec / event_phase / subject_ids / event / state_before / state_after / speaker_ids`；`event_phase ∈ {introduction, continuation, escalation, turning_point, resolution, transition}` |

**关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--concurrency` | `2` | async semaphore |
| `--timeout` | `1800` | 单次请求超时 |
| `--retries` | `3` | 失败重试轮数，每轮 temperature 递增 0.2（封顶 1.0），间隔指数退避（60 → 120 → 240s，封顶 300s） |
| `--max-completion-tokens` | `8192` | V3 schema 长，源码注释要求 76s global 不可 < 16k；`mimo_video_pipeline` 默认传 `32768` |
| `--temperature` | `0.2` | |
| `--request-format` | `video_url` | `auto` 时按 `video_url → input_video` 顺序尝试 |
| `--thinking` | `False` | 透传到 `chat_template_kwargs.enable_thinking` |
| `--skip-existing` | `False` | 已成功的 `global_NNNNN.json` 跳过 |
| `--flat` | `False` | 单视频扁平输出 |

**Skip-existing 语义**：`_captions_all_success(manifest)` 同时校验每条 JSON `success=True`；破损的 JSONL 不会被错误锁定，会强制重跑（否则错误 caption 永远不能修复）。

---

#### 3.5.2 `mimo_global_captioner_speaker_v2.py`（Scheme A global）

**差异**：在同一次 MiMo 调用中要求模型输出 `asr_speaker_attributions`（每条 ASR 一行归属到 S-ID 或 `unknown`），本地用 `align_authoritative_dialogue` 做 **不可变 join**：ASR 的文本/时间戳永远以 ASR 为准，本地再计算 `audio_asset_candidate` 准入门（归属 resolved + 高置信 + 无 overlap + 非画外 + 直接证据 + 时长/语速阈值）。

**Schema version**：`global-caption-speaker-attribution-v2`

**产物额外**：`speaker_attributed_asr.jsonl`（聚合 `asr_speaker_attributions`），`speaker_attribution_summary` 含 `asr_count / resolved_count / accepted_for_audio_asset_count`。

**强制要求**：

- 每次请求必须能解析出至少一条与 global 时间重叠的 ASR 行；否则直接抛错（`raise ValueError("No ASR rows overlap global {id}")`）。
- `authoritative_asr_timeline` 与历史一致才复用缓存，否则失效重跑。

**准入门**：

- `MAX_AUDIO_ASSET_ASR_SPAN_SECONDS = 10.0`
- `MIN_AUDIO_ASSET_SPEECH_UNITS_PER_SECOND = 0.75`（粗略数 Latin tokens + CJK 字符）
- `direct_identity_evidence ∈ {visible_lip_sync, exclusive_audio_visual_cooccurrence, repeated_voice_face_cooccurrence}`

---

### 3.6 Shot Caption（MiMo 单 shot）

#### 3.6.1 `mimo_shot_captioner.py`（V1 调度）

**输入**：`shots.jsonl` + `global_captions.jsonl`（**必需**，用于复用 subject_registry / global_event_timeline 上下文）
**输出**：

```text
<output>/<video>/shot_captions/shot_NNNNN.json
<output>/<video>/raw_responses/shot_NNNNN.txt
<output>/<video>/shot_captions.jsonl
<output>/<video>/summary.json
```

**`load_global_subject_context`**：把 global 的 `subject_registry / subject_event_timeline / global_event_timeline / global_description` 按 `shot_id` 索引；空的 global 上下文会写 WARNING。

**请求内容**：

```text
1. text: Shot 主 Prompt
2. video_url: shot clip base64
3. text: "This is shot {id}. Its absolute source-video time range is {range}. Annotate only this clip.\n"
   后面接 GLOBAL PERSON CONTEXT（json.dumps 缩进）
```

**Validator `validate_shot_subject_ids`**：防止模型偷偷生成 S-ID 替换全局注册表——出现未注册的 `S\d+` 直接抛错。允许 `unknown / new_subject_candidate` 以及白名单内的 ID。

**`normalize_shot_description`**：把 `<d>[lang]text</d>` 标签归一（修复嵌套/重复）；把人物 ID 显式追加到描述；移除嵌套的 `integrated_multimodal_description` 字段。

**关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--concurrency` | `4` | |
| `--timeout` | `1200` | |
| `--retries` | `3` | 同样指数退避 |
| `--max-completion-tokens` | `2048` | `mimo_video_pipeline` 默认传 `8192` |
| `--temperature` | `0.2` | |

---

#### 3.6.2 `mimo_shot_captioner_v2.py`（key_object_candidates 合约）

**差异**：默认 prompt 换成 `prompts/shot_caption_prompt_en_semantic_roles_objects_v4.md`，新增 `normalize_key_object_candidates` 校验：

| 字段 | 要求 |
|---|---|
| `priority` | ∈ {must, high, optional, ignore} |
| `reason_codes` | ⊂ {character_interaction, narrative_prop, recurring, state_change, closeup, distinctive, text_information} |
| `confidence` | ∈ {high, medium, low, unknown} |
| `related_subject_ids` | ⊂ registered_ids ∪ {unknown / new_subject_candidate} |
| `visible_time_range_sec` | 相对当前 shot 的时间窗；可缺失但下游 grounding 会丢失时窗约束 |
| `local_object_id` | shot 内唯一，必须为 `OC\d+` 形式 |
| `proposal_status` | V2 强制为 `unverified`（本模块不创建全局 OBJ 身份） |

**只**不**改 network/orchestration；**与 V1 共享请求/重试/持久化逻辑。

---

#### 3.6.3 `mimo_shot_captioner_v3.py`（V3 entry）

**差异**：复用 V2 全部，仅把默认 prompt 换成 `prompts/shot_caption_prompt_en_semantic_roles_objects_v5.md`，schema_version 升级为 `shot-caption-atomic-key-objects-v3`。

**调用**：

- 主链路（`mimo_video_pipeline.stage_shot_caption`）固定调用 `mimo_shot_captioner.py`（V1），通过 `--prompt-file` 传 V3/V4 prompt 实现切换。
- `mimo_video_pipeline_scheme_a.py` 固定 `--prompt-file prompts/shot_caption_prompt_en_semantic_roles_objects_v5.md`（即 V3 内容）。

---

### 3.7 Segment Caption（MiMo 15-30s）

> **段边界来源的三种可能**（本仓库同时支持，由你决定走哪一条）：
>
> 1. **MiMo 自己切**（V2 当前默认）—— `mimo_segment_captioner_v2.py` 给 MiMo 发 `valid_boundary_guide`（枚举所有 15-30s 连续 shot 区间 + 完整划分的 top-40 候选），强制 MiMo 精确拷贝一份，不准自由发挥。
> 2. **上游固定切 + MiMo 填内容**（Scheme A）—— `mimo_segment_captioner_fixed_segments_v4.py` 接 `--segment-input <任一窗口切 segments.jsonl>`（来自 `m07_segment/run.py` 或 `segmentation/three_way.py`），MiMo 不重新规划段边界，只填 caption / dialogue / object alignment。
> 3. **窗口切只产边界、不产 caption**—— `m07_segment/run.py`（带 QualityCard 加权）或 `segmentation/three_way.py`（三路对比 window/v1/v2），纯算法，不需要 MiMo。
>
> 详细对比（何时用哪条、怎么串起来、产物 schema 差异、运行成本）见 [`PIPELINE_COMPARISON.md`](./PIPELINE_COMPARISON.md)。本节只讲"MiMo 切"对应的 caption stage。

#### 3.7.1 `mimo_segment_captioner.py`（V1 基底，被 V2/V3/V4 monkey-patch）

**输入**：`global_captions.jsonl` + `shot_captions.jsonl`
**输出**：

```text
<output>/<video>/segments/global_NNNN.json
<output>/<video>/raw_responses/global_NNNN.txt
<output>/<video>/segment_captions.jsonl
<output>/<video>/summary.json
```

**`prepare_shots`**：每个 shot_id 必需在 shot_captions.jsonl 命中；`time_range_sec` 校验；归一为 `[local, source]` 双轨：

```python
local["time_range_sec"]   = [max(0, src_start - global_start), max(0, src_end - global_start)]
local["source_time_range_sec"] = [src_start, src_end]
```

**ASR 旁路 `prepare_authoritative_asr`**：自动判断 timebase（`auto` / `source` / `global-local`）。支持 JSON 与 JSONL 输入；优先匹配 `asr_timeline.json` / `asr_segments.jsonl`；以 `global_id` 数字后缀或 `global_<id>` 文件名作 scope。

**请求内容（`build_input_text`）**：

```text
1. text: Segment 主 Prompt
2. video_url: global clip base64
3. text: INPUT DATA 段
   [Global Caption and Event Evidence]   ← global_payload JSON
   [Authoritative ASR Timeline]          ← list[asr_row]
   [Shot N | local_time: a-b sec]        ← 每条 shot_caption
```

**`validate_and_enrich`**：

- 校验 segments 覆盖整段 global clip（无 overlap、无 gap）；
- 调用 `flatten_speaker_attributions` 把每个 segment 的 speaker 归属平铺为下游音频切割器的直接输入契约；
- 当权威 ASR 存在时，调 `align_authoritative_dialogue` 不可变 join。

**`align_authoritative_dialogue`** 的核心规则：

```text
- 文本/时间戳永远以 ASR 为准
- 仅接受 MiMo 的"speaker 归属判断"
- 每条 ASR 只能出现一次（MiMo 不准重复/省略/自创）
- 本地计算 audio_asset_candidate 与 rejection_reasons
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--concurrency` | `1` | segment 上下文最重，单并发默认 |
| `--timeout` | `1800` | |
| `--retries` | `3` | `--retries=6` 由 `mimo_video_pipeline.stage_segment_caption` 默认传入 |
| `--max-completion-tokens` | `16384` | V2 默认 |
| `--temperature` | `0.1` | |

---

#### 3.7.2 `mimo_segment_captioner_v2.py`（当前默认）

**差异**：

| 维度 | V1 | V2 |
|---|---|---|
| Segment 时长 | 无硬合约 | **15 ≤ dur ≤ 30**（`MIN_SEGMENT_SECONDS / MAX_SEGMENT_SECONDS`，`TIME_EPSILON=0.25`） |
| 边界规划 | MiMo 自由切分 | 给模型发 `valid_boundary_guide`（枚举所有 15-30s 连续 shot 段 + 完整划分的 top-40 候选） |
| 物体候选 | 不显式管理 | `object_candidate_inventory` + `normalize_segment_objects`（V2 物体合约） |
| 顶层附加字段 | — | `object_candidate_index` / `object_quality_issues` / `duration_contract` / `schema_version=segment-caption-key-objects-v2` |
| Prompt | V1 prompt | `prompts/segment_caption_prompt_en_semantic_roles_objects_v4.md` |

**`valid_boundary_guide`**：枚举所有满足 `MIN-ε ≤ duration ≤ MAX+ε` 的连续 shot 区间，并产出 `valid_complete_partitions`（完整无重叠划分的前 40 名候选，按 `target_distance_sum_sec` 排序）。`mandatory_instruction` 强制模型从中**精确复制**一个划分，不允许自由发挥。

**`normalize_segment_objects`**（V2 物体合约）：

| 校验 | 规则 |
|---|---|
| `source_candidate_refs` | 必须命中 shot 注入的 inventory；不接受未注册的 ref |
| 重复/已拒绝 | `source_candidate_refs` 不能重复或同时在 `rejected_object_candidate_refs` |
| `canonical_name` | 非空 |
| `priority` | ∈ {must, high, optional, ignore} |
| `related_subject_ids` | ⊂ registered_ids ∪ {unknown / new_subject_candidate} |
| `segment_role` | ∈ {narrative_prop, interaction_prop, recurring_prop, emphasized_prop, information_prop, uncertain} |
| `confidence` | `[0.0, 1.0]` 截断并 round 6 位 |
| `segment_object_id` | 由代码统一赋值 `SEG_{segment_id:03d}_OC{NNN}` |

未在模型输出中使用但存在于 inventory 的 candidate 自动降级为 `_fallback_candidate`（`proposal_status=shot_only_unconsolidated`），写到 `object_quality_issues`。

**实现技巧**：`V2` 用 `import mimo_segment_captioner as v1` 复用 V1 的请求/重试/持久化/JSONL 写入；只 monkey-patch `v1.build_input_text / v1.validate_and_enrich`，再用 `v1.main_async(args)` 触发。

---

#### 3.7.3 `mimo_segment_captioner_global_speaker_v3.py`（Scheme A 中段）

**与 V2 的关系**：在 V2 上叠加 Scheme A 说话人策略。

**核心**：

- `prepare_authoritative_asr` 被替换为 `global_speaker_attributions`：从 global caption 的 `asr_speaker_attributions` 拿只读列表（不是 ASR 文本），按时间排好。
- `build_input_text` 在 V2 输入基础上追加 `[READ-ONLY GLOBAL SPEAKER-ATTRIBUTED ASR]`，并要求模型返回 `dialogue: []`（本地填）。
- `validate_and_enrich` 把模型的 dialogue 直接丢弃（计数 `ignored_model_dialogue_count`），再把 global 的 speaker attribution 按 **最大 overlap + 最小距离** 写到对应 segment 的 `dialogue` 里（不可变 ASR）。
- `key_object_candidates` 中 `source_candidate_refs` 不在 inventory 内的整段丢弃（`ignored_unsourced_object_count`）。
- schema_version：`segment-caption-global-speakers-v3`

**注意**：`--asr-input` 在 Scheme A 中**禁用**（parser 强制报错）；说话人永远来自 `--global-input` 的 global caption。

---

#### 3.7.4 `mimo_segment_captioner_fixed_segments_v4.py`（Scheme A 末段）

**适用场景**：上游（level3 m07 / `m07_segment` / 三路聚合）已经产出固定 `segments.jsonl`（按时间窗/视觉信号切好），MiMo 不允许重新规划段边界，只填内容。

**核心**：

- `UPSTREAM_SEGMENTS = load_fixed_segments(--segment-input)`：读上游 `segments.jsonl`；必须是单文件；
- `fixed_plan_for_global`：每个上游段投影到当前 global clip 范围（按 `source_time_range_sec` 重叠判定），并按 `shot_ids` 与当前 global 的 `shot_ids` 求交集；同时校验：
  - **完整覆盖**：`actual == expected`，每个 expected shot 恰好出现一次；
  - **ID 唯一**：上游段 ID 不重复。
- prompt 切换为 `prompts/segment_caption_prompt_en_fixed_segments_global_speakers_v6.md`，要求模型返回 `segment_captions: [{upstream_segment_id, ...}, ...]`，**严格按上游 ID 一对一**，不容许新增/省略/重排。
- 校验：模型的 `key_object_candidates` 中 `source_candidate_refs` 必须在 inventory；`dialogue` 直接丢弃，由本地把 global 的 ASR 归属按最大 overlap 写入。
- schema_version：`segment-caption-fixed-upstream-v4`

**实施**：v4 把 v1 的 `build_input_text / validate_and_enrich / annotate_one / prepare_authoritative_asr / load_asr_sidecars` 全部 monkey-patch，再调 `v1.main_async`。`args.asr_input = args.segment_input` + `args.asr_timebase = "fixed-upstream"` 让 V1 的缓存失效判定与上游变更对齐。

---

### 3.8 切 segment 为 mp4 `mimo_segment_cutter.py`

**输入**：`segment_captions.jsonl` + `global_clips.jsonl`（可选 clip-path 兜底）
**输出**：

```text
<output>/<video>/clips/<video>_global_NNNN_segment_NNN.mp4
<output>/<video>/segments.jsonl
<output>/<video>/summary.json
```

**关键逻辑**：

1. 解析 `global_clips.jsonl` 的 `clip_path` map；
2. 每条 segment record 重新计算 `time_range_sec`（按 global clip 内的 local 时间，且 `end ≤ duration + 0.25s`）；
3. `cut_clip(input_path, output_path, start, end, args)`：默认重编码 `libx264 crf=18 preset=fast`，AAC 192k；`--copy` 走流复制（keyframe 不精确）；
4. 输出 record 顺序 `(global_id, segment_id)` 排序；
5. `success_count == segment_count` 才退出码 0，否则 1。

**关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ffmpeg` | `ffmpeg` | |
| `--copy` | `False` | 流复制开关 |
| `--crf` | `18` | |
| `--audio-bitrate` | `192k` | |
| `--skip-existing` | `True` | 已存在的 mp4 跳过 |

---

## 4. 三级 Caption：共同机制

### 4.1 MiMo 客户端 `mimo_caption_client.py`

**职责**：所有 caption stage 的共享原语，**不**做网络 IO / 校验 / 重试。

| 函数 | 作用 |
|---|---|
| `load_dotenv(path)` | 读 `env/.env`（默认 `SOURCE_VIDEO_PROCESS_ROOT = /gemini/platform/public/aigc/lss/Interns/lsj/video_process`） |
| `configure_no_proxy_for_url(url)` | 防止工作站 HTTP proxy 拦截 in-cluster MiMo 服务 |
| `resolve_base_url(override)` | `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `API_BASE` / 内置 DNS `http://tcp30000-1058.default.svc.cluster.local:30000/v1` |
| `resolve_model(override)` | `MODEL_NAME` / `MiMo-V2.5` |
| `resolve_api_key(override)` | `OPENAI_API_KEY` / `EMPTY` |
| `default_prompt_path(stage, v2, version)` | V3 默认；`v2=True` / `version=v1|v2|v3`；缺失则 V3 → V2 → V1 兜底 |
| `find_*_manifest(input_path)` | 适配文件或目录输入 |
| `resolve_clip_path(raw_path, manifest_path)` | 相对路径向上找 |
| `load_jsonl / write_jsonl` | 校验行 JSON |
| `persist_item(individual_dir, raw_dir, prefix, index, payload, response_text)` | `<prefix>_NNNNN.json` + 可选 `.txt` |
| `build_video_content(prompt, encoded, request_format)` | `video_url` 或 `input_video` 格式 |
| `extract_text / extract_response_text` | `content` 退化到 `reasoning_content` |
| `parse_json_response(text)` | 容忍 think/code fence；手写花括号配对 |
| `encode_video_b64(video_path)` | base64 |
| `unwrap_nested_caption(record, key)` | 处理 `record[key][key]` 嵌套 |

**Prompt 路径常量**（按版本）：

```text
SOURCE_GLOBAL_PROMPT       = lsj/video_process/prompts/global_caption_prompt_en.md
SOURCE_SHOT_PROMPT         = lsj/video_process/prompts/shot_caption_prompt_en.md
SOURCE_SEGMENT_PROMPT      = lsj/video_process/prompts/segment_caption_prompt_en.md
SOURCE_PROMPTS_V2_DIR      = ../prompts/ (extract_full_pipeline 上一级)
SOURCE_PROMPTS_V3_DIR      = ../prompts/ (extract_full_pipeline 上一级)
```

**Prompt 切换矩阵（**主链路**实际使用的 prompt）**：

| Stage | 默认 prompt（`mimo_caption_client.default_prompt_path`） | `mimo_video_pipeline` 实际传入 | `mimo_video_pipeline_scheme_a` 实际传入 |
|---|---|---|---|
| global | `…_semantic_roles_v3.md`（V3） | `--prompt-file` 不传 → 默认 V3 | `--prompt-file prompts/global_caption_prompt_en_semantic_roles_speakers_v4.md`（speaker_v2 自带） |
| shot | `…_semantic_roles_v3.md`（V3） | `--prompt-file` 不传 → 默认 V3 | `--prompt-file prompts/shot_caption_prompt_en_semantic_roles_objects_v5.md`（V3 内容） |
| segment | `…_semantic_roles_v3.md`（V3） | 调用 `mimo_segment_captioner_v2.py`（自带 `…_objects_v4.md`） | 调用 `mimo_segment_captioner_fixed_segments_v4.py`（自带 `…_fixed_segments_global_speakers_v6.md`） |

> 也就是说 `mimo_video_pipeline` 的 segment stage 走 **V2**（自带 V4 prompt）；`scheme_a` 走 **V4**（自带 V6 prompt）。两者 schema_version 不同。

---

### 4.2 Prompt 资产目录 `prompts/`

| 文件 | 阶段 | 备注 |
|---|---|---|
| `global_caption_prompt_en_semantic_roles_speakers_v4.md` | global + speaker | Scheme A |
| `segment_caption_prompt_en_objects_global_speakers_v5.md` | segment + object + global speakers | Scheme A 中段 |
| `segment_caption_prompt_en_fixed_segments_global_speakers_v6.md` | segment + fixed + global speakers | Scheme A 末段 |
| `segment_caption_prompt_en_semantic_roles_objects_v4.md` | segment + object | V2 默认 |
| `shot_caption_prompt_en_semantic_roles_objects_v4.md` | shot + object | V2 默认 |
| `shot_caption_prompt_en_semantic_roles_objects_v5.md` | shot + object | V3 默认 / scheme_a |
| `shot_caption_prompt_en_semantic_roles_v3.md` | shot semantic roles | legacy / 兜底 |
| `tracklet_semantic_alignment_prompt_en_v1.md` | Stage1.5 V1 | （未在主链路直接使用） |
| `tracklet_semantic_object_alignment_prompt_en_v2.md` | Stage1.5 V2 | 视觉↔caption 物体对齐 |

`prompts/` 软链在 `assets/visual_person/prompts → ../../prompts`，方便 Stage1.5 直接引用。

---

### 4.3 重试 / 持久化（caption stage 通用）

```text
rounds = max(1, args.retries)
for round_no in 1..rounds:
    args.temperature = min(1.0, base + 0.2 * (round_no - 1))
    pending = records[success=False]
    tasks = [annotate_one(...) for record in pending]
    results = asyncio.as_completed(tasks)
    for result in results:
        all_results[id] = result
        persist_item(<dir>, raw_dir, prefix, id, result, raw_text)
    write_jsonl(jsonl, ordered)              ← 每轮覆盖
    summary.json                             ← 每轮覆盖
    if failed == 0 or not pending: break
    backoff = min(300, 60 * 2^(round-1))
    await asyncio.sleep(backoff)
```

`--skip-existing` 仅跳过**完全成功**的旧 JSON；任何破损的 JSON 都不会被错误复用。

---

## 5. 编排器

### 5.1 `mimo_video_pipeline.py`（主链路）

**Stage 选择**（`--stage`）：

| Stage | 调用脚本 | 输入 | 输出 |
|---|---|---|---|
| `split-shots` | `shot_splitter.py` | `--input` (mp4) | `<work_dir>/shots/` |
| `aggregate-global` | `global_clip_aggregator.py` | `<work_dir>/shots` | `<work_dir>/global/` |
| `global-caption` | `mimo_global_captioner.py` | `<work_dir>/global` | `<work_dir>/captions/global/` |
| `shot-caption` | `mimo_shot_captioner.py` | shots + captions/global | `<work_dir>/captions/shot/` |
| `segment-caption` | `mimo_segment_captioner_v2.py` | captions/global + captions/shot | `<work_dir>/captions/segment/` |
| `cut-segments` | `mimo_segment_cutter.py` | captions/segment | `<work_dir>/segments/` |
| `all` | 依次 6 步 | — | — |

**关键 CLI**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--stage` | `all` | |
| `--input` | — | 源视频路径；同时作为 video stem |
| `--work-dir` | `./pipeline_outputs/<stem>` | |
| `--force` | False | 强制重跑（即使有产物） |
| `--detector` | `auto` | TransNetV2 / adaptive / content |
| `--transnet-device` | `auto` | |
| `--scene-detector` | `scrl` | |
| `--min-global-sec` | `60.0` | |
| `--target-global-sec` | `90.0` | |
| `--max-global-sec` | `120.0` | |
| `--global-workers` | `2` | |
| `--shot-workers` | `4` | |
| `--segment-workers` | `1` | |
| `--asr-input` | — | segment stage 注入 ASR 旁路 |
| `--asr-timebase` | `auto` | |
| `--retries` | `3` | global/shot stage |
| `--segment-retries` | `6` | segment stage |
| `--max-completion-tokens` | `16384` | segment stage |
| `--request-format` | `video_url` | |
| `--request-timeout` | `1800.0` | |
| `--thinking` / `--no-thinking` | `False` | 透传到 MiMo |

**Skip-existing 规则**：仅 caption stage 用 `_captions_all_success(manifest)`；其他 stage 仅看文件是否已存在。Stage 之间**不**做严格依赖校验，建议按顺序跑。

**中断处理**：任一 stage 非 0 退出 → 停止后续 stage（避免依赖缺失导致 cascade 失败）。

---

### 5.2 `mimo_video_pipeline_scheme_a.py`（Scheme A 编排）

**与主链路差异**：

- `global-caption` 调 `mimo_global_captioner_speaker_v2.py`，**必须** `--asr-input`。
- `shot-caption` 固定 `--prompt-file prompts/shot_caption_prompt_en_semantic_roles_objects_v5.md`（V3 内容）。
- `segment-caption` 调 `mimo_segment_captioner_fixed_segments_v4.py`，**必须** `--segment-input`（上游 `segments.jsonl`）。
- `all` **不包含** `split-shots` / `aggregate-global` / `cut-segments`（这些由上游 stage 自行处理）。

**适用场景**：上游已经产出固定 shot/segment 边界（level3 m07 / 三路聚合），Scheme A 只负责 caption + 说话人归属。

---

### 5.3 `assets_runner.py`（legacy segment 级 stub）

**不是生产推荐**。保留为 segment-level 视觉/物体/音频抽资产的演示入口，提供 stub / real 两套模式：

- `--mode stub`：抽一帧 + 写 placeholder manifest；
- `--mode real`：调 `entity_asset_extractor.extract_frame_observations` / `speaker_voice_asset_pipeline.run` / `build_object_grounding_queue_v2 + object_grounding_segmenter_v3`（需要 GPU + 已下载本地权重）。

文档注释直接说明：**当前推荐路径是 ARCHITECTURE.md 4.13/4.14 的 Scheme A 链**（人物视觉：`shot_tracklet_builder → semantic_role_tracklet_refiner_v3 → visual_character_registry_builder → character_asset_bank`；人物音频：`mimo_global_captioner_speaker_v2 → export_all_asr_audio_review`；ID 统一：`character_role_aligner`）。

---

## 6. 主链路：默认阈值与边界

| 项 | 默认 | 出处 |
|---|---|---|
| 切镜 detector | auto（TransNetV2 → PySceneDetect adaptive 兜底） | `shot_splitter.detect_raw_scenes` |
| 切镜 threshold | `3.0`（PySceneDetect） | `shot_splitter.parse_args` |
| 切镜 min/max shot sec | `2.0` / `None` | 同上 |
| 切镜 merge-short | `True` | 同上 |
| 切镜 ffmpeg | `-c:v libx264 -crf 18 -preset veryfast` | `shot_splitter.export_clip` |
| 切镜 vendored 源 | `m02_shots/teleai_src/`（不 import teleAI-T2V） | DEFAULT_TRANSNET_SOURCE_DIR |
| 聚合 min/target/max global sec | `60 / 90 / 120` | `mimo_video_pipeline.parse_args` |
| 聚合 scene detector | `scrl`（auto 模式权重 + cuda 可用时启用） | `global_clip_aggregator.get_scene_groups` |
| 聚合 export_mode | 优先源视频切，失败 shot concat | `global_clip_aggregator.export_global_clip` |
| global caption concurrency | `2` | `mimo_video_pipeline` 默认 |
| global caption retries | `3` | 同上 |
| global caption max-completion-tokens | `32768`（`mimo_video_pipeline.stage_global_caption` 硬写） | 同上 |
| global caption timeout | `1800` | `mimo_caption_client` 默认 |
| global caption temperature | `0.2` | `mimo_global_captioner` 默认 |
| shot caption concurrency | `4` | `mimo_video_pipeline` 默认 |
| shot caption retries | `3` | 同上 |
| shot caption max-completion-tokens | `8192`（`mimo_video_pipeline.stage_shot_caption` 硬写） | 同上 |
| shot caption temperature | `0.2` | `mimo_shot_captioner` 默认 |
| segment caption concurrency | `1` | `mimo_video_pipeline` 默认 |
| segment caption retries | `6` | 同上 |
| segment caption max-completion-tokens | `16384` | 同上 |
| segment caption min/target/max sec | `15 / 20 / 30`（V2 硬约束） | `mimo_segment_captioner_v2` 模块常量 |
| segment caption epsilon | `0.25` | 同上 |

---

## 7. 镜头/段邻接表规则（传统 `Workspace` 路径）

### 7.1 `common/clip_store.py`

- `rebuild_links(clips)`：按 `sort_key(clip) = (start_frame, _natural_key(clip_id))` 排序后统一重挂 `prev / next`；`continuous` 仅当 `same_parent(a, b)` 为 `True`。
- `same_parent(a, b)`：`a.parent_clip_id == b.parent_clip_id`，且至少一方的 `parent_clip_id` 非空（即子 clip 关系）。
- `make_sub_clip(parent, index, start_frame, end_frame)`：派生 `clip_id = f"{parent.clip_id}-{index}"`，`parent_clip_id` 指向最初父级（保证再切子不丢溯源）。
- `split_clip(clips, clip_id, cut_frames)`：把父 clip 替换为按 cut_frames 切出的子 clip 序列，邻接指针整链重挂。

### 7.2 `m04_recut/cutter.py::find_cut_points`

切刀点优先级：

```text
1. ASR 静音间隙 > min_gap_ms（默认 300ms）→ 取间隙中点
2. 词边界（相邻 word 时间戳间隙）
3. 当前累计时长 ≥ max_sec → 硬切
4. 贪心：在 [cur_start, cur_start+max_sec] 窗内选最靠目标点的间隙中点/词边界，否则硬切
```

### 7.3 `m04_recut/run.py::recut_workspace`

- 仅对 `c.duration > max_s` 且未被质检剔除的 clip 处理；
- 子 clip 继承父 ASR（按区间裁剪 + 时间戳平移，丢弃 0.05s 擦边碎片）；
- 子 clip 继承父质检卡，**仅时长项重判**：`duration < min_seconds` → `hard_reject=True`；`extra["need_qc_recheck"]=True`；
- 物理切割走进程池（`ProcessPoolExecutor`），切点两侧留 `overlap_frames`（默认 4 帧）。

### 7.4 `m07_segment/window.py::plan_segments`

收刀优先级：

```text
1. cur_dur ≥ max_seconds → 硬收
2. cur_dur ≥ target_seconds 且"现在收"比"加下个 clip"更近 target → 收（贴近目标，宁短不长）
3. should_break(seg, next) 边界信号（视觉不一致 / 不可注册），
   仅 cur_dur ≥ min_seconds 后生效
```

完整无重叠划分；单 clip 段允许（场景边界夹着的独立长镜头不强行合并）。

### 7.5 `m07_segment/consistency.py`

- `DINOv3Embedder.embed(video_path)`：DINOv3 视觉特征（如可用）。
- `gray_hist_embedding(video_path)`：灰度直方图兜底。
- `cosine(a, b)` / `window_consistency(embs)`：段内 clip 两两余弦相似度均值。

`_BoundaryJudge` 规则：

- 同 parent 再切割子 clip 永连续（不收刀）；
- 仅 DINOv3 可用时启用硬阈值（直方图代理阈值不可靠，不硬卡）；
- 抽帧失败的零向量跳过门（不误切）。

---

## 8. 资产侧：视觉人物（推荐 4 阶段链）

### 8.1 Stage1：单帧观测 `assets/visual_person/entity_asset_extractor.py`

> 严格说 Stage1 是"单帧观测库"，已被 Stage1/1.5/3 替换；此处保留它的"作为 Stage1 底层观测库"的角色。

**单帧 Pipeline**：

```text
frame
  -> Sa2VA "all the main characters" mask
  -> 连通组件 -> OBS00/OBS01/...（frame-local）
  -> GroundingDINO("person") 拿所有高置信框（不止 top-1）
  -> 每个 OBS 绑最佳匹配框（mask coverage + IoU）
  -> SAM 3.1 用 box 作几何 prompt → instance mask
  -> InsightFace 全图人脸 → 每个 OBS 至多绑 1 张脸（中心在 bbox+mask+上半身）
  -> 资产 = body_crop / mask / masked_crop / rgba / overlay + face crop + debug overlay
```

**强 human 证据**：`has face` 或 `bbox 高宽比 ≥ 0.80`（避免横长 hand/towel 误识 person）。

**修正 nested mask**：enclosing merged OBS - 内部 OBS。

**重要**：`identity_id = None` 始终——本模块不解决跨帧身份。

**模型权重**（保留在 v2 仓库根 `local_models/`）：

| 用途 | 路径 |
|---|---|
| Sa2VA discovery | `local_models/subject_grounding/sa2va/Sa2VA-Qwen3-VL-4B` |
| GroundingDINO | `local_models/AI-ModelScope/GroundingDINO/groundingdino_swint_ogc.pth` + `bert-base-uncased` |
| InsightFace | `local_models/insightface/buffalo_l` |
| SAM 3.1 | `local_models/segmentation/sam3_1` |

**调用**：

```bash
python entity_asset_extractor.py --video demo.mp4 --output workspace/obs_demo --num-samples 10
```

---

### 8.2 Stage1（推荐）：shot 局部 tracklet `assets/visual_person/shot_tracklet_builder.py`

**职责**：把 frame-local Observations 扩展为 shot-local Tracklet。

**算法**：SAM 3.1 周期锚 + 稀疏光流双向传播 + 周期重检测 + 匈牙利关联 + ACTIVE/LOST/FINISHED 状态机；**不**跨 shot 分配 ID。

**输入**：每个 shot 的 mp4 + `entity_asset_extractor`（作为底层观测库）
**输出**：

```text
tracklet_signatures.json     face/body gallery 与每个 tracklet 的特征
tracklet_overlay.mp4         带 tracklet ID 框的可视化视频（MiMo 对齐的关键输入）
```

**schema_version**：`shot-tracklets-v2`（v2 新增 `detection_plan`/`detection_frames`/`filtered_tracklets`/`cannot_link_evidence`；短镜头联合证据准入见 Stage1 源码 docstring）

**V4 局部多尺度重检与证据化去重**（默认开启，`--disable-local-redetect` / `--disable-dedup-instance-evidence` 回滚）：

- 仅对困难帧（低检测覆盖 / 框冲突 / 存在未绑定人脸）在原图坐标系内做局部放大重检：ROI 裁剪放大（长边封顶 1600px）→ 现有 GroundingDINO/InsightFace → 坐标回投，每帧 ROI 数封顶 `--max-local-rois`（默认 3）；新框与全帧框 IoU≥0.50 去重，来源写入观测 `notes`，不新增 schema 字段。
- `deduplicate_observations` 改为证据化：重叠对先查独立实例证据——双方绑定**不同**人脸（一脸至多属一个 OBS）或 mask IoU≤0.30 → 两条都保留；同一脸框（IoU≥0.50）或 mask IoU≥0.60 → 合并；无证据回退旧框规则合并。合并/保留决策写入幸存记录 `notes`。

---

### 8.3 Stage1.5：语义对齐 `assets/visual_person/semantic_role_tracklet_refiner.py`（_v2 / _v3）

**职责**：把 overlay 视频 + 三级 caption 证据喂 MiMo，逐 tracklet 对齐 caption 注册的 S-ID，输出：

| decision | 含义 |
|---|---|
| `keep_primary` | 该 tracklet 应对齐 S-ID（权重 1.0） |
| `keep_supporting` | 辅助证据（权重 0.85） |
| `keep_context` | 上下文角色（权重 0.5） |
| `reject_background` | 非主体 |
| `reject_non_person` | 非人 |
| `review` | 不确定（人工） |

**白名单校验**：未注册的 S-ID 直接报错重试；确定性 fail-open 安全网（低置信拒绝自动降级 review）。

**产物**：

- `semantic_identity_decisions.json`（跨 shot，conf≥ 0.90 提示边）
- `stage3_inputs/{primary,narrative,safe}/`（三档视图）
- `tracklet_object_proposals`（来自 shot caption 的 `key_object_candidates` → 投影到 tracklet）

**版本演进**：

| 版本 | 关键差异 |
|---|---|
| V1 | 仅人物对齐 |
| V2 | 关键物体提案（来自 shot caption 的 `key_object_candidates`） |
| V3 | 物体可见区间修正（保留 `visible_time_range_sec` 不丢） |

**V6 语义反馈补检**（默认关闭，三段解耦，最多一轮）：

1. refiner `--enable-semantic-feedback`：首轮对齐后，仅对"未匹配角色且该角色在本 shot 有 shot-local 可见时间窗证据"的 shot 写 `feedback_requests.json`（`semantic-feedback-requests-v1`；`unknown` 等白名单外 ID 不触发，无证据角色只报告）；
2. builder `--semantic-feedback-requests`：只处理请求的 shot，把证据窗换算成额外检测锚点（每窗 start/mid/end、封顶 `--max-feedback-anchors` 6 个）注入检测计划，输出到**独立** feedback 目录（覆盖保护：目标目录已有非反馈产物时拒绝运行），manifest 记 `semantic_feedback{round, added_anchor_frames, ...}`；检测/准入/去重逻辑不变——caption 声称有人不直接创建身份；
3. refiner `--feedback-tracklets DIR`：第二轮对这些 shot 改读 feedback 目录的 overlay/tracklets 重对齐一次，结果与 Stage3 视图记 `semantic_feedback_round: 1`；反馈消费轮不再产新请求（一轮封顶），仍 unmatched 的角色如实记入 summary。反馈轮会使受影响 shot 的 Stage3/Stage4/L1 产物失效，需按执行顺序重算（使用方责任）。

---

### 8.4 Stage3：跨 shot 聚类 `assets/visual_person/visual_character_registry_builder.py`

**职责**：把 shot-local TrackletSignatures 合并为全局身份。

**算法**：InsightFace `w600k_r50` 人脸 embedding + DINOv3 身体 embedding，**cannot-link 绝对优先**的保守 complete-link 聚类 + 高置信语义边（来自 Stage1.5 的 `MUST_LINK`），晋升 **`CHR001`/`CHR002`** 等视觉身份（**非** S-ID）。

**产物**：

```text
registry.json
  characters: [{character_id: CHR001, members: [tracklet_id, ...], ...}, ...]
  pending_candidates: [...]
  merge_edges: [{decision: MUST_LINK/POSSIBLE_LINK/CANNOT_LINK/NO_EVIDENCE}, ...]
```

**schema_version**：`cross-shot-identity-v2`（v2：cannot-link 只认检测帧共现证据；纯区间相交的对进 `overlap_review_pairs` 待审，不再生成硬边；v1 输入保持旧保守 fallback）

**反脸证据**：`CONTRADICTION_FACE_MEAN = 0.25`——两 tracklet 都带高质量人脸但 `topk_mean < 0.25` 视为**反对**身份。

**图形输出**：原始 tracklet 与合并边保留，便于回滚/再拆。

---

### 8.5 Stage4：画廊 `assets/visual_person/character_asset_bank.py`

**职责**：CHR 画廊分槽（face/body × 视角 × appearance_state）。

**流程**：身份纯度校验 + pHash/embedding 去重 + top-K 竞争。

**产物**：

```text
asset_bank/CHRnnn/face/<view>/*.jpg
asset_bank/CHRnnn/body/<view>/*.png
asset_bank/CHRnnn/role_character.json
```

**schema_version**：`character-asset-bank-v2`（v2：画廊纯度用留一原型，小画廊标 `insufficient_evidence`；去重以 pHash 为主、embedding 仅辅助；身份证据不足的资产标 `identity_unverified` 可追溯）

---

## 9. 资产侧：人物音频（Scheme A 推荐链）

### 9.1 关键依赖 `mimo_global_captioner_speaker_v2.py`

**详见 §3.5.2。** 一次 MiMo 调用同时输出 caption + ASR→S-ID 归属；输出 `speaker_attributed_asr.jsonl`（每条 ASR 行的归属 + `audio_asset_candidate` 准入门）。

---

### 9.2 ASR 音频切 `assets/voice/export_all_asr_audio_review.py`

**职责**：按 ASR 句级精确切点用 ffmpeg 从 global clip 切 16kHz mono wav。

**输出**：

```text
valid|invalid/<global_id>/<S_ID>/<asr_id>.wav       ← 实际音频
all_asr_manifest.jsonl / valid_manifest.jsonl / invalid_manifest.jsonl
                                                    ← 每行附 signal_stats（mean_volume_dbfs / peak_dbfs / silence_ratio）
summary.json                                        ← 计数与驳回原因统计
voice_asset_manifest.jsonl                          ← 可选（--voice-registry / --voice-registry-root），
                                                      schema speaker-voice-asset-v2
```

**重要**：本工具**不**自动把 preview 文件晋升为生产资产库——是**人工复核**通道。`voice_asset_manifest.jsonl` 每行 `review_status` 默认 `pending`；只有 `--approvals` 提供的 JSONL（`global_id`+`asr_id`）能把对应行置为 `approved`。

**语音连接（可选）**：提供 `--voice-registry`（可重复）或 `--voice-registry-root`（递归发现 `**/consolidated/registry.json`）后，每行按 resolve 后的 `global_clip_path` 匹配 `metadata.pipeline_input` 相同的 consolidated registry（`speaker_voice_asset_pipeline` 产物），按时间交集计算主导 SPK、`spk_purity`、`asr_coverage`、与 `simultaneous_speech_regions` 的重叠及净单人语音秒数，判定 `matched / ambiguous / unmatched / insufficient_evidence / timebase_mismatch / no_registry`；matched 行附该 speaker 的净语音 `reference_audio` 路径（供 `character_role_aligner` 物化），并给出 `clean_subintervals`——主导 speaker 净语音的源时间子区间（重叠已减除），交付资产可追溯到这些子区间而非整行。

**schema**：输出字段对齐 `speaker_attributed_asr.jsonl` 行（`asr_id, time_range_sec, source_time_range_sec, speaker_id, audio_asset_candidate, …`）。

---

### 9.3 旧式音频池 `assets/voice/speaker_voice_asset_pipeline.py`

**不依赖任何 LLM/VLM**。四阶段：

```text
1. separate   MossFormer2_SE 前景/背景分离（16kHz）
2. build   Stage5(speaker_registry_builder.run)，--input 指向前景人声 wav
3. gate   speaker_turn_foreground_gate.apply_to_registry：覆盖度 / 重叠 / 污染度过滤 quality_tier
4. overlap   speaker_overlap_detector 跑 pyannote 重叠 → 减除
5. consolidate   池化重嵌入 → union-find + 时间 overlap + cannot-link 合并 → leave-one-out 身份证据
   输出 consolidated/registry.json（speaker-registry-v2 schema，drop-in 兼容 Stage5）
```

**恢复与审计通道**（默认开启，均有开关可回滚）：

- **短 turn 恢复**：仅因短于 Stage5 时长下限被拒的 turn（干净、单人、前景确认）计入池化证据；常规池太小时其音频并入原型嵌入。无常规长度参考时，≥ `--short-reference-min-duration`（默认 0.8s）的可用 turn 可走短参考通道（`recovery="short_turn"`），身份证据不随之放松。
- **部分重叠恢复（A4）**：仅因重叠占比超限被拒的 turn，若仍含 ≥ `--min-recovered-segment-seconds`（默认 1.0s）的连续单人段，最长段可作为参考交付——单段连续、不拼接，`recovery="partial_overlap"` 且记录 `parent_turn_id`；其余干净段只做池化证据。达不到恢复下限的碎片仍整体拒绝。
- **双路声纹验证（A5）**：每个参考候选同时从增强 stem 与原始混合音（`origin_16000`）各嵌入一次，比较两路身份判定——默认影子模式只记录（候选 `dual_path.status: agree/conflict/unavailable`，registry `metadata.consolidation.dual_path_verification` 汇总）；`--dual-path-enforce` 把冲突转为拒绝理由 `dual_path_conflict`。`--disable-dual-path-verification` 完全关闭。

**经验值**：MossFormer2_SE 平均减除 9.1dB 背景；真正增益来自 Stage5 的池化（CAM++ 由 1 个短 clip 扩到 tens-of-seconds）+ 不能链接规则 + leave-one-out 安全网。

**适用**：不接入 Scheme A caption 的旧链；可作为对照路径保留。

---

## 10. 资产侧：关键物体

### 10.1 队列构造 `assets/key_object/build_object_grounding_queue_v2.py`

**输入**：`shot_captions.jsonl`
**输出**：`object_grounding_queue.jsonl`，每条候选保留：

```json
{
  "candidate_id": "SHOT_001_OC1",
  "schema_version": "object-grounding-candidates-v2",
  "visible_time_range_sec": [a, b],   ← shot-relative；下游 grounding 不会采到不在的帧
  "source_caption_schema_version": "shot-caption-atomic-key-objects-v3" | ...,
  "priority": "must" | "high" | "optional" | "ignore",
  ...
}
```

**优先过滤**：`--priorities must high optional ignore`（默认 `must high`）。

---

### 10.2 分割 `assets/key_object/object_grounding_segmenter_v3.py`

**输入**：`object_grounding_queue.jsonl` + 视频 mp4
**算法**：

```text
对每个 candidate × expected shot:
  1. 在 visible_time_range_sec 内均匀采样 N 帧
  2. GroundingDINO 多 phrase 检测 → 聚类（IoU 合并）→ 选最佳 cluster（带 ambiguous 标记）
  3. SAM 3.1 用 box 作几何 prompt → mask（按 IoU 选最佳实例）
  4. 写 object_observations.jsonl: status, selected bbox/mask, area, sam_score, assets
```

**V5 多帧实例选择与窗口内补采样**（默认开启，`--window-resample-max 0` 完整回滚旧行为）：

- 采样帧的检测簇按帧序链接成跨帧实例轨迹（同帧一条轨迹至多一个簇），按 **(支持帧数, 平均最高分, phrase 票数)** 选轨迹——连续性优先于单帧置信，错位单帧高置信不能胜出。
- 无轨迹或轨迹 ambiguous 时，在同一窗口内按 Laplacian 清晰度补采最多 `--window-resample-max`（默认 4）帧再链一次；支持帧数 < `--min-track-support-frames`（默认 2）或前两名并列保持 ambiguous 且**不发资产**（`status="ambiguous_instance"`），选中轨迹缺席的帧记 `instance_track_absent`。
- 输出行升级 `object-observation-v4`（保留全部 v3 字段，新增 `sampling_pass/frame_sharpness/resample/cross_frame_support`），summary 升级 `object-grounding-segmentation-run-v4` 并附 `window_resample` 统计。

**产物**：

```text
<object_id>/mask.png        原始 mask
<object_id>/bcrop.png       bbox crop
<object_id>/masked_crop.png mask 外的透明合成
<object_id>/bbox.png       画框可视化
```

---

## 11. 资产侧：CHR ↔ S-ID 对齐 `assets/character_role_aligner.py`

**问题**：视觉资产 CHR001 自成 ID 体系，与 caption 的 `S1/S2` 不打通。

**解法**：Stage1.5 已把每条 tracklet 对齐到 `global_XXXX:S_ID`；Stage3 registry 记录每个 CHR 的成员 tracklet。对每个 CHR 做**成员加权投票**：

```text
weight(tracklet) = confidence × decision_factor
decision_factor: keep_primary=1.0, keep_supporting=0.85, keep_context=0.5
reject_*/review/unknown/non_person 不计票
```

票重先按 `(scope, shot_id)` 局部聚合（shot_id = tracklet_id 去掉 `_T\d+` 后缀）：同一 shot 内同一 subject 只计最大单票，轨迹碎片不会重复计票；再跨 shot 求和。

**按 global scope 独立判定**：

| 状态 | 条件 |
|---|---|
| `confirmed` | 第一名票数占比 ≥ `--min-share`（默认 0.6）且总权重 ≥ `--min-weight`（默认 1.0） |
| `contested` | 前两名占比差 < `--contest-margin`（默认 0.15） |
| `weak` | 有票但未达 confirmed |
| `unresolved` | 无任何有效证据 |

**只有 `confirmed` 进入 `role_claims` 并物化**；contested/weak/unresolved 仅留在报告 `chr_to_roles`，不参与 winner 竞争。

**schema_version**：`character-role-alignment-v2`（新增 `shot_votes` 聚合层与 `cross_scope_aliases` 展示）

**产物**：

```text
<output>/
  role_alignment.json                       完整报告（投票明细、冲突、未对齐项）
  characters/<scope>/<S_ID>/
    visual/                                 ← 复制自 Stage4 的 CHR 画廊（face/body/...）
    voice/                                  ← 复制自 voice_asset_manifest.jsonl 中 approved+matched
                                              行的 consolidated 净语音 reference_audio；无 manifest
                                              或无获批行则记 voice_skipped_reason、不复制语音
    role_character.json                     ← 该角色的汇总清单
```

**冲突**（多个 CHR 认领同一 S-ID）：只物化权重最高者，其余记入 `id_conflicts`，画廊仍保留在 Stage4 的 CHR 目录中，**不静默合并**。

---

## 12. 推荐运行顺序

### 12.1 主链路（单视频）

```bash
PY=/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annot/bin/python
$PY -m extract_full_pipeline.mimo_video_pipeline \
    --input /path/to/video.mp4 \
    --work-dir ./pipeline_outputs/<stem> \
    --stage all \
    [--thinking]
```

### 12.2 Scheme A（结构已预生成）

```bash
$PY -m extract_full_pipeline.mimo_video_pipeline_scheme_a \
    --input /path/to/video.mp4 \
    --work-dir ./scheme_a_outputs/<stem> \
    --global-input ./pipeline_outputs/<stem>/global \
    --shot-input ./pipeline_outputs/<stem>/shots \
    --segment-input ./upstream_segments/<stem>/segments.jsonl \
    --asr-input ./asr/<stem>.json \
    --stage all
```

### 12.3 资产链（并行于主链路）

```bash
# 1) 关键物体：依赖 shot_captions.jsonl
$PY -m extract_full_pipeline.assets.key_object.build_object_grounding_queue_v2 \
    --shot-captions ./pipeline_outputs/<stem>/captions/shot/shot_captions.jsonl \
    --output          ./assets/<stem>/key_object/queue.jsonl
$PY -m extract_full_pipeline.assets.key_object.object_grounding_segmenter_v3 \
    --queue ./assets/<stem>/key_object/queue.jsonl \
    --video /path/to/video.mp4 \
    --output-dir ./assets/<stem>/key_object/

# 2) 视觉人物：依赖 shot_captions.jsonl + global_captions.jsonl + shots/*.mp4
$PY -m extract_full_pipeline.assets.visual_person.shot_tracklet_builder \
    --shots-root   ./pipeline_outputs/<stem>/shots/ \
    --shot-captions ./pipeline_outputs/<stem>/captions/shot/shot_captions.jsonl \
    --output-dir    ./assets/<stem>/visual_person/stage1/
$PY -m extract_full_pipeline.assets.visual_person.semantic_role_tracklet_refiner_v3 \
    --stage1-dir    ./assets/<stem>/visual_person/stage1/ \
    --global-captions  ./pipeline_outputs/<stem>/captions/global/global_captions.jsonl \
    --shot-captions    ./pipeline_outputs/<stem>/captions/shot/shot_captions.jsonl \
    --segment-captions ./pipeline_outputs/<stem>/captions/segment/segment_captions.jsonl \
    --output-dir ./assets/<stem>/visual_person/stage1_5/
$PY -m extract_full_pipeline.assets.visual_person.visual_character_registry_builder \
    --stage1-dir  ./assets/<stem>/visual_person/stage1/ \
    --stage1-5-dir ./assets/<stem>/visual_person/stage1_5/ \
    --output-dir ./assets/<stem>/visual_person/stage3/
$PY -m extract_full_pipeline.assets.visual_person.character_asset_bank \
    --stage3-dir  ./assets/<stem>/visual_person/stage3/ \
    --stage1-dir  ./assets/<stem>/visual_person/stage1/ \
    --output-dir  ./assets/<stem>/visual_person/stage4/

# 3) Scheme A 音频（依赖 ASR 归属）
$PY -m extract_full_pipeline.mimo_global_captioner_speaker_v2 \
    --input       ./pipeline_outputs/<stem>/global \
    --asr-input   ./asr/<stem>.json \
    --output-dir  ./assets/<stem>/audio/global_speaker/
$PY -m extract_full_pipeline.assets.voice.export_all_asr_audio_review \
    --video      /path/to/video.mp4 \
    --global-dir ./pipeline_outputs/<stem>/global \
    --speaker-attributed-asr ./assets/<stem>/audio/global_speaker/speaker_attributed_asr.jsonl \
    --output-dir ./assets/<stem>/audio/review/

# 4) ID 统一
$PY -m extract_full_pipeline.assets.character_role_aligner \
    --identity        ./assets/<stem>/visual_person/stage3/ \
    --stage3-view     ./assets/<stem>/visual_person/stage1_5/stage3_inputs/primary/ \
    --asset-bank      ./assets/<stem>/visual_person/stage4/ \
    --voice-export    ./assets/<stem>/audio/review/ \
    --output          ./assets/<stem>/characters/
```

---

## 13. 下游依赖关系速查

| 上游产物 | 下游用途 |
|---|---|
| `<video>_shot_NNNNN.mp4` + `shots.jsonl` | global aggregator（首选源视频切）；shot caption；shot-level tracklet |
| `shots.jsonl.source_video` | global aggregator 解析源视频路径 |
| `global_clips.jsonl` | global caption；segment caption（`global clip_path` 兜底）；segment cutter |
| `global_captions.jsonl` | shot caption（subject_registry / event_timeline 上下文）；segment caption（V2 context）；Stage1.5 semantic 对齐 |
| `global_captions.jsonl.subject_registry` | shot caption 校验 S-ID；segment caption 校验 related_subject_ids |
| `shot_captions.jsonl` | segment caption（V2 的 shots 输入）；object grounding queue；Stage1.5 semantic 对齐（key_object_candidates） |
| `shot_captions.jsonl.key_object_candidates` | V2 segment caption 的 object_candidate_inventory；object_grounding_segmenter_v3 |
| `shot_captions.jsonl.visible_time_range_sec` | object_grounding_queue_v2 投影；object_grounding_segmenter_v3 帧采样窗口 |
| `segment_captions.jsonl` | segment cutter；Stage1.5 semantic 对齐 |
| `segments.jsonl`（mimo_segment_cutter 切出） | key_object 时窗；Stage1.5 时窗 |
| `tracklet_signatures.json` | Stage1.5 输入；Stage3 registry_builder 输入 |
| `tracklet_overlay.mp4` | Stage1.5 MiMo 对齐的关键可视化输入 |
| `semantic_identity_decisions.json` | Stage3 高置信 MUST_LINK；character_role_aligner 投票 |
| `stage3_inputs/primary/` | character_role_aligner 默认视图 |
| `registry.json`（Stage3） | character_asset_bank；character_role_aligner |
| `asset_bank/CHRnnn/` | character_role_aligner 复制到 `characters/<scope>/<S_ID>/visual/` |
| `speaker_attributed_asr.jsonl` | export_all_asr_audio_review 输入 |
| `valid/<scope>/<S_ID>/*.wav` | 人工复核 preview（不自动晋升正式资产） |
| `voice_asset_manifest.jsonl`（approved+matched 行的 reference_audio） | character_role_aligner 复制到 `characters/<scope>/<S_ID>/voice/` |
| `<scope>:S-ID` 对齐结果 | 训练/推理用统一 ID 资产目录 |

---

## 14. 已知约束 / 常见坑

1. **`mimo_caption_client.SOURCE_VIDEO_PROCESS_ROOT`** 默认指 `/gemini/platform/public/aigc/lss/Interns/lsj/video_process` —— 上线时按需改本地根。
2. **MiMo 服务端地址** 在 `mimo_caption_client.configure_no_proxy_for_url` 已处理 NO_PROXY，避免被工作站 HTTP proxy 拦截。
3. **`entity_asset_extractor`** 输出 `identity_id = null`，跨帧 identity 链接在 Stage3 视觉链或 character_role_aligner 投票中处理。
4. **`speaker_voice_asset_pipeline`** 输出 `consolidated/registry.json` 直接被 `multimodal_image_v3`（如存在）消费——是 Stage5 schema 的 drop-in 兼容。
5. **QC 数据结构（`QualityCard`）** 保留以便后续跑 QC 时直接回填，但不在主链路里强制。
6. **`mimo_video_pipeline.py`** 的 `split-shots` / `aggregate-global` 默认调用本目录的 `shot_splitter.py` / `global_clip_aggregator.py`（不再 import teleAI-T2V）。`segment-caption` stage **固定走 V2**，没有 `--segment-impl` 开关；需要 V1 语义时独立运行 `segment_captioner_legacy/`。
7. **`assets_runner.py`** 的 real 模式是**早期 segment 级演示入口**；当前推荐走 Scheme A 链（ARCHITECTURE.md 4.13/4.14）。
8. **`shot_splitter.py`** 与 **`global_clip_aggregator.py`** 的默认检测器源码目录从 teleAI-T2V 仓库改为 vendored `m02_shots/teleai_src/`——运行期**不 import teleAI-T2V 仓库的任何代码**（仅模型权重仍引用 `teleAI-T2V/local_models/`）。
9. **跳过解析失败的 caption 缓存**：`_captions_all_success` 校验每行 `success=True`，破损的 JSONL 不会被错误锁定；任何 stage 失败可干净重跑。
10. **Scheme A 中 `global-caption` 必须** `--asr-input`；否则抛错；`segment-caption` 必须 `--segment-input`（上游 segments.jsonl）。
11. **重试轮次**按 `60 * 2^(round-1)` 秒退避，封顶 300s；global / shot 默认 3 轮，segment 默认 6 轮。
12. **`--flat` 模式**只支持单视频输入（多 manifest 会报错）；产物直接落 `--output-dir`，不再追加 `<video_name>/` 子层。