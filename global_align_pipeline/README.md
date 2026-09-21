# Global Align Pipeline（global 尺度 Tracklet→S-ID 直连，跳过 CHR 聚类）

## 为什么

v4 链路（shot 级 Stage1.5 → Stage3 CHR 聚类 → Stage5 投票）召回损耗大：
190 tracklet → primary 视图 42 → 7 CHR + 29 个被丢弃的 pending candidate
→ 最终只有 10 个 (scope, S-ID) 绑定。本流程把每个 global clip 的
shot 级 `tracklet_overlay.mp4` 拼回一整段视频，让 MiMo 在 global 尺度上
直接完成“轨迹框 ↔ 角色 ID”的对应，完全跳过聚类。

## 流程

```text
tracklets/shots/<SHOT>/tracklet_overlay.mp4        （box 上带全局唯一 tracklet_id）
  → ffmpeg concat → stitched/global_00XX_tracklets.mp4
  → MiMo（video + caption 证据 + 全量 tracklet inventory）      global_tracklet_aligner.py
  → global_tracklet_alignment.jsonl / alignments/global_00XX.json
  → 按 (scope, S-ID) 物化 visual（face/body/rgba crop）+ voice   global_character_materializer.py
  → characters/<scope>/<S-ID>/{visual,voice,role_character.json}
```

与旧链的关键差异：

- MiMo 一次看整个 global（约 90s、19 个 shot），可跨 shot 比对外观后归并
  同一角色，不再依赖人脸聚类。
- 不做 primary/supporting 视图过滤：任何被指到 S-ID 的 tracklet 都贡献
  资产（prompt 明确要求“宁可选上也不要 unknown”）。
- 输出 schema：`global-tracklet-character-alignment-v1`。

## 运行（服务器上）

```bash
bash global_align_pipeline/run_global_align_pipeline.sh
# 可用环境变量覆盖：SRC_WS / OUT_WS / PY
```

结果默认写到 `movie_full_pipeline_20260915_global_align_test/`：

```text
global_alignment/
  stitched/global_00XX_tracklets.mp4     # 拼接后的带框视频（可人工抽查）
  requests/global_00XX_package.json      # 发给模型的完整输入包（审计用）
  alignments/global_00XX.json
  global_tracklet_alignment.jsonl
  summary.json                           # assignment_rate = assigned/total
characters/<scope>/<S-ID>/
  visual/{face,body,rgba}/*.jpg|*.png    # top-K by quality，跨 tracklet 合并
  voice/{valid,invalid}/*.wav            # 复用 v4 的 voice_by_role_src
  role_character.json                    # tracklet 列表、置信度、统计
materialization_summary.json
```

## 单独运行 / 调试

```bash
# 只拼视频 + 生成请求包，不调用模型（验证拼接与 inventory）
python global_align_pipeline/global_tracklet_aligner.py ... --dry-run

# 只要视觉资产，或调大 top-k / 放宽置信度
python global_align_pipeline/global_character_materializer.py \
  --alignment ... --tracklets ... --voice-root ... --output ... \
  --face-top-k 12 --min-confidence 0.0 --include-invalid-voice
```

## 已知问题：混入他人照片与身份一致性过滤

实测发现 MiMo 会把同一镜头里的相邻被摄者误并到同一 S-ID（案例：
SHOT_00790 中床上的小病人 T000 与床边女青年 T001 都被标成 S1，且两者
时间重叠但 Stage1 没有生成 cannot_link，无硬约束可拦截）。两层防护：

1. **prompt 侧（上游）**：`prompts/global_tracklet_alignment_prompt_en_v1.md`
   的 "Identity-mismatch guard" 段要求跨 shot 归并前比对年龄/脸型/体型，
   并明确"同 shot 时间重叠的 tracklet 通常是不同人"。修改需重跑 Step 1。
2. **materializer 侧（下游，不重跑 MiMo 即可生效）**：`--identity-check`
   复用 Stage3 同款 InsightFace（buffalo_l）对每个 (scope, S-ID) 内各
   tracklet 的 top-3 人脸算平均 embedding，两两 cosine 建图（阈值
   `--face-sim-threshold`，默认 0.40），最大连通分量（置信度加权破平）为
   身份核，核外的 tracklet 判为 outlier：
   - 默认仅剔除 outlier 的 face crop（body/rgba 保留）；
   - `--exclude-outlier-visual` 连 body/rgba 一起剔除；
   - 结果写进 `role_character.json::identity_check`（含相似度矩阵）与
     `identity_conflicts.json`，供人工复核/重新指派；
   - fail-open：模型或人脸 crop 不可用时跳过检查（不会清空画廊），
     全体人脸互不相似的弱证据角色标记 `inconclusive` 并保留。

   注意：outlier 只是被隔离，不会被自动改派到别的 S-ID——错误指派的
   纠正仍需人工看 `identity_conflicts.json`。

## 已知问题：画中人轨迹（照片/海报/屏幕/镜面）与 entity_nature

Stage1 检测栈（GroundingDINO + InsightFace + SAM）只能判断"画面里有没有
人形"，照片、海报、电视/手机屏幕、镜面、雕像里的人同样会被追成轨迹；
且当画面上是主角本人时，人脸 embedding 与角色核完全一致，identity-check
也拦不住。为此对齐结果新增与 S-ID 指派**解耦**的语义维度：

```json
"entity_nature": "real_person | depicted_person | non_person | uncertain"
```

- **prompt**：要求 MiMo 对每个轨迹判定实体性质（画框/屏幕边框/刚体平面
  移动/caption 提及 photo/poster/screen/mirror 等线索），照片上是主角时
  仍可 `subject_id=S1 + entity_nature=depicted_person`——归属与性质互不
  影响；背景真人必须仍是 `real_person`，不许用该字段清洗配角。
- **aligner**：schema 升至 `global-tracklet-character-alignment-v2`，非法/
  缺失值降级 `uncertain`（不触发整包重试）；`summary.json` 增加
  `entity_nature_totals`。
- **materializer**：`depicted_person`/`non_person` 的轨迹不进资产库，但按
  其指派的 S-ID 记录在 `role_character.json::depicted_references`（轨迹、
  置信度、理由），供剧情侧后续使用；某 S-ID 只有 depicted 轨迹时不建目录。
  `uncertain` 默认照常物化（fail-open 保召回），加
  `--strict-entity-nature` 才一并 withhold。identity-check 只在 real
  tracklet 上运行，主角照片不会污染人脸核。
- **兼容性**：读取旧 v1 alignment（无 entity_nature 字段）时全部按
  `uncertain` 处理，默认行为与旧版一致；语音链不受影响（说话人归属来自
  ASR，与轨迹无关）。

## 校验要点

| 检查 | 文件 | 期望 |
| --- | --- | --- |
| 拼接完整性 | `stitched/*.mp4` | 3 个 global，时长 ≈ global duration |
| 分配召回 | `global_alignment/summary.json` | `assignment_rate` 显著高于 v4 的 42/190≈22% |
| 跨 shot 归并 | `alignments/*.json` | 同一 S-ID 含多个不同 SHOT 前缀的 tracklet |
| cannot-link 尊重 | `alignments/*.json` | 同帧共现 tracklet 不应同 S-ID（可脚本核查） |
| 资产物化 | `materialization_summary.json` | character 数、voice_copied > 0 |

已知限制：语音资产沿用 v4 的句级归属结果（S-ID 口径一致，直接可复用）；
物体与场景链本次不涉及。
