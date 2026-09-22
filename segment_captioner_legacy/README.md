# segment_captioner_legacy —— 原始"global→shot→segment 级联"实现

## 用途

本目录保留**最初版本**的 segment caption 实现，作为与 `mimo_segment_captioner_v2`（位于 extract_full_pipeline 顶层）并行的另一条逻辑分支。当需要回到原系统的语义（仅做"基于 global+shot 上下文，由 MiMo 一次性给出 15-30s segment 切分与描述"，不引入 object/duration_contract 等 V2 约束）时使用本目录。

## 与 `mimo_segment_captioner_v2.py` 的对比

| 维度 | `segment_captioner_legacy/segment_captioner.py`（原版 / V1） | `mimo_segment_captioner_v2.py`（V2 = V1 + 物体 + 合约） |
|---|---|---|
| Segment 时长 | 15-30s | 15-30s |
| 时间基准 | 同时维护 `source_time_range_sec` 与 `global-local time_range_sec` 双轨 | 仅 `global-local` |
| ASR 旁路 | `prepare_authoritative_asr` + `_read_asr_file` + `_global_scope` 自动归一到 global | 不读 ASR |
| 对话标注 | `dialogue_tags` + `deduplicate_dialogue_tags` + `align_authoritative_dialogue`（自动套 speaker id）| 不做 dialogue tagging |
| 说话人归属 | `flatten_speaker_attributions` + `_attribution_confidence` + `_attribution_evidence_type` | 不处理说话人 |
| Scene 转移标记 | `remove_scenetrans_markers` | 不做 |
| Shot 标签完整性检查 | `has_all_shot_labels` + `fallback_shot_labeled_description` | 不强制 |
| 物体候选 | 不显式管理 | `object_candidate_inventory` + `valid_boundary_guide` + `normalize_segment_objects`（V2 物体合约） |
| 顶层附加字段 | 仅 `success` / `error` / `response_seconds` | 加 `object_candidate_index` / `object_quality_issues` / `duration_contract` / `schema_version=segment-caption-key-objects-v2` |
| Prompt | `default_prompt_path("segment")`（V1 prompt，未做角色对齐）| 同上（V2 未改 prompt，但 v3 shot_captioner 用 V3 prompt） |

> **关键差异**：V2 的"v1"在源码层就是 `import mimo_segment_captioner as v1` —— V2 把本目录的实现当作底层，叠加物体合约。`segment_captioner_legacy/segment_captioner.py` 让您可以**独立运行** V1 逻辑而不需要 V2 的物体合约。

## 调用方式

```bash
PY=/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annot/bin/python
# 从 extract_full_pipeline/ 顶层 run：
$PY -m extract_full_pipeline.segment_captioner_legacy.segment_captioner \
    --global-input ./outputs/global_caption_outputs/<video> \
    --shot-input   ./outputs/shot_caption_outputs/<video> \
    --output-dir   ./outputs/segment_caption_outputs
```

> 注意：`mimo_video_pipeline.py` 目前没有 `--segment-impl` 开关（segment-caption stage 固定走 V2）。
> 需要 V1 语义时，按上面方式直接独立运行本目录即可。

## 输出 schema

```text
<output>/<video>/
├── segments/global_<id>.json          # 单个 global clip 的 segment 结果
├── raw_responses/global_<id>.txt       # 模型原始回复
├── segment_captions.jsonl             # 所有 global clip 的合并清单
└── summary.json                       # 成功/失败计数
```

每条 `segment_captions.jsonl` 行：
```json
{
  "global_id": 1,
  "success": true,
  "model": "MiMo-V2.5",
  "segment_caption": {
    "segments": [
      {
        "segment_id": "seg_001",
        "time_range_sec": [12.3, 27.6],
        "source_time_range_sec": [142.3, 157.6],
        "description": "[S1][S2] ... {Speaker_001}",
        "speaker_attributions": [...]
      }
    ]
  },
  "response_seconds": 14.2
}
```

## 依赖

- `mimo_caption_client`（同目录层级的 `mimo_caption_client.py`）
- `mimo_segment_captioner_v2`（**不依赖**——本目录独立运行；V2 反过来依赖本目录作为 v1）

## 何时使用

- 需要 ASR/说话人归属/dialogue tag → 用本目录（V1）
- 需要 object inventory / duration contract / 物体 grounding 后续链路 → 用 V2
- A/B 对比 → 两套都可独立运行