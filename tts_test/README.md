# TTS 生成效果对比

本目录包含两个本地声音克隆实现：`tts_voxcpm2.py` 和
`tts_fish_s2pro.py`。各自的虚拟环境、权重与 Fish Speech 源码位于
`.venvs/`、`local_models/` 和 `third_party/`；这些本地内容未必存在于远端 Git。

当前 `run_full_pipeline.py` 默认在角色物化后调用批处理入口：

```bash
python tts_test/run_pipeline_tts.py \
  --workspace ../workspace/<run>/<video> \
  --dry-run
```

批处理读取 `characters/<scope>/<subject>/voice/valid/*.wav` 和
`captions/global/speaker_attributed_asr.jsonl`，按参考音频总时长排序，默认选择
最多 3 个至少 5 秒的角色，并分别运行 VoxCPM2 与 Fish S2 Pro。输出为
`tts_test/outputs/<model>/<scope>_<subject>/`，根目录另写 `manifest.json`；已有
`meta.json` 会断点跳过。`--max-speakers 0` 表示全量，统一编排的 `--no-tts`
可关闭本阶段。

两个原子脚本的 `--transcripts` 同时接受 JSON、JSONL 和 basename→文本映射。
TTS 结果只用于声音生成效果对比，不代表候选声音已经人工审核或人物身份正确。
