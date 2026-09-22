# Xiaomi-CocktailASR-1 integration

This directory replaces the previous `faster-whisper-large-v3` ASR with
[Xiaomi-CocktailASR-1](https://huggingface.co/Ease3/Xiaomi-CocktailASR-1) — an
industrial-grade target-speaker ASR released by Xiaomi in September 2026.

## Why a wrapper?

CocktailASR-1 is fundamentally different from Whisper in two ways:

1. **It requires a reference voiceprint.** The model's signature is
   `model(target_audio, ref_audio)`. The output is a single string —
   the target speaker's transcription only. If the reference speaker is
   absent, the model returns the empty string (negative-sample rejection).
2. **It returns a single string, not a segment timeline.** Whisper emits
   `{start, end, text}` per segment; CocktailASR-1 emits one string per call.

`assets/voice/cocktail_asr.py` reconciles this with the rest of the pipeline:

* `CocktailASRTranscriber.transcribe(audio)` — auto-reference mode. The
  wrapper picks the loudest 1–4 second clip of the audio itself as the
  reference voiceprint, then returns a single row spanning the full audio.
  This is the drop-in equivalent of the previous Whisper call site in
  `speaker_registry_builder.Transcriber`.
* `CocktailASRTranscriber.transcribe_segments(audio, segments, ref_audio=...)`
  — per-segment mode. Use this when the pipeline has a known voiceprint
  (a Stage5 turn is a natural source). Consecutive non-empty segments
  with a small gap are merged into one transcription.

## Files

| File | Purpose |
|------|---------|
| `assets/voice/cocktail_asr.py` | The wrapper class and its CLI entry point. |
| `assets/voice/speaker_registry_builder.py` | The Stage5 ASR consumer. The existing `Transcriber` class now wraps `CocktailASRTranscriber`; the new `--asr-ref-audio` and `--asr-segments-json` flags wire it into the pipeline. |
| `local_models/Xiaomi-CocktailASR-1/` | Weights, config, tokenizer. |
| `test_cocktail_asr.py` | End-to-end test driver. |

## How to download

```bash
# Small files (~13 MB total): config, tokenizer, model code
for f in config.json configuration_mic_asr.py d2v2_config.json \
         feature_extraction_mic_asr.py modeling_mic_asr.py \
         tokenizer.json tokenizer_config.json README.md .gitattributes; do
  curl -sL --tlsv1.2 -o local_models/Xiaomi-CocktailASR-1/$f \
    "https://hf-mirror.com/Ease3/Xiaomi-CocktailASR-1/resolve/main/$f"
done

# Main checkpoint (~33 GB) — slow, do it once
curl -L --tlsv1.2 --max-time 7200 -o local_models/Xiaomi-CocktailASR-1/pytorch_model.bin \
  "https://hf-mirror.com/Ease3/Xiaomi-CocktailASR-1/resolve/main/pytorch_model.bin"
```

`hf-mirror.com` is used as a fallback because the official HuggingFace
`us.aws.cdn.hf.co` CDN times out from this network. The model files are
byte-for-byte identical — `hf-mirror` proxies the same Xet-backed storage.

## How to test

The test driver lives at the pipeline root: `test_cocktail_asr.py`.

```bash
# Auto-reference mode (no voiceprint needed)
python test_cocktail_asr.py /path/to/audio.wav

# Target-speaker mode (cocktail-party scenario)
python test_cocktail_asr.py /path/to/mixed.wav \
    --ref /path/to/speaker_clip.wav \
    --segment 12.5 30.0

# Through the pipeline shim (replicates what speaker_registry_builder does)
python test_cocktail_asr.py /path/to/audio.wav --via registry
```

Sample output:

```
[input] /tmp/test.wav
        duration=42.137s  sample_rate=16000  channels=1  frames=674192

--- mode: auto-reference (no explicit voiceprint) ---
[model] local_models/Xiaomi-CocktailASR-1 (device=cuda, dtype=bfloat16)
[result] 1 row(s) in 5.42s
  [0.00s - 42.14s] conf=1.000
    人们设计AI配方的过程本质上主要还是一个不断试错的过程
```

## Pipeline integration

`assets/voice/speaker_registry_builder.py` now exposes two new flags that
hand off control of the ASR run to the wrapper:

```
--asr-ref-audio PATH       # 16 kHz mono wav used as the reference voiceprint
--asr-segments-json PATH   # JSON list of {start, end} rows; one row per
                           # (merged) segment instead of one over the audio
```

When neither is provided, the shim falls back to the auto-reference mode
above. The output schema is unchanged — Stage5 still sees
`{asr_id, start, end, text, confidence}` rows and the downstream
`has_lexical_speech` gate still works on the `text` field.

## Caveats

* **Confidence is binary.** CocktailASR-1 does not emit per-segment
  logprobs, so `confidence` is `1.0` when the model produced text and
  `0.0` when it returned the empty string. Whisper's `exp(avg_logprob)` is
  gone, but the downstream `has_lexical_speech` gate only inspects the
  text, not the confidence.
* **Auto-reference mode is not a general multi-speaker ASR.** With no
  reference, the wrapper picks the loudest 1–4 second run of speech and
  treats that voice as the target. Multi-speaker clips will only be
  transcribed for that voice. For full multi-speaker coverage, use
  `--asr-ref-audio` per speaker after Stage5 has produced a voiceprint, or
  pair this with an external diarization step that loops over each
  speaker.
* **33 GB checkpoint.** The model is 8B parameters (Qwen3-8B decoder +
  Data2Vec2 encoder). bfloat16 inference needs ~16 GB of VRAM. Make sure
  the host has that headroom before loading.

## References

* [HuggingFace model page](https://huggingface.co/Ease3/Xiaomi-CocktailASR-1)
* [GitHub repository](https://github.com/xiaomi-research/xiaomi-cocktailasr-1)
* [Technical report (arXiv 2609.11274)](https://arxiv.org/pdf/2609.11274)
