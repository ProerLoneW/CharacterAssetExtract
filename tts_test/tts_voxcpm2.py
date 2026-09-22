#!/usr/bin/env python3
"""Generate a clean single-character voice asset with VoxCPM2 (zero-shot voice cloning).

The character's raw voice folder usually contains many very short clips (1-2s),
which are too short for stable cloning. This script:
  1. Scans a folder of reference audio clips of ONE character.
  2. Optionally loads transcripts (pipeline asr_timeline.json / basename mapping
     / faster-whisper auto transcription).
  3. Concatenates clips (same speaker) into one reference wav of suitable length.
  4. Synthesizes a clean, prosody-rich voice asset designed for downstream
     video-generation models (audio-driven avatar / lip-sync reference).

Model weights: tts_test/local_models/VoxCPM2 (hf download openbmb/VoxCPM2)

Example:
  python tts_voxcpm2.py \
      --ref-dir ../workspace/movie_full_pipeline_20260912/assets/characters/global_0040/S2/voice \
      --transcripts ../workspace/movie_full_pipeline_20260912/assets/voice/global_0040/stage5/asr_timeline.json \
      --text-preset en_expressive \
      --out-dir ./outputs/voxcpm2/global_0040_S2

  python tts_voxcpm2.py \
      --ref-dir ../workspace/movie_full_pipeline_20260912/assets/characters/global_0041/S3/voice \
      --transcripts ../workspace/movie_full_pipeline_20260912_v3/assets/voice/global_0041/stage5/asr_timeline.json \
      --text-preset en_expressive \
      --out-dir ./outputs/voxcpm2/global_0041_S3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Text presets for the synthesized voice asset.
#
# Design goals (the audio feeds video-generation models as a voice reference):
#   * single speaker, clean studio-like delivery, no shouting / whispering
#   * 20-35 s of speech so the downstream model sees enough vocal range
#   * mixed prosody: statement + question + mild emotion + natural pauses
#   * no proper nouns / numbers that are easy to mispronounce
# ---------------------------------------------------------------------------
TEXT_PRESETS = {
    "en_expressive": (
        "You know, when I first heard the news, I honestly couldn't believe it. "
        "But then I sat down and thought about it for a while, and slowly, "
        "it all started to make sense. So here is what we are going to do: "
        "we take our time, we do it right, and we do not look back. "
        "Does that sound like a plan? Good. Because I promise you, "
        "this is going to be something truly special."
    ),
    "en_calm": (
        "Take a deep breath, and let the day slow down for a moment. "
        "There is no rush. We have plenty of time to get this right, "
        "so let's just enjoy the process, one small step at a time."
    ),
    "zh_expressive": (
        "说实话，刚听到这个消息的时候，我还有点不敢相信。"
        "不过静下心来想了想，其实一切都有迹可循。"
        "所以接下来，我们一步一步来，把每一件事情都做好。"
        "你觉得怎么样？好，那就这么说定了。"
        "相信我，这一次，一定会非常精彩。"
    ),
    "zh_calm": (
        "别急，先深呼吸，让自己慢下来。"
        "时间还很充裕，我们一点一点来，"
        "把每一步都走稳，结果自然不会差。"
    ),
}

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_CJK_RE = re.compile(r"[一-鿿]")


def load_transcripts(path: Path) -> dict:
    """Accepts pipeline asr_timeline.json (list of {asr_id, text}) or a plain
    {basename_or_stem: text} mapping. JSONL emitted by the current mainline is
    accepted as well. Returns {stem: text}."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        data = data["segments"]
    mapping = {}
    if isinstance(data, dict):
        items = data.items()
        for k, v in items:
            mapping[Path(str(k)).stem] = str(v).strip()
    elif isinstance(data, list):
        for seg in data:
            if not isinstance(seg, dict):
                continue
            seg_id = seg.get("asr_id") or seg.get("id") or seg.get("audio_path")
            text = (seg.get("text") or "").strip()
            if seg_id and text:
                mapping[Path(str(seg_id)).stem] = text
    return mapping


def whisper_transcribe(clips: list, model_dir: str, language: str = None) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_dir, device="cuda", compute_type="float16")
    out = {}
    for c in clips:
        segments, info = model.transcribe(str(c), language=language, beam_size=5)
        out[c.stem] = "".join(s.text for s in segments).strip()
        print(f"  [whisper] {c.name}: {out[c.stem]}")
    return out


def build_reference(ref_dir: Path, transcripts: dict, target_seconds: float,
                    max_seconds: float, gap_ms: int, work_dir: Path):
    """Concatenate same-speaker clips into one reference wav + aligned transcript."""
    clips = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in _AUDIO_EXTS)
    if not clips:
        sys.exit(f"[error] no audio files found in {ref_dir}")

    waves, texts, total, sr = [], [], 0.0, None
    for clip in clips:
        wav, cur_sr = sf.read(str(clip), dtype="float32", always_2d=False)
        if wav.ndim > 1:  # downmix stereo
            wav = wav.mean(axis=1)
        if sr is None:
            sr = cur_sr
        if cur_sr != sr:
            print(f"  [warn] {clip.name} sample rate {cur_sr} != {sr}, skipped")
            continue
        if total >= max_seconds:
            break
        waves.append(wav)
        t = transcripts.get(clip.stem, "")
        if t:
            texts.append(t)
        total += len(wav) / sr
        if total >= target_seconds:
            break

    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    merged = []
    for i, w in enumerate(waves):
        if i:
            merged.append(gap)
        merged.append(w)
    merged = np.concatenate(merged) if merged else np.zeros(0, dtype=np.float32)
    peak = np.abs(merged).max()
    if peak > 0:
        merged = merged * (0.95 / peak if peak > 0.95 else 1.0)

    work_dir.mkdir(parents=True, exist_ok=True)
    ref_path = work_dir / "reference_merged.wav"
    sf.write(str(ref_path), merged, sr)
    ref_text = " ".join(texts)
    print(f"[reference] {len(waves)} clips -> {ref_path} "
          f"({len(merged)/sr:.1f}s, sr={sr}), transcript {len(ref_text)} chars")
    return ref_path, ref_text


def pick_preset(ref_text: str, preset: str) -> str:
    if preset != "auto":
        return TEXT_PRESETS[preset]
    return TEXT_PRESETS["zh_expressive" if _CJK_RE.search(ref_text or "") else "en_expressive"]


def main():
    ap = argparse.ArgumentParser(description="Clean character voice asset via VoxCPM2")
    ap.add_argument("--ref-dir", required=True, type=Path,
                    help="folder with the character's reference audio clips")
    ap.add_argument("--transcripts", type=Path, default=None,
                    help="asr_timeline.json or {name: text} json for the clips")
    ap.add_argument("--whisper-model", type=str, default=None,
                    help="faster-whisper model dir to transcribe clips missing text "
                         "(default: auto-detect ../local_models/faster-whisper-large-v3)")
    ap.add_argument("--text-preset", default="auto",
                    choices=["auto"] + list(TEXT_PRESETS))
    ap.add_argument("--text", default=None, help="override synthesis text entirely")
    ap.add_argument("--style-prompt", default=None,
                    help="optional VoxCPM2 style hint, e.g. 'warm, slightly faster'")
    ap.add_argument("--target-ref-seconds", type=float, default=12.0)
    ap.add_argument("--max-ref-seconds", type=float, default=25.0)
    ap.add_argument("--gap-ms", type=int, default=150)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--num-candidates", type=int, default=1,
                    help="generate N takes (cloning varies run to run; 2-3 recommended)")
    ap.add_argument("--model-dir", type=Path,
                    default=Path(__file__).parent / "local_models" / "VoxCPM2")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "outputs" / "voxcpm2")
    args = ap.parse_args()

    transcripts = load_transcripts(args.transcripts) if args.transcripts else {}

    if not transcripts:
        whisper_dir = args.whisper_model
        if whisper_dir is None:
            for cand in (Path(__file__).parent / "local_models" / "faster-whisper-large-v3",
                         Path(__file__).parent.parent / "local_models" / "faster-whisper-large-v3"):
                if cand.exists():
                    whisper_dir = str(cand)
                    break
        if whisper_dir:
            try:
                clips = sorted(p for p in args.ref_dir.iterdir()
                               if p.suffix.lower() in _AUDIO_EXTS)
                transcripts = whisper_transcribe(clips, whisper_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] whisper transcription failed ({e}); cloning without prompt text")

    ref_path, ref_text = build_reference(
        args.ref_dir, transcripts, args.target_ref_seconds,
        args.max_ref_seconds, args.gap_ms, args.out_dir)
    # language of the preset follows the reference transcript unless overridden
    text = args.text or pick_preset(ref_text, args.text_preset)
    if args.style_prompt:
        text = f"({args.style_prompt}){text}"
    print(f"[text] {text}")

    from voxcpm import VoxCPM

    print(f"[model] loading VoxCPM2 from {args.model_dir}")
    model = VoxCPM.from_pretrained(str(args.model_dir), load_denoiser=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sr = model.tts_model.sample_rate
    for i in range(args.num_candidates):
        import random
        import torch
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        wav = model.generate(
            text=text,
            prompt_wav_path=str(ref_path) if ref_text else None,
            prompt_text=ref_text or None,
            reference_wav_path=str(ref_path),
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
        )
        out = args.out_dir / f"voice_asset_take{i+1}.wav"
        sf.write(str(out), wav, sr)
        print(f"[done] {out} ({len(wav)/sr:.1f}s @ {sr}Hz)")

    meta = {
        "model": "openbmb/VoxCPM2",
        "ref_dir": str(args.ref_dir),
        "reference_wav": str(ref_path),
        "reference_text": ref_text,
        "synthesis_text": text,
        "cfg": args.cfg, "timesteps": args.timesteps, "seed": args.seed,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
