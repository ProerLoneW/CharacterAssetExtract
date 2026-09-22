#!/usr/bin/env python3
"""Generate a clean single-character voice asset with Fish Audio S2 Pro
(zero-shot voice cloning) via the fish-speech toolkit.

The character's raw voice folder usually contains many very short clips (1-2s),
which are too short for stable cloning. This script:
  1. Scans a folder of reference audio clips of ONE character.
  2. Optionally loads transcripts (pipeline asr_timeline.json / basename mapping
     / faster-whisper auto transcription).
  3. Concatenates clips (same speaker) into one reference wav of suitable length.
  4. Calls fish_speech text2semantic inference (with inline [tag] prosody control)
     to synthesize a clean, prosody-rich voice asset for downstream
     video-generation models (audio-driven avatar / lip-sync reference).

Model weights: tts_test/local_models/s2-pro (hf download fishaudio/s2-pro)
Toolkit:         tts_test/third_party/fish-speech (pip install -e .[cu128])

Example:
  .venvs/fish/bin/python tts_fish_s2pro.py \
      --ref-dir ../workspace/movie_full_pipeline_20260912/assets/characters/global_0040/S2/voice \
      --transcripts ../workspace/movie_full_pipeline_20260912/assets/voice/global_0040/stage5/asr_timeline.json \
      --text-preset en_expressive \
      --out-dir ./outputs/s2pro/global_0040_S2
"""

import argparse
import json
import re
import subprocess
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
#   * S2 Pro supports free-form inline [tags]; use them sparingly for pauses
#     and gentle emotion shifts only, to keep the timbre clean
# ---------------------------------------------------------------------------
TEXT_PRESETS = {
    "en_expressive": (
        "You know, when I first heard the news, I honestly couldn't believe it. "
        "[short pause] But then I sat down and thought about it for a while, "
        "and slowly, it all started to make sense. "
        "So here is what we are going to do: "
        "we take our time, we do it right, and we do not look back. "
        "[short pause] Does that sound like a plan? "
        "Good. [warm tone] Because I promise you, "
        "this is going to be something truly special."
    ),
    "en_calm": (
        "Take a deep breath, and let the day slow down for a moment. "
        "[short pause] There is no rush. "
        "We have plenty of time to get this right, "
        "so let's just enjoy the process, one small step at a time."
    ),
    "zh_expressive": (
        "说实话，刚听到这个消息的时候，我还有点不敢相信。"
        "[停顿] 不过静下心来想了想，其实一切都有迹可循。"
        "所以接下来，我们一步一步来，把每一件事情都做好。"
        "你觉得怎么样？好，那就这么说定了。"
        "[温和的语气] 相信我，这一次，一定会非常精彩。"
    ),
    "zh_calm": (
        "别急，先深呼吸，让自己慢下来。"
        "[停顿] 时间还很充裕，我们一点一点来，"
        "把每一步都走稳，结果自然不会差。"
    ),
}

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_CJK_RE = re.compile(r"[一-鿿]")

_HERE = Path(__file__).resolve().parent
_FISH_REPO = _HERE / "third_party" / "fish-speech"


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
        for k, v in data.items():
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
        segments, _ = model.transcribe(str(c), language=language, beam_size=5)
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


def main():
    ap = argparse.ArgumentParser(description="Clean character voice asset via Fish Audio S2 Pro")
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
    ap.add_argument("--target-ref-seconds", type=float, default=12.0)
    ap.add_argument("--max-ref-seconds", type=float, default=25.0)
    ap.add_argument("--gap-ms", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--num-candidates", type=int, default=1,
                    help="generate N takes (cloning varies run to run; 2-3 recommended)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (~10x faster after warmup)")
    ap.add_argument("--chunk-length", type=int, default=300)
    ap.add_argument("--model-dir", type=Path,
                    default=_HERE / "local_models" / "s2-pro")
    ap.add_argument("--out-dir", type=Path, default=_HERE / "outputs" / "s2pro")
    args = ap.parse_args()

    transcripts = load_transcripts(args.transcripts) if args.transcripts else {}

    if not transcripts:
        whisper_dir = args.whisper_model
        if whisper_dir is None:
            for cand in (_HERE / "local_models" / "faster-whisper-large-v3",
                         _HERE.parent / "local_models" / "faster-whisper-large-v3"):
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

    if args.text:
        text = args.text
    else:
        preset = args.text_preset
        if preset == "auto":
            preset = "zh_expressive" if _CJK_RE.search(ref_text or "") else "en_expressive"
        text = TEXT_PRESETS[preset]
    print(f"[text] {text}")

    infer_script = _FISH_REPO / "fish_speech" / "models" / "text2semantic" / "inference.py"
    if not infer_script.exists():
        sys.exit(f"[error] fish-speech repo not found at {_FISH_REPO}; "
                 f"clone https://github.com/fishaudio/fish-speech there first")

    cmd = [
        sys.executable, str(infer_script),
        "--text", text,
        "--checkpoint-path", str(args.model_dir.resolve()),
        "--output", str((args.out_dir / "voice_asset.wav").resolve()),
        "--output-dir", str((args.out_dir / "codes").resolve()),
        "--num-samples", str(args.num_candidates),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--seed", str(args.seed),
        "--chunk-length", str(args.chunk_length),
    ]
    if args.compile:
        cmd.append("--compile")
    cmd += ["--prompt-audio", str(ref_path.resolve())]
    if ref_text:
        cmd += ["--prompt-text", ref_text]

    print(f"[run] {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, cwd=str(_FISH_REPO), check=True)

    meta = {
        "model": "fishaudio/s2-pro",
        "ref_dir": str(args.ref_dir),
        "reference_wav": str(ref_path),
        "reference_text": ref_text,
        "synthesis_text": text,
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "seed": args.seed,
        "num_candidates": args.num_candidates,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
