#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test driver for the Xiaomi-CocktailASR-1 wrapper.

This script exercises every public entry point of
``assets.voice.cocktail_asr.CocktailASRTranscriber`` so you can sanity-check a
new audio file end-to-end without going through the speaker registry
pipeline.

It does three things, in this order:

1. Builds a 16 kHz mono copy of the input (any wav/flac/mp3 works) and prints
   a one-line summary so you can see what the model is about to transcribe.
2. Runs ``transcribe(audio)`` -- the auto-reference mode -- and prints the
   single row.  This is what the legacy ``speaker_registry_builder.Transcriber``
   call site falls back to, so it's the closest analogue of the previous
   faster-whisper pipeline.
3. (Optional) Runs ``transcribe_segments(audio, segments)`` against a tiny
   list of demo segments if you want to confirm the per-turn path that
   ``speaker_registry_builder`` uses once a Stage5 voiceprint is available.

Usage
-----

The script works in three modes:

A. Just point it at a wav and let it run end-to-end (auto-reference).

       python test_cocktail_asr.py /path/to/audio.wav

B. Tell it to transcribe only one target segment with your own reference
   voiceprint (target-speaker mode).  This is the cocktail party scenario
   the model is designed for.

       python test_cocktail_asr.py /path/to/mixed.wav \\
           --ref /path/to/speaker_clip.wav \\
           --segment 12.5 30.0

C. Use it through the legacy ``speaker_registry_builder.Transcriber`` shim
   to confirm the pipeline integration is intact.

       python test_cocktail_asr.py /path/to/audio.wav --via registry

The model weights are read from
``local_models/Xiaomi-CocktailASR-1`` by default; override with
``--model-dir /path/to/weights`` if you keep them elsewhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = REPO_ROOT / "local_models" / "Xiaomi-CocktailASR-1"
SAMPLE_RATE = 16000


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio to 1D float32 mono at the wrapper's native rate (16 kHz)."""
    waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(
            __import__("torch").from_numpy(waveform), sr, SAMPLE_RATE).numpy()
    return waveform.astype(np.float32, copy=False), SAMPLE_RATE


def _print_audio_summary(audio_path: Path) -> dict[str, Any]:
    """Print a one-line description of the input and return its metadata."""
    info = sf.info(str(audio_path))
    print(f"[input] {audio_path}")
    print(f"        duration={info.duration:.3f}s  sample_rate={info.samplerate}  "
          f"channels={info.channels}  frames={info.frames}")
    return {"path": str(audio_path), "duration": info.duration,
            "sample_rate": info.samplerate, "channels": info.channels}


def test_auto_reference(audio_path: Path, model_dir: Path, device: str,
                        dtype: str, output_path: Path | None) -> None:
    """Run ``CocktailASRTranscriber.transcribe`` and print the result."""
    sys.path.insert(0, str(REPO_ROOT / "assets" / "voice"))
    from cocktail_asr import CocktailASRTranscriber

    print("\n--- mode: auto-reference (no explicit voiceprint) ---")
    print(f"[model] {model_dir} (device={device}, dtype={dtype})")
    transcriber = CocktailASRTranscriber(model_dir=str(model_dir), device=device, dtype=dtype)
    started = time.perf_counter()
    rows = transcriber.transcribe(str(audio_path))
    elapsed = time.perf_counter() - started
    print(f"[result] {len(rows)} row(s) in {elapsed:.2f}s")
    for row in rows:
        print(f"  [{row['start']:.2f}s - {row['end']:.2f}s] conf={row['confidence']:.3f}")
        print(f"    {row['text']}")
    if output_path is not None:
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"[saved] {output_path}")


def test_target_speaker(audio_path: Path, ref_audio: Path, segment: tuple[float, float] | None,
                        model_dir: Path, device: str, dtype: str,
                        output_path: Path | None) -> None:
    """Run ``CocktailASRTranscriber.transcribe_segments`` with an explicit ref."""
    sys.path.insert(0, str(REPO_ROOT / "assets" / "voice"))
    from cocktail_asr import CocktailASRTranscriber

    if segment is None:
        # Default: the whole audio, single segment.  transcribe_segments
        # already does the ref+silence+target concatenation, so the
        # resulting row covers exactly the audio length the user passed in.
        waveform, _ = _load_audio(audio_path)
        segs = [(0.0, len(waveform) / SAMPLE_RATE)]
    else:
        segs = [segment]
    print("\n--- mode: target-speaker (explicit reference voiceprint) ---")
    print(f"[model] {model_dir} (device={device}, dtype={dtype})")
    print(f"[ref]   {ref_audio}")
    print(f"[segs]  {segs}")
    transcriber = CocktailASRTranscriber(model_dir=str(model_dir), device=device, dtype=dtype)
    started = time.perf_counter()
    rows = transcriber.transcribe_segments(
        str(audio_path), segs, ref_audio=str(ref_audio))
    elapsed = time.perf_counter() - started
    print(f"[result] {len(rows)} row(s) in {elapsed:.2f}s")
    for row in rows:
        print(f"  [{row['start']:.2f}s - {row['end']:.2f}s] conf={row['confidence']:.3f}")
        print(f"    {row['text']}")
    if output_path is not None:
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"[saved] {output_path}")


def test_via_registry_builder(audio_path: Path, model_dir: Path, device: str,
                              compute_type: str, ref_audio: Path | None,
                              segments_json: Path | None,
                              output_path: Path | None) -> None:
    """Confirm the ``speaker_registry_builder.Transcriber`` shim still works."""
    sys.path.insert(0, str(REPO_ROOT / "assets" / "voice"))
    from speaker_registry_builder import Transcriber

    print("\n--- mode: speaker_registry_builder.Transcriber (pipeline shim) ---")
    print(f"[model] {model_dir} (device={device}, compute_type={compute_type})")
    if ref_audio is not None:
        print(f"[ref]   {ref_audio}")
    if segments_json is not None:
        print(f"[segs]  {segments_json}")
    transcriber = Transcriber(Path(model_dir), device=device, compute_type=compute_type,
                              ref_audio=ref_audio, asr_segments_json=segments_json)
    started = time.perf_counter()
    rows = transcriber(str(audio_path), language=None)
    elapsed = time.perf_counter() - started
    print(f"[result] {len(rows)} row(s) in {elapsed:.2f}s")
    for row in rows:
        print(f"  [{row['start']:.2f}s - {row['end']:.2f}s] conf={row['confidence']:.3f}")
        print(f"    {row['text']}")
    if output_path is not None:
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"[saved] {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test Xiaomi-CocktailASR-1 against a local audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path,
                   help="Path to a wav/flac/mp3 file (auto-resampled to 16 kHz mono).")
    p.add_argument("--ref", type=Path,
                   help="Reference voiceprint (16 kHz mono).  When provided, runs the "
                        "target-speaker mode in addition to the auto-reference mode.")
    p.add_argument("--segment", nargs=2, type=float, metavar=("START", "END"),
                   help="Optional [start, end] window in seconds for the target-speaker mode. "
                        "Defaults to the whole audio.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help=f"Path to the Xiaomi-CocktailASR-1 weights.  Default: {DEFAULT_MODEL_DIR}")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16",
                   help="Torch dtype for the model.  bfloat16 is fastest on Ampere+.")
    p.add_argument("--compute-type", default="float16",
                   help="Same kwarg as the old faster-whisper path; mapped to a torch dtype "
                        "in the shim.  Default: float16.")
    p.add_argument("--segments-json", type=Path,
                   help="Optional JSON file with [{start, end}, ...] rows.  Used in "
                        "--via registry mode to drive per-segment transcription.")
    p.add_argument("--output", type=Path,
                   help="Optional path to dump the ASR rows of the most recent run as JSON.")
    p.add_argument("--via", choices=["auto", "registry"], default="auto",
                   help="auto   = call CocktailASRTranscriber directly (default); "
                        "registry = call through the speaker_registry_builder.Transcriber "
                        "shim so you can confirm the pipeline integration is intact.")
    p.add_argument("--skip-auto", action="store_true",
                   help="Skip the always-on auto-reference run.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio not found: {args.audio}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model dir not found: {args.model_dir}  "
                         f"-- did the download finish?  ls {args.model_dir}")

    _print_audio_summary(args.audio)

    if not args.skip_auto:
        test_auto_reference(args.audio, args.model_dir, args.device, args.dtype,
                            output_path=args.output if args.via == "auto" else None)

    if args.ref is not None:
        if not args.ref.is_file():
            raise SystemExit(f"ref audio not found: {args.ref}")
        seg = tuple(args.segment) if args.segment is not None else None
        test_target_speaker(args.audio, args.ref, seg, args.model_dir, args.device, args.dtype,
                            output_path=None)

    if args.via == "registry":
        test_via_registry_builder(args.audio, args.model_dir, args.device, args.compute_type,
                                  ref_audio=args.ref, segments_json=args.segments_json,
                                  output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
