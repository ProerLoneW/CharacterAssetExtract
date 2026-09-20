#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper for Xiaomi-CocktailASR-1 that produces a Whisper-compatible timeline.

CocktailASR-1 is a *target-speaker* ASR: it takes a reference voice clip and a
target audio, and transcribes only the speaker whose voice matches the
reference.  Calling ``model(target.wav, ref.wav)`` returns a single string,
not word- or segment-level timestamps.

The pipeline (``speaker_registry_builder.Transcriber``) needs a list of
``{"asr_id", "start", "end", "text", "confidence"}`` rows so it can split
speech windows, score reference candidates, and decide lexicality.  This
wrapper bridges the two:

* **Per-segment transcription** with an external reference
  (``transcribe_segments(audio, segments, ref_audio)``) hands a known good
  voiceprint to the model and reuses the existing VAD/diarization output as
  the time axis.  This is the mode used by the voice asset pipeline, where
  each Stage5 turn already has a single dominant speaker.

* **Auto-transcription** without an external reference
  (``transcribe(audio_path, language=None)``) bootstraps a reference from
  the loudest 1-4 second span of the audio itself, so the same model can
  still produce a usable timeline when no voiceprint is supplied.  The
  resulting timeline is the loudest speaker's transcription -- not a full
  multi-speaker ASR, but a stop-gap that lets the existing pipeline keep
  using ``asr_timeline.json`` for window splitting and lexicality scoring.

The confidence score is approximated: CocktailASR-1 has no per-segment logprob,
so we report ``1.0`` when the output is non-empty and ``0.0`` when the model
rejected the reference (negative-sample empty string).  This is enough for the
``has_lexical_speech`` gate to work, which only needs a binary "is there text".

Loading is lazy: ``CocktailASRTranscriber`` only touches the GPU the first
time ``transcribe`` / ``transcribe_segments`` is called, so importing this
module never requires the model to be on disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = "local_models/Xiaomi-CocktailASR-1"
REF_MIN_SECONDS = 1.0
REF_MAX_SECONDS = 4.0
SILENCE_SECONDS = 1.0


def _load_audio(audio: Union[str, Path, np.ndarray, torch.Tensor],
                target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio to 1D float32 mono at ``target_sr``."""
    if isinstance(audio, (str, Path)):
        waveform, sr = sf.read(str(audio), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            import torchaudio
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sr, target_sr).numpy()
    elif isinstance(audio, np.ndarray):
        waveform = audio.astype(np.float32, copy=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    elif isinstance(audio, torch.Tensor):
        waveform = audio.detach().to(torch.float32).cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
    else:
        raise TypeError(f"unsupported audio type: {type(audio).__name__}")
    return waveform


def _peak_normalize(waveform: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak == 0.0 or peak == target_peak:
        return waveform
    return (waveform * (target_peak / peak)).astype(np.float32, copy=False)


def _slice(waveform: np.ndarray, start: float, end: float,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    left = max(0, int(round(start * sample_rate)))
    right = min(len(waveform), int(round(end * sample_rate)))
    if right <= left:
        return np.zeros(0, dtype=np.float32)
    return waveform[left:right]


def pick_auto_reference(audio: np.ndarray,
                        min_seconds: float = REF_MIN_SECONDS,
                        max_seconds: float = REF_MAX_SECONDS,
                        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pick a 1-4 second voiceprint clip from the loudest run of speech.

    The wrapper is bootstrapping a reference from the audio itself, so the
    most representative clip is the one with the highest sustained energy.
    """
    if len(audio) == 0:
        return audio
    window = int(0.05 * sample_rate)
    if window <= 0 or len(audio) < window:
        return audio.copy()
    n_frames = len(audio) // window
    if n_frames == 0:
        return audio.copy()
    frames = audio[: n_frames * window].reshape(n_frames, window)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    target_len = int(min(max_seconds, max(min_seconds, 0.5)) * sample_rate)
    if target_len > len(audio):
        return audio.copy()
    # Convolve a rectangular window of length target_len/window over the RMS to
    # find the highest-sustained-energy window.  The padding at the ends picks
    # up the natural boundary.
    n_target = max(1, target_len // window)
    kernel = np.ones(n_target, dtype=np.float64)
    smoothed = np.convolve(rms, kernel, mode="full")[: len(rms)]
    start_frame = int(np.argmax(smoothed))
    start_sample = start_frame * window
    return audio[start_sample: start_sample + target_len].astype(np.float32, copy=False)


def _prepare_waveform(target: np.ndarray, ref: np.ndarray,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Reproduce ``MicAsrFeatureExtractor.prepare_audio`` for the wav path."""
    target = _peak_normalize(target.astype(np.float32, copy=False))
    ref = ref.astype(np.float32, copy=False)
    if ref.shape[0] < int(REF_MIN_SECONDS * sample_rate):
        pad = int(REF_MIN_SECONDS * sample_rate) - ref.shape[0]
        ref = np.concatenate([np.zeros(pad, dtype=np.float32), ref])
    elif ref.shape[0] > int(REF_MAX_SECONDS * sample_rate):
        ref = ref[: int(REF_MAX_SECONDS * sample_rate)]
    silence = np.zeros(int(SILENCE_SECONDS * sample_rate), dtype=np.float32)
    return np.concatenate([ref, silence, target])


def _concat_segments(audio: np.ndarray, segments: Sequence[Sequence[float]],
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Concatenate the audio inside ``segments`` (each ``[start, end]``) into
    a single 1D waveform.  Used to test whether CocktailASR-1 can transcribe
    short isolated clips; for the standard pipeline we run per-segment, not
    here, but the helper is convenient for the test script.
    """
    pieces = []
    for seg in segments:
        piece = _slice(audio, float(seg[0]), float(seg[1]), sample_rate)
        if len(piece):
            pieces.append(piece)
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32, copy=False)


@dataclass
class SegmentResult:
    asr_id: str
    start: float
    end: float
    text: str
    confidence: float


class CocktailASRTranscriber:
    """Drop-in replacement for ``speaker_registry_builder.Transcriber``.

    Returns ``list[dict]`` rows compatible with the existing pipeline
    (``{"asr_id", "start", "end", "text", "confidence"}``).
    """

    def __init__(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 cot: bool = False,
                 max_new_tokens: Optional[int] = None) -> None:
        self.model_dir = str(model_dir)
        self.device = device
        self.dtype = dtype
        self.cot = cot
        self.max_new_tokens = max_new_tokens
        self._model: Optional[Any] = None
        self._torch_dtype: Optional[torch.dtype] = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        from transformers import AutoModel
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        self._torch_dtype = torch_dtype
        model = AutoModel.from_pretrained(
            self.model_dir, trust_remote_code=True, torch_dtype=torch_dtype
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        model = model.to(self.device).eval()
        self._model = model
        return model

    def _decode(self, target: np.ndarray, ref: np.ndarray) -> str:
        model = self._ensure_loaded()
        waveform = _prepare_waveform(target, ref)
        # ``model(...)`` accepts file paths, numpy arrays, torch tensors; it
        # runs ``_load`` (which handles resampling) + ``prepare_audio``
        # internally.  We pre-prepared the waveform, so pass a tensor and
        # the model will re-load and re-prepare.  That double-handling is
        # cheap (16k mono), and re-using the public API keeps the wrapper
        # immune to internal refactorings of ``MicAsrModel``.
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
        # Use a short silent reference so the model's ref+silence+target
        # layout still works -- we already concatenated everything.
        silence_ref = np.zeros(int(REF_MIN_SECONDS * SAMPLE_RATE), dtype=np.float32)
        result = model(
            waveform_tensor,
            silence_ref,
            cot=self.cot,
            max_new_tokens=self.max_new_tokens,
        )
        text = result if isinstance(result, str) else str(result)
        return _strip_cot_wrappers(text)

    def transcribe(self, audio: Union[str, Path, np.ndarray],
                   language: Optional[str] = None) -> list[dict[str, Any]]:
        """Transcribe ``audio`` using a self-bootstrapped reference.

        Returns a list with a single row spanning the full audio.  This is
        the stand-in for a general ASR; the row covers the entire audio
        duration so ``build_speech_windows`` keeps all VAD windows.  Use
        ``transcribe_segments`` when you have a real reference and need
        per-turn results.
        """
        waveform = _load_audio(audio)
        if len(waveform) == 0:
            return []
        ref = pick_auto_reference(waveform)
        text = self._decode(waveform, ref).strip()
        duration = len(waveform) / SAMPLE_RATE
        if not text:
            return []
        return [{
            "asr_id": "ASR00001",
            "start": 0.0,
            "end": round(duration, 3),
            "text": text,
            "confidence": 1.0,
        }]

    def transcribe_segments(self, audio: Union[str, Path, np.ndarray],
                            segments: Iterable[Sequence[float]],
                            ref_audio: Optional[Union[str, Path, np.ndarray]] = None,
                            language: Optional[str] = None,
                            merge_gap: float = 0.3) -> list[dict[str, Any]]:
        """Transcribe each ``[start, end]`` segment with a shared reference.

        ``ref_audio`` is the voiceprint clip.  When omitted the loudest run
        in the audio itself is used, the same way ``transcribe`` does it.
        Consecutive non-empty segments with a gap smaller than
        ``merge_gap`` are concatenated into one row -- the model is a string-
        in / string-out system and emitting one row per window is what the
        downstream timeline builder expects.
        """
        waveform = _load_audio(audio)
        if len(waveform) == 0:
            return []
        if ref_audio is None:
            ref = pick_auto_reference(waveform)
        else:
            ref = _load_audio(ref_audio)
            if len(ref):
                ref = _peak_normalize(ref)
        seg_list = [(float(seg[0]), float(seg[1])) for seg in segments
                    if float(seg[1]) > float(seg[0])]
        seg_list.sort()
        rows: list[dict[str, Any]] = []
        counter = 0
        i = 0
        while i < len(seg_list):
            seg_start, seg_end = seg_list[i]
            piece = _slice(waveform, seg_start, seg_end)
            if len(piece) < int(0.1 * SAMPLE_RATE):
                i += 1
                continue
            # Greedily extend into the next segment if the gap is small.
            cursor_end = seg_end
            j = i + 1
            while j < len(seg_list):
                nxt_start, nxt_end = seg_list[j]
                if nxt_start - cursor_end > merge_gap:
                    break
                extra = _slice(waveform, cursor_end, nxt_end)
                if extra.size:
                    piece = np.concatenate([piece, extra])
                cursor_end = nxt_end
                j += 1
            text = self._decode(piece, ref).strip()
            counter += 1
            confidence = 1.0 if text else 0.0
            rows.append({
                "asr_id": f"ASR{counter:05d}",
                "start": round(seg_start, 3),
                "end": round(cursor_end, 3),
                "text": text,
                "confidence": confidence,
            })
            i = j
        return rows


def _strip_cot_wrappers(text: str) -> str:
    """CocktailASR-1's CoT mode wraps the answer in <think>...</think><answer>...</answer>."""
    if "<think>" in text and "</think>" in text:
        end = text.find("</think>")
        text = text[end + len("</think>"):]
    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        text = text[start:end]
    return text.strip()


# --------------------------------------------------------------------------- CLI

def _parse_args() -> "argparse.Namespace":
    import argparse
    p = argparse.ArgumentParser(
        description="Run Xiaomi-CocktailASR-1 on a single audio and print the text."
    )
    p.add_argument("--audio", required=True,
                   help="Path to a wav/flac/mp3 file (auto-resampled to 16 kHz mono).")
    p.add_argument("--ref",
                   help="Optional reference voiceprint wav.  When omitted, the loudest "
                        "1-4 s clip in --audio is used as the reference.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cot", action="store_true",
                   help="Enable chain-of-thought decoding (debugging only; prints "
                        "<think>...</think><answer>...</answer> instead of plain text).")
    p.add_argument("--output",
                   help="Optional path to dump the segment list as JSON.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    transcriber = CocktailASRTranscriber(
        model_dir=args.model_dir, device=args.device, dtype=args.dtype, cot=args.cot)
    if args.ref is not None:
        result = transcriber.transcribe_segments(
            args.audio, [(0.0, float("inf"))], ref_audio=args.ref)
    else:
        result = transcriber.transcribe(args.audio)
    if args.output:
        import json
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for row in result:
        print(f"[{row['start']:.2f}s - {row['end']:.2f}s] {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
