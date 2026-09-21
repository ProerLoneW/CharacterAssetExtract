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

* **Multi-speaker transcription**
  (``transcribe_multi_speaker(audio, references)``) approximates the
  multi-talker mode the model does not ship with: one shared energy-gated
  time axis, one ``transcribe_segments`` pass per reference voiceprint,
  empty-text spans dropped (the target was absent -- CocktailASR-1's natural
  negative-sample rejection), and cross-speaker duplicates collapsed by
  time overlap plus a text-prefix match.

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


def energy_segments(audio: np.ndarray,
                    min_sec: float = 0.5,
                    max_sec: float = 30.0,
                    gap_merge: float = 0.3,
                    sample_rate: int = SAMPLE_RATE) -> list[list[float]]:
    """Split ``audio`` into ``[start, end]`` speech candidates on frame energy.

    Same 50 ms-window RMS measurement as :func:`pick_auto_reference`, but
    instead of returning the single loudest run it returns *every* run whose
    frames stay above an energy threshold.  The threshold sits 10 % of the
    way from the noise floor (20th-percentile frame RMS) to the speech peak
    (95th percentile), which keeps the gate tolerant of recording levels
    while still rejecting background hiss.  Runs separated by less than
    ``gap_merge`` seconds are bridged so a breath does not split a turn,
    runs shorter than ``min_sec`` are dropped as clicks, and runs longer
    than ``max_sec`` are split into even chunks so a single decode never
    exceeds a comfortable span.

    The result is speaker-agnostic: every reference voiceprint scored in
    ``transcribe_multi_speaker`` is decoded against these same windows,
    which is what makes the later IoU-based cross-speaker dedup meaningful.
    """
    if len(audio) == 0:
        return []
    window = int(0.05 * sample_rate)
    if window <= 0 or len(audio) < window:
        return [[0.0, round(len(audio) / sample_rate, 3)]]
    n_frames = len(audio) // window
    frames = audio[: n_frames * window].reshape(n_frames, window)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    noise_floor = float(np.percentile(rms, 20))
    speech_peak = float(np.percentile(rms, 95))
    if speech_peak <= 1e-4:
        # Digital silence (or a numerically flat waveform): nothing to gate.
        return []
    runs: list[list[float]]
    if speech_peak - noise_floor <= 1e-6:
        # Flat but non-silent energy: treat the whole clip as one candidate.
        runs = [[0.0, n_frames * window / sample_rate]]
    else:
        threshold = noise_floor + 0.1 * (speech_peak - noise_floor)
        active = rms > threshold
        runs = []
        start_frame: Optional[int] = None
        for i, is_active in enumerate(active):
            if is_active and start_frame is None:
                start_frame = i
            elif not is_active and start_frame is not None:
                runs.append([start_frame * window / sample_rate,
                             i * window / sample_rate])
                start_frame = None
        if start_frame is not None:
            runs.append([start_frame * window / sample_rate,
                         n_frames * window / sample_rate])
    # Bridge short pauses so one breath does not split a turn.
    merged: list[list[float]] = []
    for seg in runs:
        if merged and seg[0] - merged[-1][1] <= gap_merge:
            merged[-1][1] = seg[1]
        else:
            merged.append([seg[0], seg[1]])
    # Drop clicks and cap the per-decode span.
    result: list[list[float]] = []
    for start, end in merged:
        if end - start < min_sec:
            continue
        n_chunks = max(1, int(math.ceil((end - start) / max_sec)))
        chunk_len = (end - start) / n_chunks
        for k in range(n_chunks):
            result.append([round(start + k * chunk_len, 3),
                           round(start + (k + 1) * chunk_len, 3)])
    return result


def _segment_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``[start, end]`` spans in seconds."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


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

    def transcribe_multi_speaker(self,
                                 audio: Union[str, Path, np.ndarray],
                                 references: Sequence[dict[str, Any]],
                                 min_sec: float = 0.5,
                                 max_sec: float = 30.0,
                                 gap_merge: float = 0.3,
                                 dedup_iou: float = 0.7,
                                 dedup_prefix_chars: int = 20
                                 ) -> dict[str, Any]:
        """Transcribe several target speakers with one model pass per reference.

        CocktailASR-1 has no native multi-speaker mode: one reference
        voiceprint buys one target speaker, and an absent target yields an
        empty string (a natural negative sample).  The multi-speaker
        timeline is therefore assembled in three steps:

        1. One shared, speaker-agnostic time axis is cut once with
           :func:`energy_segments` (the 50 ms frame-RMS gate behind
           ``pick_auto_reference``) and reused for every reference, so rows
           from different speakers land on comparable spans.
        2. ``transcribe_segments`` runs once per ``references`` entry over
           that axis.  Empty-text rows are dropped -- they mean the target
           was not speaking in that span -- and surviving rows are tagged
           with ``speaker_id`` and ``matched_ref``.
        3. Rows are sorted by start, renumbered ``ASR#####`` across all
           speakers, and cross-speaker duplicates (IoU over the time span
           greater than ``dedup_iou`` *and* identical first
           ``dedup_prefix_chars`` characters of text) are collapsed to the
           row with the longer text, the loser recorded in
           ``dropped_duplicates``.

        ``references`` is a list of ``{"speaker_id": "S1", "ref_audio":
        "path.wav"}`` dicts.  Returns ``{"segments": rows,
        "references_used": [...], "dropped_duplicates": [...],
        "per_reference_segment_count": {...}}``; the per-reference counts
        are taken *before* dedup, i.e. how many spans each voiceprint
        claimed.
        """
        if not references:
            raise ValueError("references must contain at least one entry")
        waveform = _load_audio(audio)
        segments = energy_segments(waveform, min_sec=min_sec, max_sec=max_sec,
                                   gap_merge=gap_merge)
        references_used: list[dict[str, Any]] = []
        per_reference_segment_count: dict[str, int] = {}
        candidates: list[dict[str, Any]] = []
        for i, entry in enumerate(references):
            ref_audio = entry.get("ref_audio")
            if ref_audio is None:
                raise ValueError(f"reference entry {i} is missing 'ref_audio'")
            speaker_id = str(entry.get("speaker_id") or f"SPK{i + 1:02d}")
            matched_ref = str(ref_audio) if isinstance(ref_audio, (str, Path)) \
                else f"<ref[{i}]>"
            rows = self.transcribe_segments(waveform, segments,
                                            ref_audio=ref_audio)
            kept_rows = [row for row in rows if row["text"]]
            for row in kept_rows:
                row["speaker_id"] = speaker_id
                row["matched_ref"] = matched_ref
            candidates.extend(kept_rows)
            references_used.append({"speaker_id": speaker_id,
                                    "ref_audio": matched_ref})
            per_reference_segment_count[speaker_id] = (
                per_reference_segment_count.get(speaker_id, 0) + len(kept_rows))
        candidates.sort(key=lambda row: (row["start"], row["speaker_id"]))
        kept: list[dict[str, Any]] = []
        dropped_duplicates: list[dict[str, Any]] = []
        for row in candidates:
            winner = None
            for other in kept:
                if other["speaker_id"] == row["speaker_id"]:
                    continue
                if _segment_iou((other["start"], other["end"]),
                                (row["start"], row["end"])) <= dedup_iou:
                    continue
                if (other["text"][:dedup_prefix_chars]
                        != row["text"][:dedup_prefix_chars]):
                    continue
                winner = other
                break
            if winner is None:
                kept.append(row)
            elif len(row["text"]) > len(winner["text"]):
                # The newcomer has the longer text: evict the incumbent.
                kept.remove(winner)
                dropped_duplicates.append(
                    {**winner, "superseded_by": row["speaker_id"]})
                kept.append(row)
            else:
                dropped_duplicates.append(
                    {**row, "superseded_by": winner["speaker_id"]})
        kept.sort(key=lambda row: row["start"])
        for i, row in enumerate(kept):
            row["asr_id"] = f"ASR{i + 1:05d}"
        return {
            "segments": kept,
            "references_used": references_used,
            "dropped_duplicates": dropped_duplicates,
            "per_reference_segment_count": per_reference_segment_count,
        }


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
    p.add_argument("--ref", action="append", metavar="REF",
                   help="Reference voiceprint wav.  A plain path keeps the legacy "
                        "single-reference mode (one whole-audio decode against that "
                        "voiceprint); the SPEAKER_ID=path.wav form, repeatable, adds "
                        "one multi-speaker reference per flag instead.  When omitted "
                        "entirely, the loudest 1-4 s clip in --audio is used as the "
                        "reference.")
    p.add_argument("--ref-list", metavar="JSON",
                   help="Path to a JSON array of {\"speaker_id\", \"ref_audio\"} rows; "
                        "switches to multi-speaker mode and is merged with any "
                        "--ref SPEAKER_ID=path.wav entries.  Use this when a path "
                        "itself contains '='.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cot", action="store_true",
                   help="Enable chain-of-thought decoding (debugging only; prints "
                        "<think>...</think><answer>...</answer> instead of plain text).")
    p.add_argument("--output",
                   help="Optional path to dump the segment list as JSON.")
    return p.parse_args()


def _collect_references(args: "argparse.Namespace"
                        ) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]:
    """Split ``--ref`` / ``--ref-list`` into the legacy and multi-speaker modes.

    Returns ``(single_ref, multi_refs)`` where exactly one element is set.
    ``single_ref`` is the pre-multi-speaker CLI contract (one plain ``--ref
    path.wav``, no ``--ref-list``) so ``main`` keeps its old behavior; a
    ``SPEAKER_ID=path.wav`` entry or a ``--ref-list`` file (or several plain
    ``--ref`` paths) switch to ``multi_refs`` instead, with any plain paths
    folded in under auto-assigned ``SPK##`` ids.
    """
    plain: list[str] = []
    entries: list[dict[str, Any]] = []
    for raw in args.ref or []:
        if "=" not in raw:
            plain.append(raw)
            continue
        speaker_id, _, path = raw.partition("=")
        speaker_id, path = speaker_id.strip(), path.strip()
        if not speaker_id or not path:
            raise SystemExit(f"--ref expects SPEAKER_ID=path.wav, got: {raw}")
        entries.append({"speaker_id": speaker_id, "ref_audio": path})
    if args.ref_list is not None:
        import json
        payload = json.loads(Path(args.ref_list).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"--ref-list expects a JSON array: {args.ref_list}")
        for i, item in enumerate(payload):
            if not isinstance(item, dict) or not item.get("ref_audio"):
                raise SystemExit(f"--ref-list entry {i} needs a 'ref_audio' field: "
                                 f"{item!r}")
            speaker_id = str(item.get("speaker_id") or f"SPK{i + 1:02d}")
            entries.append({"speaker_id": speaker_id,
                            "ref_audio": str(item["ref_audio"])})
    if not entries and len(plain) <= 1:
        return (plain[0] if plain else None), None
    # Multi-speaker run: fold plain legacy paths in with auto speaker ids.
    used = {entry["speaker_id"] for entry in entries}
    n = 1
    for path in plain:
        while f"SPK{n:02d}" in used:
            n += 1
        sid = f"SPK{n:02d}"
        used.add(sid)
        entries.append({"speaker_id": sid, "ref_audio": path})
    return None, entries


def _run_multi_speaker(transcriber: CocktailASRTranscriber,
                       args: "argparse.Namespace",
                       references: list[dict[str, Any]]) -> int:
    """CLI front-end for ``transcribe_multi_speaker`` (one pass per reference)."""
    import json
    payload = transcriber.transcribe_multi_speaker(args.audio, references)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    for ref in payload["references_used"]:
        count = payload["per_reference_segment_count"].get(ref["speaker_id"], 0)
        print(f"[ref] {ref['speaker_id']}: {ref['ref_audio']} -> {count} segment(s)")
    print(f"[dedup] dropped {len(payload['dropped_duplicates'])} "
          f"cross-speaker duplicate(s)")
    for row in payload["segments"]:
        print(f"[{row['start']:.2f}s - {row['end']:.2f}s] {row['speaker_id']}: "
              f"{row['text']}")
    return 0


def main() -> int:
    args = _parse_args()
    transcriber = CocktailASRTranscriber(
        model_dir=args.model_dir, device=args.device, dtype=args.dtype, cot=args.cot)
    single_ref, multi_refs = _collect_references(args)
    if multi_refs is not None:
        return _run_multi_speaker(transcriber, args, multi_refs)
    if single_ref is not None:
        result = transcriber.transcribe_segments(
            args.audio, [(0.0, float("inf"))], ref_audio=single_ref)
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
