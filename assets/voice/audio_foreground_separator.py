#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a video/audio track into foreground speech and background residual.

Migrated from teleAI-T2V ``new_data_ppl_v1.0/video_audio_seperation.py``:
ffmpeg extract -> ClearVoice speech enhancement -> ``background = origin - foreground``.

Standalone CLI, and callable from Stage 5 through :func:`split_foreground_background`
plus :func:`foreground_gate_for_turn`, which supplies the ``music_contamination`` value
that ``speaker_registry_builder.turn_quality`` currently hardcodes to -1.0.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf

SCHEMA_VERSION = "audio-foreground-v1"
REPO_ROOT = Path(__file__).resolve().parent
# 迁移到 assets/voice/ 后工作区根（wyn/）在上四级；ClearVoice 仍引用 teleAI-T2V 内的副本。
DEFAULT_CLEARVOICE_ROOT = REPO_ROOT.parents[3] / "teleAI-T2V" / "new_data_ppl_v1.0"
DEFAULT_MODEL = "MossFormer2_SE_48K"
MODEL_SAMPLE_RATE = 48000
STAGE5_SAMPLE_RATE = 16000

LOGGER = logging.getLogger("audio_foreground")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _clearvoice_cwd(root: Path):
    # ClearVoice builds its config path as the relative string
    # 'others/clearvoice/config/inference/<model>.yaml', so cwd must be the pipeline root.
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def load_enhancer(clearvoice_root: Path, model_name: str = DEFAULT_MODEL):
    """Load the ClearVoice speech-enhancement model, returning ``(model, resolved_root)``."""
    root = Path(clearvoice_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ClearVoice root not found: {root}")
    vendor = root / "others" / "clearvoice"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    with _clearvoice_cwd(root):
        from clearvoice import ClearVoice

        return ClearVoice(task="speech_enhancement", model_names=[model_name]), root


def extract_audio(source: Path, destination: Path, sample_rate: int, channels: int,
                  ffmpeg_bin: str) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-vn", "-ac", str(channels), "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(destination)],
        check=True,
    )
    info = sf.info(str(destination))
    return {"path": str(destination), "sample_rate": info.samplerate, "channels": info.channels,
            "frames": int(info.frames), "duration": round(info.duration, 6)}


def resample_to(source: Path, destination: Path, sample_rate: int, ffmpeg_bin: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(destination)],
        check=True,
    )
    return destination


def to_mono(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, np.float32)
    return waveform.mean(axis=1) if waveform.ndim > 1 else waveform


def activity_intervals(waveform: np.ndarray, sample_rate: int, *, frame_ms: float = 32.0,
                       hop_ms: float = 10.0, min_dbfs: float = -50.0,
                       dynamic_range_db: float = 35.0, merge_gap_seconds: float = 0.20,
                       min_segment_seconds: float = 0.15) -> List[Tuple[float, float]]:
    mono = to_mono(waveform)
    total = len(mono) / float(sample_rate)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if total <= 0.0:
        return []
    offsets = np.arange(0, max(1, len(mono) - frame + 1), hop)
    rms = np.array([np.sqrt(np.mean(mono[i:i + frame] ** 2) + 1e-12) for i in offsets])
    db = 20.0 * np.log10(rms + 1e-12)
    threshold = max(min_dbfs, float(db.max()) - dynamic_range_db)
    active = db >= threshold

    raw: List[List[float]] = []
    for index, flag in enumerate(active):
        start = offsets[index] / float(sample_rate)
        end = min(total, (offsets[index] + frame) / float(sample_rate))
        if not flag:
            continue
        if raw and start - raw[-1][1] <= merge_gap_seconds:
            raw[-1][1] = end
        else:
            raw.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in raw
            if b - a >= min_segment_seconds]


def subtract_intervals(spans: Sequence[Tuple[float, float]],
                       holes: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return ``spans`` minus ``holes``, dropping empty fragments."""
    output: List[Tuple[float, float]] = []
    for start, end in spans:
        pending = [(start, end)]
        for hole_start, hole_end in holes:
            nxt: List[Tuple[float, float]] = []
            for a, b in pending:
                if hole_end <= a or hole_start >= b:
                    nxt.append((a, b))
                    continue
                if hole_start > a:
                    nxt.append((a, min(b, hole_start)))
                if hole_end < b:
                    nxt.append((max(a, hole_end), b))
            pending = nxt
        output.extend((round(a, 3), round(b, 3)) for a, b in pending if b - a > 1e-3)
    return sorted(output)


def _rms_db(values: np.ndarray) -> float:
    return round(float(20.0 * np.log10(np.sqrt(np.mean(values ** 2) + 1e-12))), 3)


def _region_metrics(origin: np.ndarray, foreground: np.ndarray, background: np.ndarray,
                    sample_rate: int, start: float, end: float) -> Dict[str, Any] | None:
    left = max(0, int(start * sample_rate))
    right = min(len(origin), int(end * sample_rate))
    if right <= left:
        return None
    o, f, b = origin[left:right], foreground[left:right], background[left:right]
    energy_o = float(np.sum(o ** 2))
    energy_f = float(np.sum(f ** 2))
    energy_b = float(np.sum(b ** 2))
    split_total = energy_f + energy_b
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "rms_db": _rms_db(o),
        "foreground_rms_db": _rms_db(f),
        "background_rms_db": _rms_db(b),
        "foreground_energy_ratio": round(energy_f / split_total, 6) if split_total else None,
        # 0.0 = clean speech, 1.0 = nothing but music/SFX residual.
        "music_contamination": round(energy_b / split_total, 6) if split_total else None,
        "energy_closure": round(split_total / energy_o, 6) if energy_o else None,
    }


def subtract_background(origin: np.ndarray, foreground: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    origin = np.asarray(origin, np.float32)
    foreground = np.asarray(foreground, np.float32)
    if origin.ndim != foreground.ndim or origin.shape[1:] != foreground.shape[1:]:
        raise ValueError(
            f"Channel layout mismatch: origin {origin.shape} vs foreground {foreground.shape}"
        )
    length = min(len(origin), len(foreground))
    return origin[:length] - foreground[:length], {
        # ClearVoice resamples back to the source rate on write, which can add one sample.
        # Extracting at the model's native 48 kHz keeps this at zero.
        "length_delta_samples": int(len(foreground) - len(origin)),
        "used_samples": int(length),
    }


def split_foreground_background(
    input_path: Path,
    output_dir: Path,
    *,
    clearvoice_root: Path = DEFAULT_CLEARVOICE_ROOT,
    model_name: str = DEFAULT_MODEL,
    ffmpeg_bin: str | None = None,
    extract_sample_rate: int = MODEL_SAMPLE_RATE,
    extract_channels: int = 1,
    stage5_sample_rate: int | None = STAGE5_SAMPLE_RATE,
    frame_ms: float = 32.0,
    hop_ms: float = 10.0,
    min_dbfs: float = -50.0,
    dynamic_range_db: float = 35.0,
    merge_gap_seconds: float = 0.20,
    min_segment_seconds: float = 0.15,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Extract audio, enhance it into a foreground stem, and derive the background residual."""
    started = time.perf_counter()
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input not found: {source}")
    destination = Path(output_dir).expanduser().resolve()
    report_path = destination / "foreground_report.json"
    if report_path.is_file() and not overwrite:
        LOGGER.info("Reusing existing report at %s", report_path)
        return json.loads(report_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(ffmpeg_bin)

    audio_dir = destination / "audio"
    origin_path = audio_dir / f"origin_{extract_sample_rate}.wav"
    foreground_path = audio_dir / f"foreground_{extract_sample_rate}.wav"
    background_path = audio_dir / f"background_{extract_sample_rate}.wav"

    extraction = extract_audio(source, origin_path, extract_sample_rate, extract_channels, ffmpeg)
    sample_rate = extraction["sample_rate"]

    enhancer, resolved_root = load_enhancer(clearvoice_root, model_name)
    with _clearvoice_cwd(resolved_root):
        enhanced = enhancer(input_path=str(origin_path))
        enhancer.write(enhanced, output_path=str(foreground_path))

    origin_wave, _ = sf.read(str(origin_path), dtype="float32", always_2d=True)
    foreground_wave, _ = sf.read(str(foreground_path), dtype="float32", always_2d=True)
    background_wave, length_info = subtract_background(origin_wave, foreground_wave)
    sf.write(str(background_path), background_wave, sample_rate, subtype="PCM_16")

    # Re-read what actually landed on disk: PCM_16 quantisation is the only thing that can
    # break the origin = foreground + background identity.
    origin_disk = to_mono(sf.read(str(origin_path), dtype="float64")[0])
    foreground_disk = to_mono(sf.read(str(foreground_path), dtype="float64")[0])
    background_disk = to_mono(sf.read(str(background_path), dtype="float64")[0])
    identity_length = min(len(origin_disk), len(foreground_disk), len(background_disk))
    residual = (origin_disk[:identity_length]
                - foreground_disk[:identity_length]
                - background_disk[:identity_length])

    energy_o = float(np.sum(origin_disk ** 2))
    energy_f = float(np.sum(foreground_disk ** 2))
    energy_b = float(np.sum(background_disk ** 2))
    split_total = energy_f + energy_b

    def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
        n = min(len(left), len(right))
        if n == 0 or float(np.std(left[:n])) < 1e-12 or float(np.std(right[:n])) < 1e-12:
            return None
        return round(float(np.corrcoef(left[:n], right[:n])[0, 1]), 6)

    speech_spans = activity_intervals(
        foreground_disk, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms, min_dbfs=min_dbfs,
        dynamic_range_db=dynamic_range_db, merge_gap_seconds=merge_gap_seconds,
        min_segment_seconds=min_segment_seconds,
    )
    origin_spans = activity_intervals(
        origin_disk, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms, min_dbfs=min_dbfs,
        dynamic_range_db=dynamic_range_db, merge_gap_seconds=merge_gap_seconds,
        min_segment_seconds=min_segment_seconds,
    )
    background_only_spans = subtract_intervals(origin_spans, speech_spans)

    speech_regions = [metrics for metrics in (
        _region_metrics(origin_disk, foreground_disk, background_disk, sample_rate, a, b)
        for a, b in speech_spans) if metrics]
    background_regions = [metrics for metrics in (
        _region_metrics(origin_disk, foreground_disk, background_disk, sample_rate, a, b)
        for a, b in background_only_spans) if metrics]

    files: Dict[str, str] = {
        "origin": str(origin_path),
        "foreground": str(foreground_path),
        "background": str(background_path),
    }
    if stage5_sample_rate:
        for tag, path in (("origin", origin_path), ("foreground", foreground_path),
                          ("background", background_path)):
            files[f"{tag}_{stage5_sample_rate}"] = str(
                resample_to(path, audio_dir / f"{tag}_{stage5_sample_rate}.wav",
                            stage5_sample_rate, ffmpeg))

    duration = round(len(origin_disk) / float(sample_rate), 6)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "HAS_SPEECH" if speech_regions else "NO_SPEECH",
        "input": str(source),
        "output_dir": str(destination),
        "duration": duration,
        "extraction": extraction,
        "model": {
            "backend": "clearvoice",
            "task": "speech_enhancement",
            "name": model_name,
            "clearvoice_root": str(resolved_root),
        },
        "gate_parameters": {
            "frame_ms": frame_ms, "hop_ms": hop_ms, "min_dbfs": min_dbfs,
            "dynamic_range_db": dynamic_range_db, "merge_gap_seconds": merge_gap_seconds,
            "min_segment_seconds": min_segment_seconds,
        },
        "files": files,
        "subtraction": length_info,
        "identity_check": {
            "samples_compared": int(identity_length),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "residual_max_abs": float(np.max(np.abs(residual))) if identity_length else 0.0,
        },
        "global_metrics": {
            "origin_rms_db": _rms_db(origin_disk),
            "foreground_rms_db": _rms_db(foreground_disk),
            "background_rms_db": _rms_db(background_disk),
            "foreground_energy_ratio": round(energy_f / split_total, 6) if split_total else None,
            "background_energy_ratio": round(energy_b / split_total, 6) if split_total else None,
            "energy_closure": round(split_total / energy_o, 6) if energy_o else None,
            "corr_origin_foreground": _corr(origin_disk, foreground_disk),
            "corr_foreground_background": _corr(foreground_disk, background_disk),
            # background = origin - foreground can leave the PCM_16 range when the stems are
            # anti-phase; clipping there would silently corrupt the residual.
            "origin_peak": round(float(np.max(np.abs(origin_disk))), 6) if len(origin_disk) else None,
            "foreground_peak": round(float(np.max(np.abs(foreground_disk))), 6) if len(foreground_disk) else None,
            "background_peak": round(float(np.max(np.abs(background_disk))), 6) if len(background_disk) else None,
            "clipped_samples": {
                tag: int(np.sum(np.abs(values) >= 0.999999))
                for tag, values in (("origin", origin_disk), ("foreground", foreground_disk),
                                    ("background", background_disk))
            },
        },
        "activity": {
            "foreground_seconds": round(sum(b - a for a, b in speech_spans), 3),
            "origin_seconds": round(sum(b - a for a, b in origin_spans), 3),
            "background_only_seconds": round(sum(b - a for a, b in background_only_spans), 3),
            "foreground_coverage": round(
                sum(b - a for a, b in speech_spans) / duration, 6) if duration else None,
        },
        "speech_regions": speech_regions,
        "background_only_regions": background_regions,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _atomic_json(report_path, report)
    return report


def foreground_gate_for_turn(report: Dict[str, Any], start: float, end: float,
                             min_coverage: float = 0.3) -> Dict[str, Any]:
    """Score one Stage-5 turn against the foreground report.

    Returns the ``music_contamination`` value plus how much of the turn is backed by
    enhanced speech.  ``speaker_turn_foreground_gate`` turns this into tier decisions;
    ``speaker_registry_builder`` uses it directly to gate reference-clip candidates.
    """
    regions = report.get("speech_regions") or []
    total = max(float(end) - float(start), 1e-6)
    covered = 0.0
    weighted = 0.0
    for region in regions:
        overlap = min(float(end), float(region["end"])) - max(float(start), float(region["start"]))
        if overlap <= 0:
            continue
        covered += overlap
        contamination = region.get("music_contamination")
        if contamination is not None:
            weighted += overlap * float(contamination)
    coverage = min(covered / total, 1.0)
    return {
        "music_contamination": round(weighted / covered, 6) if covered else None,
        "foreground_coverage": round(coverage, 6),
        "has_foreground_speech": coverage >= min_coverage,
    }


def gate_turns(report: Dict[str, Any], turns: Sequence[Dict[str, Any]],
               min_coverage: float = 0.3) -> List[Dict[str, Any]]:
    rows = []
    for turn in turns:
        gate = foreground_gate_for_turn(report, turn["start"], turn["end"], min_coverage)
        gate.update({"turn_id": turn.get("turn_id"), "start": turn.get("start"),
                     "end": turn.get("end"), "speaker_id": turn.get("speaker_id")})
        rows.append(gate)
    return rows


def load_turns(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("turns", "speaker_turns"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValueError(f"No 'turns' list in {path}")
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported turns file: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a video/audio track into foreground speech and background residual."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clearvoice-root", type=Path, default=DEFAULT_CLEARVOICE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ffmpeg")
    parser.add_argument(
        "--extract-sample-rate", type=int, default=MODEL_SAMPLE_RATE,
        help="48000 avoids ClearVoice's resample round-trip; teleAI used 44100.",
    )
    parser.add_argument(
        "--extract-channels", type=int, default=1, choices=[1, 2],
        help="The enhancement model is mono either way; teleAI used 2.",
    )
    parser.add_argument("--stage5-sample-rate", type=int, default=STAGE5_SAMPLE_RATE,
                        help="Also emit mono copies at this rate; 0 disables.")
    parser.add_argument("--frame-ms", type=float, default=32.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--min-dbfs", type=float, default=-50.0)
    parser.add_argument("--activity-dynamic-range-db", type=float, default=35.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.20)
    parser.add_argument(
        "--min-segment-seconds", type=float, default=0.15,
        help=("Deliberately shorter than the separator's 0.45: discarding short bursts here "
              "mislabels real interjections as background. Stage 5 enforces turn duration."),
    )
    parser.add_argument("--min-coverage", type=float, default=0.3,
                        help="Foreground coverage required to call a turn speech-backed.")
    parser.add_argument("--turns-json", type=Path,
                        help="Stage-5 turns.json / speaker_turns.json to gate per turn.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    positive = {
        "--extract-sample-rate": args.extract_sample_rate,
        "--frame-ms": args.frame_ms,
        "--hop-ms": args.hop_ms,
        "--min-segment-seconds": args.min_segment_seconds,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"Must be positive: {', '.join(invalid)}")
    if not 0.0 <= args.min_coverage <= 1.0:
        raise SystemExit("--min-coverage must be within [0, 1]")
    try:
        report = split_foreground_background(
            args.input, args.output_dir,
            clearvoice_root=args.clearvoice_root, model_name=args.model,
            ffmpeg_bin=args.ffmpeg, extract_sample_rate=args.extract_sample_rate,
            extract_channels=args.extract_channels,
            stage5_sample_rate=args.stage5_sample_rate or None,
            frame_ms=args.frame_ms, hop_ms=args.hop_ms, min_dbfs=args.min_dbfs,
            dynamic_range_db=args.activity_dynamic_range_db,
            merge_gap_seconds=args.merge_gap_seconds,
            min_segment_seconds=args.min_segment_seconds, overwrite=args.overwrite,
        )
    except Exception:  # noqa: BLE001 - CLI needs one concise failure contract
        LOGGER.exception("Foreground/background split failed")
        return 1

    if args.turns_json:
        rows = gate_turns(report, load_turns(args.turns_json), args.min_coverage)
        _atomic_json(Path(args.output_dir).expanduser().resolve() / "foreground_gate.json",
                     {"schema_version": SCHEMA_VERSION, "turn_count": len(rows), "turns": rows})

    print(json.dumps({
        "status": report["status"],
        "duration": report["duration"],
        "foreground_energy_ratio": report["global_metrics"]["foreground_energy_ratio"],
        "foreground_seconds": report["activity"]["foreground_seconds"],
        "background_only_seconds": report["activity"]["background_only_seconds"],
        "speech_regions": len(report["speech_regions"]),
        "residual_rms": report["identity_check"]["residual_rms"],
        "report": str(Path(args.output_dir).expanduser().resolve() / "foreground_report.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
