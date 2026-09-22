#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export every speaker-attributed ASR interval for human review.

Rows are separated by the existing ``audio_asset_candidate`` decision.  This
tool does not promote preview files into the production character asset bank.

Optionally (``--voice-registry`` / ``--voice-registry-root``) each ASR row is
also linked to the consolidated speaker registry
(``speaker_voice_asset_pipeline`` output) that was built from the same global
clip, matched by resolved ``metadata.pipeline_input`` against the row's
``global_clip_path``.  ASR ``time_range_sec`` and Stage5 turn ``start``/``end``
share the same global-clip-relative timeline, so the link is a pure time
intersection: which consolidated speaker covers the row, how much of the row
any turn covers, how pure that coverage is, and how much of it survives after
subtracting ``simultaneous_speech_regions``.  The verdict per row is
``matched`` / ``ambiguous`` / ``unmatched`` / ``insufficient_evidence`` /
``timebase_mismatch`` / ``no_registry`` and is written, with the metrics, into
``voice_asset_manifest.jsonl`` (schema ``speaker-voice-asset-v2``).  Every
manifest row starts at ``review_status: pending`` -- a preview existing says
nothing about approval; pass ``--approvals`` with a JSONL of
``{"global_id": ..., "asr_id": ...}`` rows to mark reviewed rows ``approved``.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import re
import subprocess
import wave
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def safe_component(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return result.strip("._") or "unknown"


def number_range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain start/end")
    start, end = float(value[0]), float(value[1])
    if start < 0 or end <= start:
        raise ValueError(f"invalid {label}: {value}")
    return start, end


def clip_relative_range(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = float(value[0]), float(value[1])
        if start < 0.0 < end:
            value = [0.0, end]
    return number_range(value, label)


def cut_wav(source: Path, output: Path, start: float, end: float, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.wav")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}",
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(temporary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    temporary.replace(output)


def signal_stats(path: Path, silence_threshold_dbfs: float = -40.0) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        payload = handle.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Expected PCM s16le WAV, got sample width {sample_width}: {path}")
    samples = array.array("h")
    samples.frombytes(payload)
    if not samples:
        return {
            "duration_sec": 0.0, "sample_rate": sample_rate, "channels": channels,
            "mean_volume_dbfs": None, "peak_dbfs": None, "sample_silence_ratio": 1.0,
            "silence_threshold_dbfs": silence_threshold_dbfs,
        }
    peak = max(abs(value) for value in samples)
    rms = math.sqrt(sum(float(value) * value for value in samples) / len(samples))
    full_scale = 32768.0
    threshold = full_scale * (10.0 ** (silence_threshold_dbfs / 20.0))
    return {
        "duration_sec": round(frame_count / sample_rate, 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "mean_volume_dbfs": round(20.0 * math.log10(rms / full_scale), 3) if rms else None,
        "peak_dbfs": round(20.0 * math.log10(peak / full_scale), 3) if peak else None,
        "sample_silence_ratio": round(
            sum(abs(value) <= threshold for value in samples) / len(samples), 6
        ),
        "silence_threshold_dbfs": silence_threshold_dbfs,
    }


VOICE_ASSET_SCHEMA_VERSION = "speaker-voice-asset-v2"
# The ASR timeline and the registry timeline must be the same global clip's seconds; a row
# ending this far past the registry's audio duration signals a different timebase, not noise.
TIMEBASE_TOLERANCE_SECONDS = 0.5


def load_voice_registries(registries: list[Path], roots: list[Path]) -> dict[str, dict[str, Any]]:
    """Index consolidated registries by the resolved clip path they were built from."""
    paths = [path.expanduser().resolve() for path in registries]
    for root in roots:
        base = root.expanduser().resolve()
        paths.extend(sorted(base.glob("**/consolidated/registry.json")))
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        registry = json.loads(path.read_text(encoding="utf-8"))
        pipeline_input = (registry.get("metadata") or {}).get("pipeline_input")
        if not pipeline_input:
            raise ValueError(f"{path}: metadata.pipeline_input is missing; "
                             "is this a consolidated registry?")
        key = str(Path(pipeline_input).expanduser().resolve())
        if key in index:
            index[key].setdefault("__duplicate_registries__", []).append(str(path))
            continue
        registry["__registry_path__"] = str(path)
        index[key] = registry
    return index


def load_approvals(path: Path) -> set[tuple[int, str]]:
    return {(int(row.get("global_id") or 0), str(row.get("asr_id") or ""))
            for row in load_jsonl(path.expanduser().resolve())}


def _overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, hi) - max(start, lo)) for lo, hi in intervals)


def _subtract_regions(spans: list[tuple[float, float]],
                      regions: list[tuple[float, float]]) -> list[list[float]]:
    """``spans`` minus ``regions`` as a list of [start, end] source sub-intervals."""
    result: list[list[float]] = []
    for lo, hi in spans:
        cursor = lo
        for rlo, rhi in sorted(regions):
            if rhi <= cursor or rlo >= hi:
                continue
            if rlo > cursor:
                result.append([round(cursor, 6), round(min(rlo, hi), 6)])
            cursor = max(cursor, rhi)
        if cursor < hi:
            result.append([round(cursor, 6), round(hi, 6)])
    return [pair for pair in result if pair[1] > pair[0]]


def link_voice_row(start: float, end: float, registry: dict[str, Any], *,
                   min_coverage: float, min_purity: float,
                   min_net_seconds: float) -> dict[str, Any]:
    """Decide which consolidated speaker, if any, owns this ASR interval.

    Stage5 diarization is a strict time partition, so per-turn intersections
    never double-count and plain sums are exact.
    """
    link: dict[str, Any] = {"registry_path": registry.get("__registry_path__")}
    duration = float((registry.get("timeline") or {}).get("duration") or 0.0)
    if duration and end > duration + TIMEBASE_TOLERANCE_SECONDS:
        return {**link, "status": "timebase_mismatch", "registry_duration": duration}
    per_speaker: dict[str, float] = {}
    speaker_spans: dict[str, list[tuple[float, float]]] = {}
    unknown_seconds = 0.0
    for turn in registry.get("turns") or []:
        lo, hi = max(start, float(turn["start"])), min(end, float(turn["end"]))
        if hi <= lo:
            continue
        speaker = turn.get("speaker_id")
        if speaker is None:
            unknown_seconds += hi - lo
        else:
            per_speaker[speaker] = per_speaker.get(speaker, 0.0) + hi - lo
            speaker_spans.setdefault(speaker, []).append((lo, hi))
    covered_seconds = sum(per_speaker.values()) + unknown_seconds
    window = end - start
    coverage = covered_seconds / window if window > 0 else 0.0
    base = {**link, "asr_coverage": round(coverage, 6),
            "covered_seconds": round(covered_seconds, 6),
            "unknown_speaker_seconds": round(unknown_seconds, 6),
            "speaker_seconds": {sid: round(sec, 6)
                                for sid, sec in sorted(per_speaker.items())}}
    attributed = sum(per_speaker.values())
    if attributed <= 0.0:
        return {**base, "status": "unmatched"}
    dominant_id, dominant_seconds = max(per_speaker.items(), key=lambda item: item[1])
    purity = dominant_seconds / attributed
    simultaneous = [(float(region["start"]), float(region["end"]))
                    for region in registry.get("simultaneous_speech_regions") or []]
    dominant_overlap = sum(_overlap_seconds(lo, hi, simultaneous)
                           for lo, hi in speaker_spans[dominant_id])
    net_seconds = max(0.0, dominant_seconds - dominant_overlap)
    base.update({
        "speaker_id": dominant_id,
        "spk_purity": round(purity, 6),
        "attributed_seconds": round(attributed, 6),
        "dominant_speaker_seconds": round(dominant_seconds, 6),
        "simultaneous_speech_seconds": round(dominant_overlap, 6),
        "net_single_speaker_seconds": round(net_seconds, 6),
        # Source-time sub-intervals of the dominant speaker with simultaneous speech
        # removed: a delivered asset traces to these segments, never to the whole row.
        "clean_subintervals": _subtract_regions(speaker_spans[dominant_id], simultaneous),
    })
    if purity < min_purity:
        status = "ambiguous"
    elif coverage < min_coverage or net_seconds < min_net_seconds:
        status = "insufficient_evidence"
    else:
        status = "matched"
    base["status"] = status
    if status == "matched":
        speaker_row = next((row for row in registry.get("speakers") or []
                            if row.get("speaker_id") == dominant_id), {})
        base["reference_audio"] = speaker_row.get("reference_audio") or []
        base["pooled_clean_voice"] = speaker_row.get("pooled_clean_voice")
    return base


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut every attributed ASR interval into review WAVs.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Strict speaker_attributed_asr.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--voice-registry", type=Path, action="append", default=[],
                        help="consolidated registry.json to link ASR rows against; repeatable")
    parser.add_argument("--voice-registry-root", type=Path, action="append", default=[],
                        help="directory scanned recursively for **/consolidated/registry.json")
    parser.add_argument("--approvals", type=Path,
                        help="JSONL of {\"global_id\": ..., \"asr_id\": ...} rows that human "
                             "review approved; matching voice manifest rows become approved")
    parser.add_argument("--min-voice-coverage", type=float, default=0.5,
                        help="matched requires this share of the ASR interval covered by turns")
    parser.add_argument("--min-voice-purity", type=float, default=0.6,
                        help="below this dominant-speaker share the row is ambiguous")
    parser.add_argument("--min-voice-net-seconds", type=float, default=2.0,
                        help="matched requires this much net single-speaker speech")
    args = parser.parse_args()

    source_manifest = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_registries = load_voice_registries(args.voice_registry, args.voice_registry_root)
    approved_keys = load_approvals(args.approvals) if args.approvals else set()
    for name in ("min_voice_coverage", "min_voice_purity"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.min_voice_net_seconds < 0.0:
        raise SystemExit("--min-voice-net-seconds must not be negative")
    results = []
    for source_row in load_jsonl(source_manifest):
        row = dict(source_row)
        valid = bool(row.get("audio_asset_candidate"))
        review_status = "valid" if valid else "invalid"
        global_id = int(row.get("global_id") or 0)
        speaker_id = safe_component(row.get("speaker_id"))
        asr_id = safe_component(row.get("asr_id"))
        relative = Path(review_status) / f"global_{global_id:04d}" / speaker_id / f"{asr_id}.wav"
        output = output_dir / relative
        try:
            source = Path(str(row.get("global_clip_path") or "")).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"global clip not found: {source}")
            start, end = clip_relative_range(row.get("time_range_sec"), f"{asr_id}.time_range_sec")
            cut_wav(source, output, start, end, args.ffmpeg)
            row.update({
                "review_status": review_status,
                "review_audio_path": str(output),
                "review_audio_relative_path": str(relative),
                "cut_success": True,
                "signal_stats": signal_stats(output),
            })
        except Exception as exc:  # noqa: BLE001
            row.update({
                "review_status": review_status,
                "review_audio_path": str(output),
                "review_audio_relative_path": str(relative),
                "cut_success": False,
                "cut_error": f"{type(exc).__name__}: {exc}",
            })
        if voice_registries:
            clip = str(Path(str(row.get("global_clip_path") or "")).expanduser().resolve())
            registry = voice_registries.get(clip)
            if registry is None:
                row["voice_link"] = {"status": "no_registry"}
            else:
                try:
                    link_start, link_end = clip_relative_range(
                        row.get("time_range_sec"), f"{asr_id}.time_range_sec")
                    row["voice_link"] = link_voice_row(
                        link_start, link_end, registry,
                        min_coverage=args.min_voice_coverage,
                        min_purity=args.min_voice_purity,
                        min_net_seconds=args.min_voice_net_seconds)
                except Exception as exc:  # noqa: BLE001
                    row["voice_link"] = {"status": "link_error",
                                         "error": f"{type(exc).__name__}: {exc}"}
        results.append(row)

    results.sort(key=lambda row: (
        int(row.get("global_id") or 0),
        float((row.get("time_range_sec") or [0.0])[0]),
        str(row.get("asr_id")),
    ))
    valid_rows = [row for row in results if row["review_status"] == "valid"]
    invalid_rows = [row for row in results if row["review_status"] == "invalid"]
    write_jsonl(output_dir / "all_asr_manifest.jsonl", results)
    write_jsonl(output_dir / "valid_manifest.jsonl", valid_rows)
    write_jsonl(output_dir / "invalid_manifest.jsonl", invalid_rows)
    voice_rows: list[dict[str, Any]] = []
    if voice_registries:
        for row in results:
            link = row.get("voice_link") or {}
            voice_rows.append({
                "schema_version": VOICE_ASSET_SCHEMA_VERSION,
                "asr_id": row.get("asr_id"),
                "global_id": int(row.get("global_id") or 0),
                "asr_speaker_id": row.get("speaker_id"),
                "global_clip_path": row.get("global_clip_path"),
                "time_range_sec": row.get("time_range_sec"),
                "audio_asset_candidate": bool(row.get("audio_asset_candidate")),
                "preview_audio_path": row.get("review_audio_path"),
                "cut_success": bool(row.get("cut_success")),
                "voice_speaker_id": link.get("speaker_id"),
                "voice_link": link,
                # Previews are never implicitly approved: a human decision arrives via
                # --approvals, everything else stays pending.
                "review_status": ("approved"
                                  if (int(row.get("global_id") or 0), str(row.get("asr_id")))
                                  in approved_keys else "pending"),
            })
        write_jsonl(output_dir / "voice_asset_manifest.jsonl", voice_rows)
    reasons = Counter(
        reason
        for row in invalid_rows
        for reason in row.get("audio_asset_rejection_reasons", [])
    )
    summary = {
        "source_manifest": str(source_manifest),
        "classification_field": "audio_asset_candidate",
        "valid_definition": "audio_asset_candidate=true (preliminary candidate, not final admission)",
        "invalid_definition": "audio_asset_candidate=false",
        "audio_format": "PCM s16le, 16000 Hz, mono",
        "total_count": len(results),
        "valid_count": len(valid_rows),
        "invalid_count": len(invalid_rows),
        "cut_success_count": sum(row["cut_success"] for row in results),
        "cut_failed_count": sum(not row["cut_success"] for row in results),
        "counts_by_global_and_status": {
            f"global_{global_id:04d}/{status}": count
            for (global_id, status), count in sorted(Counter(
                (int(row.get("global_id") or 0), row["review_status"]) for row in results
            ).items())
        },
        "invalid_rejection_reason_counts": dict(sorted(reasons.items())),
        "all_manifest": str(output_dir / "all_asr_manifest.jsonl"),
        "valid_manifest": str(output_dir / "valid_manifest.jsonl"),
        "invalid_manifest": str(output_dir / "invalid_manifest.jsonl"),
    }
    if voice_registries:
        summary["voice_link"] = {
            "schema_version": VOICE_ASSET_SCHEMA_VERSION,
            "registries_indexed": len(voice_registries),
            "min_coverage": args.min_voice_coverage,
            "min_purity": args.min_voice_purity,
            "min_net_seconds": args.min_voice_net_seconds,
            "status_counts": dict(sorted(Counter(
                str((row.get("voice_link") or {}).get("status"))
                for row in voice_rows).items())),
            "approved_count": sum(row["review_status"] == "approved" for row in voice_rows),
            "manifest": str(output_dir / "voice_asset_manifest.jsonl"),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["cut_failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
