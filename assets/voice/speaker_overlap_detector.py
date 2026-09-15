#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect simultaneous speech, which CAM++ diarization structurally cannot report.

``speaker_registry_builder`` marks a turn ``overlap`` from its own diarizer's output, but that
diarizer returns a strict time partition: every segment belongs to exactly one label, so
``overlap`` is ``False`` for every turn it ever produces.  The consequence is invisible in the
registry and fatal for a voice asset -- a reference clip cut from a turn where two people talk
at once is labelled single-speaker and reads as single-speaker to every downstream consumer.

Measured on ``global_0005``: 1.376 s of the 32.47 s of delivered reference audio, 4.2%, was
simultaneous speech.  One speaker's *only* reference clip was 26.2% two people talking.

pyannote's ``SpeakerDiarization`` models overlap natively, and its weights are already on disk
under ``teleAI-T2V/local_models/pyannote_test/`` (5.9 MB segmentation-3.0 + 26.6 MB wespeaker
ResNet34 embedding), so this needs no download and no network.

Scope, deliberately narrow
--------------------------
This module answers **"when are two people talking at once"** and nothing else.  It does NOT
decide who anyone is.  pyannote reported 2 speakers for ``global_0005`` where CAM++ reported
6-7, and it missed an audible second voice around 24-26 s -- its clustering is configured for
teleAI's T2V packing job, whose wrapper goes so far as to keep only the longest segment per
speaker.  So its speaker *labels* are written to the report for audit and cross-checking, and
are never used for identity.  Identity stays with the pooled CAM++ clustering in
``speaker_voice_asset_pipeline.py``, which is the part that has been validated.

The consumer subtracts ``overlap_regions`` from the intervals it embeds and cuts, which makes
"single speaker" literally true of the delivered audio instead of merely asserted.  That costs
content -- 1.376 s on this clip -- and recovering it would need blind source separation
(``MossFormer2_SS_16K``, 670 MB, not on disk) with embedding-based permutation resolution.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "speaker-overlap-v1"
REPO_ROOT = Path(__file__).resolve().parent
# Sibling repo, same convention as audio_foreground_separator.DEFAULT_CLEARVOICE_ROOT.
# 迁移到 assets/voice/ 后工作区根（wyn/）在上四级。
DEFAULT_PYANNOTE_CONFIG = (REPO_ROOT.parents[3] / "teleAI-T2V" / "local_models" / "pyannote_test"
                           / "model" / "pyannote_diarization_config.yaml")
# pyannote's segmentation runs on 16 kHz mono, which is exactly what Stage5 and the enhanced
# stem already use, so no resampling is needed on the normal path.
EXPECTED_SAMPLE_RATE = 16000
# Below this an "overlap" is shorter than the segmentation model's own frame resolution can
# justify.  global_0005 has one at 0.017 s (272 samples).  Keeping the same number as
# speaker_voice_asset_pipeline.DEFAULT_MAX_TEMPORAL_OVERLAP so the two overlap notions agree.
DEFAULT_MIN_OVERLAP_SECONDS = 0.10


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _config_cwd(config: Path):
    # The shipped config references its weights by relative path ("segmentation-3.0/
    # pytorch_model.bin", "embedding/pytorch_model.bin"), so cwd must be their common parent.
    # teleAI's video_pyannote_pipeline.load_pipeline_from_pretrained does the same chdir.
    previous = Path.cwd()
    os.chdir(config.parent.parent)
    try:
        yield
    finally:
        os.chdir(previous)


def load_pipeline(config: Path, device: Optional[str] = None):
    """Load pyannote's SpeakerDiarization fully offline.  Returns ``(pipeline, resolved_config)``."""
    config = Path(config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(
            f"pyannote config not found: {config}. Weights are expected alongside it "
            f"(segmentation-3.0/ and embedding/); this module never downloads.")
    # Forced rather than inherited: a stray HF_HUB_OFFLINE=0 would let pyannote reach the
    # network and silently substitute a different checkpoint, making runs unreproducible.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from pyannote.audio import Pipeline

    with _config_cwd(config):
        pipeline = Pipeline.from_pretrained(config)
    if pipeline is None:
        raise SystemExit(f"{config}: Pipeline.from_pretrained returned None")
    if device:
        import torch

        pipeline.to(torch.device(device))
    elif _cuda_available():
        import torch

        pipeline.to(torch.device("cuda"))
    return pipeline, config


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _capture_tf32() -> Tuple[bool, bool]:
    import torch

    return bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32)


def _restore_tf32(saved: Tuple[bool, bool]) -> None:
    """Undo pyannote's process-wide TF32 disable.

    ``pyannote.audio.utils.reproducibility.fix_reproducibility`` sets both flags to False and
    never puts them back, so a caller that runs another CUDA model afterwards silently gets
    different numerics from it.  In the voice asset pipeline that other model is CAM++, whose
    embeddings decide speaker identity.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _round6(value: float) -> float:
    return round(float(value), 6)


def overlap_from_annotation(annotation: Any, min_overlap_seconds: float) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Derive overlap regions by pairwise intersection of per-speaker timelines.

    ``annotation.get_overlap()`` returns a bare ``Timeline`` with no labels, and
    ``annotation.get_labels(segment)`` comes back empty for a segment that was not itself
    indexed -- so neither can say *who* is overlapping.  Intersecting the per-speaker
    timelines does, and pyannote's own total is then used as a cross-check rather than as
    the source of truth.
    """
    per_speaker: Dict[str, List[Tuple[float, float]]] = {}
    segments: List[Dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        per_speaker.setdefault(str(speaker), []).append((float(segment.start), float(segment.end)))
        segments.append({"start": _round6(segment.start), "end": _round6(segment.end),
                         "speaker": str(speaker)})
    segments.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))

    regions: List[Dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(per_speaker), 2):
        for a_start, a_end in per_speaker[left]:
            for b_start, b_end in per_speaker[right]:
                start, end = max(a_start, b_start), min(a_end, b_end)
                if end - start > min_overlap_seconds:
                    regions.append({"start": _round6(start), "end": _round6(end),
                                    "duration": _round6(end - start),
                                    "speakers": [left, right]})
    # Overlapping pairs can themselves overlap (3+ people at once); merge so the consumer
    # gets disjoint intervals it can subtract in one pass.
    regions = _merge_regions(regions)

    speech = sum(end - start for rows in per_speaker.values() for start, end in rows)
    overlap_seconds = sum(region["duration"] for region in regions)
    # pyannote's own number, kept as an independent check on the derivation above.
    builtin = annotation.get_overlap()
    builtin_seconds = sum(float(seg.end) - float(seg.start) for seg in builtin)
    stats = {
        "speaker_count": len(per_speaker),
        "speaker_labels": sorted(per_speaker),
        "speaker_seconds": {spk: _round6(sum(end - start for start, end in rows))
                            for spk, rows in sorted(per_speaker.items())},
        # Summed per speaker, so it double-counts the overlap; that is the point -- compare
        # it against the merged speech support to see how much was shared.
        "segment_seconds": _round6(speech),
        "speech_support_seconds": _round6(annotation.get_timeline().support().duration()),
        "overlap_seconds": _round6(overlap_seconds),
        "overlap_region_count": len(regions),
        "pyannote_builtin_overlap_seconds": _round6(builtin_seconds),
        "segment_count": len(segments),
    }
    return regions, segments, sorted(per_speaker), stats


def _merge_regions(regions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for region in sorted(regions, key=lambda row: (row["start"], row["end"])):
        if merged and region["start"] <= merged[-1]["end"]:
            previous = merged[-1]
            previous["end"] = max(previous["end"], region["end"])
            previous["duration"] = _round6(previous["end"] - previous["start"])
            previous["speakers"] = sorted(set(previous["speakers"]) | set(region["speakers"]))
        else:
            merged.append(dict(region, speakers=sorted(set(region["speakers"]))))
    return merged


def detect_overlap(audio: Path, *, config: Path = DEFAULT_PYANNOTE_CONFIG,
                   device: Optional[str] = None, pipeline: Any = None,
                   min_overlap_seconds: float = DEFAULT_MIN_OVERLAP_SECONDS,
                   duration: Optional[float] = None) -> Dict[str, Any]:
    """Run pyannote over one audio file and return the overlap report."""
    started = time.perf_counter()
    audio = Path(audio).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")
    tf32 = _capture_tf32()
    try:
        loaded, resolved_config = (pipeline, Path(config).expanduser().resolve()) \
            if pipeline is not None else load_pipeline(config, device)
        annotation = loaded(str(audio))
    finally:
        _restore_tf32(tf32)
    regions, segments, speakers, stats = overlap_from_annotation(annotation, min_overlap_seconds)
    if duration is None:
        import soundfile as sf

        duration = float(sf.info(str(audio)).duration)
    config_text = resolved_config.read_text(encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "HAS_OVERLAP" if regions else "NO_OVERLAP",
        "input": str(audio),
        "duration": _round6(duration),
        "model": _model_audit(loaded, resolved_config, config_text),
        "parameters": {"min_overlap_seconds": min_overlap_seconds},
        # Audit only -- see the module docstring for why these labels never drive identity.
        "speakers": speakers,
        "segments": segments,
        "overlap_regions": regions,
        "activity": stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _model_audit(pipeline: Any, config: Path, config_text: str) -> Dict[str, Any]:
    """Record which checkpoint configuration actually ran.

    Read off the live pipeline object rather than re-parsed from the YAML, so a caller that passes
    its own pre-built pipeline still gets an honest record.
    """
    clustering = getattr(pipeline, "clustering", None)
    # pyannote 3.1.1 exposes the segmentation model as `_segmentation`; the public `.models`
    # registry this used to read does not exist (it is `_models`), which is why device was null.
    device = getattr(getattr(pipeline, "_segmentation", None), "device", None)
    threshold = getattr(clustering, "threshold", None)
    return {
        "backend": "pyannote.audio",
        "pipeline": "SpeakerDiarization",
        "config": str(config),
        "config_sha1": _sha1(config_text),
        # A looser threshold merges two voices into one label and erases the overlap entirely, so
        # the clustering knobs are part of what produced `overlap_regions`.
        "clustering_threshold": _round6(threshold) if isinstance(threshold, (int, float)) else None,
        "clustering_method": getattr(clustering, "method", None),
        "min_cluster_size": getattr(clustering, "min_cluster_size", None),
        "embedding_exclude_overlap": getattr(pipeline, "embedding_exclude_overlap", None),
        "segmentation_batch_size": getattr(pipeline, "segmentation_batch_size", None),
        "embedding_batch_size": getattr(pipeline, "embedding_batch_size", None),
        "device": str(device) if device is not None else None,
    }


def _sha1(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_report(path: Path) -> Dict[str, Any]:
    report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"{path}: expected schema {SCHEMA_VERSION}, "
                         f"got {report.get('schema_version')!r}")
    return report


def overlap_intervals(report: Dict[str, Any]) -> List[Tuple[float, float]]:
    """The report's overlap regions as plain ``(start, end)`` pairs, ready to subtract."""
    return [(float(region["start"]), float(region["end"]))
            for region in report.get("overlap_regions") or []]


def check_duration(report: Dict[str, Any], duration: Optional[float],
                   tolerance: float = 0.5) -> None:
    expected = report.get("duration")
    if duration is None or expected is None:
        return
    if abs(float(expected) - float(duration)) > tolerance:
        raise SystemExit(f"overlap report covers {float(expected):.3f}s but the timeline is "
                         f"{float(duration):.3f}s; they are not the same clip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Runs fully offline; the pyannote weights must already exist next to --config.",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="wav to diarize (16 kHz mono is what the model expects)")
    parser.add_argument("--output", required=True, type=Path,
                        help="where to write overlap_report.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_PYANNOTE_CONFIG,
                        help="pyannote diarization config yaml with local weight paths")
    parser.add_argument("--device", help="torch device, e.g. cuda or cpu; defaults to cuda if present")
    parser.add_argument("--min-overlap-seconds", type=float, default=DEFAULT_MIN_OVERLAP_SECONDS,
                        help="ignore overlaps shorter than this; below it the region is shorter "
                             "than the segmentation model can justify")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_overlap_seconds < 0:
        raise SystemExit("--min-overlap-seconds must be >= 0")
    output = Path(args.output).expanduser()
    if output.is_file() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {output}; pass --overwrite")
    report = detect_overlap(Path(args.input), config=args.config, device=args.device,
                            min_overlap_seconds=args.min_overlap_seconds)
    _atomic_json(output, report)
    activity = report["activity"]
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "speaker_count": activity["speaker_count"],
        "overlap_seconds": activity["overlap_seconds"],
        "overlap_region_count": activity["overlap_region_count"],
        "speech_support_seconds": activity["speech_support_seconds"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
