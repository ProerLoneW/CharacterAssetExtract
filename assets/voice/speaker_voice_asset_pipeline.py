#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean-voice speaker extraction: foreground separation -> Stage5 -> pooled consolidation.

What was wrong before
---------------------
``speaker_registry_builder.py`` runs diarization, VAD, ASR, embedding and turn quality on
whatever ``--input`` hands it -- normally the mixture, music and all.  ``--foreground-report``
only changes *which wav gets written* for a reference clip; the embedding that decided who
the speaker is was still computed on music, and so was the diarization that decided where the
turns are.  Measured on ``global_0005``: 6 speakers for a 76 s clip, 5 of them singletons
with ``NO_SAFE_REFERENCE``, and the contamination numbers show why the obvious explanation is
wrong -- those phantom speakers sit on audio with contamination 0.0040-0.1476, i.e. essentially
clean.  They are short turns (0.94-4.45 s) whose CAM++ embeddings are simply too noisy to
clear the fixed 0.60 cluster threshold.

What this module does
---------------------
Four stages, none of which edits an existing file and none of which calls an LLM or VLM:

1. ``separate``     MossFormer2_SE foreground/background split (``audio_foreground_separator``).
2. ``build``        Stage5 with ``--input`` pointed at the *16 kHz foreground stem*.  The builder
                    already re-encodes ``--input`` to 16 kHz mono, so feeding it the stem is a
                    no-op re-encode and every stage -- diarization, VAD, ASR, embedding, turn
                    quality -- runs on clean voice with zero edits to the builder.
3. ``gate``         ``speaker_turn_foreground_gate.apply_to_registry`` before any identity
                    decision of ours.  Running it here rather than after Stage5 closes the
                    capability loss documented in the migration doc section 4.7 -- a demoted
                    turn cannot seed a cluster, because the cluster that matters is built next.
4. ``overlap``      ``speaker_overlap_detector`` runs pyannote over the stem and reports where
                    two people talk at once.  Those spans are subtracted from every interval the
                    pipeline embeds or cuts, so nothing delivered contains a second voice.
5. ``consolidate``  Re-embed each speaker from its *pooled* clean audio (seconds-to-tens-of-
                    seconds concatenated, instead of one 0.94 s clip), then merge speakers whose
                    pooled voiceprints agree and whose turns never talk at the same time.
                    Representative audio is re-cut from the stem with leave-one-out identity
                    evidence, so a clip is never compared against a prototype it is inside.
                    Merges are voted pair-wise but executed group-wise: before each union the
                    whole groups on both sides are re-checked for cannot-links and simultaneous
                    speech, so a chain A-B, B-C can never smuggle in an A-C conflict, and a pair
                    where either side has less than --min-speaker-clean-seconds of clean
                    evidence is never merged.  Turns rejected by Stage5 only for being shorter
                    than its minimum duration (clean, single-speaker, foreground-confirmed)
                    still count as evidence: their spans join the speaker's pooled seconds and,
                    when the regular pool is too small, the prototype embedding.  Usable turns
                    below the regular reference-duration bar can seed a short-turn reference
                    (>= --short-reference-min-duration) when no regular-length reference exists.
                    A reference with no leave-one-out support is rejected outright
                    (``no_independent_support``) instead of silently lowering the identity
                    requirement to zero.  A turn rejected only for excessive simultaneous
                    speech is not dropped whole when it still holds a continuous single-speaker
                    span of at least --min-recovered-segment-seconds: that span may anchor a
                    reference delivered as one unbroken segment (``recovery="partial_overlap"``,
                    with the parent turn recorded), while the turn's remaining clean spans stay
                    pooling material and are never stitched into a delivered clip.
                    Every reference candidate is additionally embedded from the *original
                    mixture* (the foreground report's ``origin_16000`` file) and that second
                    opinion is compared with the stem-path identity verdict -- by default as a
                    recorded shadow audit only (``dual_path`` on each candidate:
                    ``agree``/``conflict``/``unavailable``).  ``--dual-path-enforce`` turns a
                    conflict into a rejection reason so a clip whose identity depends on the
                    enhancer's say-so goes to review instead of being delivered.

Where the gain actually comes from
----------------------------------
Not from stage 2.  A non-circular probe (anchor = the 10 turns Stage5 itself grouped into its
biggest speaker, so no merge this pipeline made is used as evidence) compared every turn
embedded from the mixture against the same turn embedded from the stem:

    separation margin (within-anchor mean - best other-speaker cosine)
    mixture:  within 0.6030  max_between 0.5443  margin 0.0587
    enhanced: within 0.6084  max_between 0.5567  margin 0.0517

Enhancement does not make speakers more separable, and it made this margin slightly worse.
Stage5 fed the stem even returned 7 speakers for the clip versus 6 for the mixture, and the
mean between-speaker prototype cosine rose 0.2214 -> 0.2762.  What stage 2 does buy, measured:
the delivered reference audio is the enhanced stem bit-for-bit (mean 9.1 dB of background
removed, 4.3-16.8 dB across five clips), and the enhancer's speech regions expose segments
Stage5 would otherwise turn into turns -- two here, one of which is a whisper repetition loop
whose transcript duplicates its two neighbours and has zero overlapping speech regions.

The clustering improvement is stage 5's: pooling gives CAM++ tens of seconds instead of one
0.94 s clip, the evidence floor refuses to invent an identity below 2 s of clean speech, and
leave-one-out stops a single-turn gallery from scoring 1.0 against a prototype built out of
itself.  Stage 2 supplies the clean audio those three decisions need, not the decisions.

Two different kinds of impurity
-------------------------------
Background contamination and a simultaneous second speaker are orthogonal, and stage 2 only
fixes the first.  MossFormer2_SE removes music and noise -- measured mean 9.1 dB -- but both
voices in an overlap are speech, so enhancement keeps both.

Stage5 cannot report the second kind at all.  CAM++ diarization returns a strict time partition:
every segment belongs to exactly one label, so ``speaker_registry_builder`` marks ``overlap=False``
on every turn it ever produces and its ``overlap_regions`` is always empty.  The registry is
structurally blind to two people talking at once.  pyannote's ``SpeakerDiarization`` models
overlap natively, which is the only reason stage 4 exists.

Intersecting the two on ``global_0005`` found 1.376 s of simultaneous speech inside 32.47 s of
delivered reference audio (4.2%), including one speaker whose *only* reference clip was 26.2%
overlap.  Subtraction, not separation, is the response: separating two overlapping voices needs
a source-separation checkpoint that is not on disk here, whereas cutting the overlap out costs
at most a second or two per clip and leaves what survives unambiguously one person -- which is
what a voice asset has to be.

Outputs ``consolidated/registry.json`` in the Stage5 ``speaker-registry-v2`` schema, so
``multimodal_character_registry_builder.py --speakers`` consumes it unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

import speaker_overlap_detector as sod
import speaker_registry_builder as srb
from audio_foreground_separator import (DEFAULT_CLEARVOICE_ROOT, DEFAULT_MODEL,
                                        foreground_gate_for_turn, split_foreground_background)
from speaker_registry_builder import (SAMPLE_RATE, SpeakerEmbedder, has_lexical_speech,
                                      intersect_intervals, merge_intervals,
                                      reference_identity_evidence, unit, write_json)
from speaker_turn_foreground_gate import (DEFAULT_MAX_CONTAMINATION, DEFAULT_MIN_COVERAGE,
                                          apply_to_registry, load_report)

SCHEMA_VERSION = "speaker-voice-asset-v2"
# Deliberately the Stage5 schema: downstream readers key on it, and this registry is a
# drop-in replacement for one, not a new artefact type.
REGISTRY_SCHEMA_VERSION = "speaker-registry-v2"

# Measured on global_0005 (76 s, 7 Stage5 speakers, 21 pairs): the top pair is 0.6213 and the
# next is 0.5354, so 0.62 sits in a real 0.086 gap -- but that is not the widest gap in the
# distribution (0.4205 -> 0.3161 is 0.104), and the upper bound is a single observation.
# Treat this as a starting point to re-check per clip via `merge_decisions`, not a calibrated
# constant.
DEFAULT_POOLED_MERGE_THRESHOLD = 0.62
DEFAULT_POOLED_POSSIBLE_THRESHOLD = 0.50
DEFAULT_MAX_TEMPORAL_OVERLAP = 0.10
DEFAULT_MAX_POOL_SECONDS = 40.0
DEFAULT_TURN_EMBED_PADDING = 0.75
DEFAULT_MIN_POOL_SECONDS = 1.5
# Two turns of a speaker Stage5 already agreed on disagree by up to 0.42 cosine when each is
# under ~2 s (measured: within-speaker mean 0.608, min 0.224 over 10 turns of global_0005).
# Pooling cannot manufacture evidence that was never recorded, so below this much total clean
# speech a speaker is reported but issued no voice asset.
DEFAULT_MIN_SPEAKER_CLEAN_SECONDS = 2.0
# Subtracting overlap from a turn that is mostly overlap leaves fragments, and
# unrepresentative ones, because the surviving part is only wherever the other person
# happened to stop talking.  Such a turn cannot join the regular pool -- but it is no
# longer dropped whole either: when its longest continuous single-speaker span reaches
# --min-recovered-segment-seconds, that span is recovered as a candidate reference segment
# and the rest counts as pooling evidence.  0.5 remains a judgement call, not a measured
# threshold.
DEFAULT_MAX_OVERLAP_FRACTION = 0.5
# Floor for a reference recovered from a duration-only short turn.  Matches Stage5's own
# --min-turn-duration default: below it CAM++ gets too few frames for the clip to identify
# anyone, no matter which channel it came through.
DEFAULT_SHORT_REFERENCE_MIN_DURATION = 0.8
# A turn rejected for excessive simultaneous speech is recoverable only if it still holds
# one continuous single-speaker span at least this long; shorter fragments are too easily
# just the gaps between the other person's words.
DEFAULT_MIN_RECOVERED_SEGMENT_SECONDS = 1.0


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_stem(report: Dict[str, Any], sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, Path]:
    """Read the enhanced mono stem the whole pipeline will run on."""
    files = report.get("files") or {}
    source = files.get(f"foreground_{sample_rate}")
    if not source:
        raise SystemExit(f"foreground report has no foreground_{sample_rate} stem; "
                         f"re-run separation with --stage5-sample-rate {sample_rate}")
    path = _resolve(source)
    waveform, rate = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != sample_rate:
        raise SystemExit(f"{path}: expected {sample_rate} Hz, got {rate}")
    return np.asarray(waveform, np.float32), path


def load_origin(report: Dict[str, Any], sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Read the unprocessed mixture at the pipeline sample rate, for the dual-path audit.

    Returns ``None`` (rather than failing) when the report predates the resampled copies or
    the file was removed: the audit is a shadow check and must never block the pipeline.
    """
    files = report.get("files") or {}
    source = files.get(f"origin_{sample_rate}")
    if not source:
        return None
    path = _resolve(source)
    if not path.is_file():
        return None
    waveform, rate = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != sample_rate:
        return None
    return np.asarray(waveform, np.float32)


def _slice(stem: np.ndarray, start: float, end: float) -> np.ndarray:
    left, right = max(0, int(round(start * SAMPLE_RATE))), min(len(stem), int(round(end * SAMPLE_RATE)))
    return stem[left:right] if right > left else np.zeros(0, np.float32)


def _concat(stem: np.ndarray, intervals: Sequence[Tuple[float, float]]) -> np.ndarray:
    pieces = [_slice(stem, a, b) for a, b in sorted(intervals)]
    pieces = [piece for piece in pieces if len(piece)]
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)


# --------------------------------------------------------------------------- stage 4

@dataclass
class OverlapExclusions:
    """Spans where two people talk at once, to be removed from everything downstream.

    Stage5's ``overlap_regions`` cannot supply these: its diarizer returns a strict time
    partition, so every turn it emits carries ``overlap=False`` no matter what the audio does.
    """
    status: str = "DISABLED"
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    def spans(self, start: float, end: float) -> List[Tuple[float, float]]:
        """The part of ``[start, end]`` that is single-speaker speech."""
        return srb.subtract_intervals([(start, end)], self.intervals)

    def seconds(self, start: float, end: float) -> float:
        return round(sum(b - a for a, b in self.spans(start, end)), 6)

    def fraction(self, start: float, end: float) -> float:
        duration = end - start
        if duration <= 0:
            return 0.0
        # Rounded kept seconds can exceed the duration by one unit in the last place, which
        # would otherwise serialise as -0.0.
        return max(0.0, round(1.0 - self.seconds(start, end) / duration, 6))


def stage_overlap(args: argparse.Namespace, stem_path: Path, duration: float,
                  root: Path) -> OverlapExclusions:
    """Locate simultaneous speech on the stem so it can be subtracted, not separated."""
    if args.disable_overlap_exclusion:
        print("[overlap] exclusion disabled", file=sys.stderr)
        return OverlapExclusions()
    if args.overlap_report:
        path = _resolve(args.overlap_report)
        report = sod.load_report(path)
        print(f"[overlap] reusing {path}", file=sys.stderr)
    else:
        path = root / "overlap" / "overlap_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        report = sod.detect_overlap(stem_path, config=Path(args.overlap_config),
                                    device=args.overlap_device,
                                    min_overlap_seconds=args.min_overlap_seconds,
                                    duration=duration)
        write_json(path, report)
        activity = report["activity"]
        print(f"[overlap] status={report['status']} pyannote_speakers={activity['speaker_count']} "
              f"overlap_seconds={activity['overlap_seconds']} "
              f"regions={len(report['overlap_regions'])} "
              f"builtin={activity['pyannote_builtin_overlap_seconds']} "
              f"device={report['model']['device']} ({time.perf_counter() - started:.1f}s)",
              file=sys.stderr)
    # The report's timeline must be the one the registry was built on, or every subtraction
    # lands on the wrong samples.
    sod.check_duration(report, duration)
    intervals = sorted(sod.overlap_intervals(report))
    if report.get("parameters", {}).get("min_overlap_seconds") != args.min_overlap_seconds:
        print(f"[overlap] reusing a report filtered at "
              f"{report['parameters']['min_overlap_seconds']}s, not the requested "
              f"{args.min_overlap_seconds}s", file=sys.stderr)
    return OverlapExclusions(status=report.get("status") or "NO_OVERLAP", intervals=intervals,
                             report=report, path=path)


# --------------------------------------------------------------------------- stage 1

def stage_separate(args: argparse.Namespace, root: Path) -> Tuple[Dict[str, Any], Path]:
    if args.foreground_report:
        path = _resolve(args.foreground_report)
        report = load_report(path)
        print(f"[separate] reusing {path}", file=sys.stderr)
        return report, path
    destination = root / "foreground"
    report = split_foreground_background(
        Path(args.input), destination,
        clearvoice_root=Path(args.clearvoice_root), model_name=args.enhancer_model,
        ffmpeg_bin=args.ffmpeg, extract_sample_rate=args.enhance_extract_sample_rate,
        extract_channels=args.enhance_extract_channels, stage5_sample_rate=SAMPLE_RATE,
        overwrite=args.enhance_overwrite)
    print(f"[separate] status={report['status']} duration={report['duration']}s "
          f"foreground_seconds={report['activity']['foreground_seconds']} "
          f"background_only_seconds={report['activity']['background_only_seconds']}",
          file=sys.stderr)
    return report, destination / "foreground_report.json"


# --------------------------------------------------------------------------- stage 2

def stage_build(args: argparse.Namespace, stem_path: Path, report_path: Path,
                root: Path) -> Dict[str, Any]:
    """Run Stage5 unmodified, but on the enhanced stem instead of the mixture."""
    registry_dir = root / "stage5"
    if registry_dir.exists() and any(registry_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{registry_dir} is not empty; pass --overwrite")
        shutil.rmtree(registry_dir)
    # The builder re-encodes --input to 16 kHz mono with ffmpeg; feeding it a 16 kHz mono wav
    # is a no-op re-encode, which is what lets this work without touching the builder.
    namespace = argparse.Namespace(**vars(args))
    namespace.input = str(stem_path)
    namespace.output = str(registry_dir)
    namespace.foreground_report = str(report_path)
    started = time.perf_counter()
    registry = srb.run(namespace)
    print(f"[build] speaker_count={registry['speaker_count']} turn_count={registry['turn_count']} "
          f"quality_stats={registry['quality_stats']} "
          f"reference_audio_count={registry['reference_audio_count']} "
          f"({time.perf_counter() - started:.1f}s)", file=sys.stderr)
    return registry


# --------------------------------------------------------------------------- stage 5

@dataclass
class SpeakerPool:
    source_speaker_id: str
    # Pooled into the prototype, bounded by --max-pool-seconds.
    turn_ids: List[str] = field(default_factory=list)
    # Every turn that passed the gate, regardless of the pool budget.  Membership and
    # reference-clip candidacy come from this set; only the prototype uses turn_ids.
    all_turn_ids: List[str] = field(default_factory=list)
    # Turns rejected only for being shorter than Stage5's min duration: no
    # contamination, no overlap, real foreground speech.  They never anchor a
    # reference clip, but their clean seconds still count as speaker evidence.
    short_turn_ids: List[str] = field(default_factory=list)
    short_seconds: float = 0.0
    # Turns rejected only for excessive simultaneous speech that still contain a
    # long enough continuous single-speaker span.  The continuous span may anchor
    # a reference clip (delivered as one unbroken segment); the remaining clean
    # spans are pooling material only, never stitched into a delivered clip.
    partial_turn_ids: List[str] = field(default_factory=list)
    partial_spans: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    partial_seconds: float = 0.0
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    seconds: float = 0.0
    embedding: Optional[np.ndarray] = None
    gate_rejected_turn_ids: List[str] = field(default_factory=list)


def turn_gate(report: Dict[str, Any], turn: Dict[str, Any], exclusions: OverlapExclusions,
              min_coverage: float, max_contamination: float,
              max_overlap_fraction: float,
              min_turn_duration: float = 0.0) -> Tuple[Dict[str, Any], bool, List[str], bool]:
    """Return the gate row, whether the turn is usable as clean voice, why not,
    and whether the *only* problem is that the turn is short.

    ``duration_only_rejection`` separates a turn that is merely too short for
    Stage5's minimum duration from turns rejected for contamination, overlap or
    missing foreground speech.  Only the former are eligible for the short-turn
    recovery pool; the others carry real pollution and stay excluded.
    """
    gate = foreground_gate_for_turn(report, turn["start"], turn["end"], min_coverage)
    start, end = float(turn["start"]), float(turn["end"])
    single_speaker_seconds = exclusions.seconds(start, end)
    overlap_fraction = exclusions.fraction(start, end)
    gate = {**gate, "single_speaker_seconds": single_speaker_seconds,
            "overlap_seconds": max(0.0, round(end - start - single_speaker_seconds, 6)),
            "overlap_fraction": overlap_fraction}
    reasons: List[str] = []
    if turn.get("overlap"):
        reasons.append("overlap")
    if turn.get("quality_tier") == "rejected":
        reasons.append("quality_tier_rejected")
    if not gate["has_foreground_speech"]:
        reasons.append("no_foreground_speech")
    contamination = gate.get("music_contamination")
    if contamination is not None and contamination > max_contamination:
        reasons.append("excessive_music_contamination")
    if overlap_fraction > max_overlap_fraction:
        reasons.append("excessive_simultaneous_speech")
    usable = not reasons
    quality = turn.get("quality") or {}
    duration = float(turn.get("duration") or (end - start))
    duration_only = (
        not usable
        and reasons == ["quality_tier_rejected"]
        and duration < min_turn_duration
        and float(quality.get("rms") or 0.0) >= 0.003
        and gate["has_foreground_speech"]
    )
    return gate, usable, reasons, duration_only


def collect_pools(registry: Dict[str, Any], report: Dict[str, Any], stem: np.ndarray,
                  embedder: SpeakerEmbedder, exclusions: OverlapExclusions, *,
                  max_pool_seconds: float, min_pool_seconds: float, min_coverage: float,
                  max_contamination: float,
                  max_overlap_fraction: float,
                  min_turn_duration: float = 0.0,
                  min_recovered_segment_seconds: float = 1.0
                  ) -> Tuple[List[SpeakerPool], Dict[str, Dict[str, Any]]]:
    """One pooled clean-voice voiceprint per Stage5 speaker.

    Turns are selected best-quality-first up to ``max_pool_seconds``, then concatenated in time
    order so the waveform still has natural prosody for CAM++'s statistics pooling.  Simultaneous
    speech is cut out of each turn first, so the prototype is one person throughout.

    Turns rejected only for being shorter than Stage5's minimum duration are
    kept in ``short_turn_ids``: they cannot anchor a reference clip, but when
    the regular pool is too small their clean spans are pooled into the
    prototype embedding as recovered evidence.

    Turns rejected only for excessive simultaneous speech are kept in
    ``partial_turn_ids`` when they still hold a continuous single-speaker span
    of at least ``min_recovered_segment_seconds``: the longest such span may
    anchor a reference clip delivered as one unbroken segment, and all of the
    turn's clean spans count as pooling evidence.
    """
    turns_by_id = {turn["turn_id"]: turn for turn in registry.get("turns") or []}
    gates: Dict[str, Dict[str, Any]] = {}
    for turn in registry.get("turns") or []:
        gate, usable, reasons, duration_only = turn_gate(
            report, turn, exclusions, min_coverage, max_contamination,
            max_overlap_fraction, min_turn_duration)
        spans = exclusions.spans(float(turn["start"]), float(turn["end"]))
        longest_span = max((b - a for a, b in spans), default=0.0)
        partial = (not usable and not duration_only
                   and sorted(reasons) == ["excessive_simultaneous_speech"]
                   and longest_span >= min_recovered_segment_seconds)
        gates[turn["turn_id"]] = {**gate, "usable": usable, "excluded_reasons": reasons,
                                  "duration_only_rejection": duration_only,
                                  "partial_overlap_recoverable": partial}

    pools: List[SpeakerPool] = []
    for speaker in registry.get("speakers") or []:
        pool = SpeakerPool(source_speaker_id=speaker["speaker_id"])
        rows = [(turns_by_id[tid], gates[tid]) for tid in speaker.get("turns") or [] if tid in turns_by_id]
        candidates = sorted((row for row in rows if row[1]["usable"]),
                            key=lambda row: -float(row[0].get("quality_score") or 0.0))
        short_candidates = sorted((row for row in rows
                                   if not row[1]["usable"] and row[1]["duration_only_rejection"]),
                                  key=lambda row: -float(row[0].get("quality_score") or 0.0))
        partial_candidates = sorted((row for row in rows
                                     if row[1]["partial_overlap_recoverable"]),
                                    key=lambda row: -float(row[0].get("quality_score") or 0.0))
        pool.gate_rejected_turn_ids = [row[0]["turn_id"] for row in rows
                                       if not row[1]["usable"]
                                       and not row[1]["duration_only_rejection"]
                                       and not row[1]["partial_overlap_recoverable"]]
        pool.all_turn_ids = [row[0]["turn_id"] for row in candidates]
        short_spans: List[Tuple[float, float]] = []
        for turn, _ in short_candidates:
            spans = exclusions.spans(float(turn["start"]), float(turn["end"]))
            if spans:
                pool.short_turn_ids.append(turn["turn_id"])
                short_spans.extend(spans)
        pool.short_seconds = round(sum(b - a for a, b in short_spans), 6)
        partial_span_list: List[Tuple[float, float]] = []
        for turn, _ in partial_candidates:
            spans = exclusions.spans(float(turn["start"]), float(turn["end"]))
            if spans:
                pool.partial_turn_ids.append(turn["turn_id"])
                pool.partial_spans[turn["turn_id"]] = spans
                partial_span_list.extend(spans)
        pool.partial_seconds = round(sum(b - a for a, b in partial_span_list), 6)
        for turn, _ in candidates:
            spans = exclusions.spans(float(turn["start"]), float(turn["end"]))
            duration = sum(b - a for a, b in spans)
            if duration <= 0.0:
                continue
            # Always keep the best turn, even if it alone exceeds the budget, so a speaker whose
            # only clean evidence is one long monologue still gets a prototype.
            if pool.seconds >= max_pool_seconds or (pool.seconds + duration > max_pool_seconds
                                                    and pool.seconds >= min_pool_seconds):
                continue
            pool.turn_ids.append(turn["turn_id"])
            pool.intervals.extend(spans)
            pool.seconds += duration
        pool.intervals.sort()
        pool.seconds = round(pool.seconds, 6)
        embed_intervals = list(pool.intervals)
        # Recovery: a speaker whose usable pool cannot reach the embedding floor
        # borrows its duration-only short turns and partial-overlap clean spans --
        # still clean, single-speaker, foreground-confirmed audio -- rather than
        # going without a voiceprint.
        if pool.seconds < min_pool_seconds and (short_spans or partial_span_list):
            embed_intervals = merge_intervals(list(pool.intervals) + short_spans
                                              + partial_span_list)
        waveform = _concat(stem, embed_intervals)
        pool.embedding = embedder.embed(waveform) if len(waveform) else None
        pools.append(pool)
    return pools, gates


def embed_turns(registry: Dict[str, Any], stem: np.ndarray, embedder: SpeakerEmbedder,
                exclusions: OverlapExclusions, padding: float) -> Dict[str, np.ndarray]:
    """Embed every turn from the stem, padded into time no other speaker claims.

    Padding matters because a 0.94 s clip gives CAM++ too few frames to be reliable, which is
    the direct cause of the phantom singleton speakers.  Intervals belonging to a different
    Stage5 speaker (or to no speaker) are subtracted first, so the padding can never smuggle
    another person's voice into the voiceprint -- and neither can an overlap, which Stage5's
    partition never attributes to anyone.
    """
    turns = registry.get("turns") or []
    embeddings: Dict[str, np.ndarray] = {}
    for turn in turns:
        speaker_id = turn.get("speaker_id")
        foreign = merge_intervals(list(exclusions.intervals)
                                  + [(other["start"], other["end"]) for other in turns
                                     if other.get("speaker_id") != speaker_id])
        start, end = float(turn["start"]), float(turn["end"])
        spans = srb.subtract_intervals([(max(0.0, start - padding), end + padding)], foreign)
        waveform = _concat(stem, spans)
        vector = embedder.embed(waveform) if len(waveform) else None
        if vector is not None:
            embeddings[turn["turn_id"]] = vector
    return embeddings


def temporal_overlap_seconds(left: Sequence[Tuple[float, float]],
                             right: Sequence[Tuple[float, float]]) -> float:
    return round(sum(b - a for a, b in intersect_intervals(list(left), list(right))), 6)


def snorm(score: float, left_cohort: Sequence[np.ndarray], right_cohort: Sequence[np.ndarray],
          left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Symmetric normalisation: express a cosine in units of each side's imposter spread.

    A fixed cosine threshold cannot be right for every clip -- how separable two voices are
    depends on the material.  This divides the score by the cohort's own variance, so the
    decision adapts.  Reported for every pair regardless of whether it drives the decision.
    """
    def z(vector: np.ndarray, cohort: Sequence[np.ndarray]) -> Optional[Tuple[float, float]]:
        values = [float(np.dot(unit(vector), unit(other))) for other in cohort]
        if len(values) < 2:
            return None
        std = float(np.std(values))
        return (float(np.mean(values)), std if std > 1e-6 else 1e-6)

    left_stats, right_stats = z(left, left_cohort), z(right, right_cohort)
    if left_stats is None or right_stats is None:
        return None
    return round(0.5 * ((score - left_stats[0]) / left_stats[1]
                        + (score - right_stats[0]) / right_stats[1]), 6)


def plan_merges(pools: Sequence[SpeakerPool], turn_embeddings: Dict[str, np.ndarray],
                registry: Dict[str, Any], *, threshold: float, possible_threshold: float,
                snorm_threshold: float, use_snorm: bool,
                max_temporal_overlap: float,
                min_speaker_clean_seconds: float = 0.0) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """Union-find over pairs that agree acoustically and never speak simultaneously.

    Two safeguards beyond the per-pair acoustic vote:

    * a pair may only merge when both pools carry at least
      ``min_speaker_clean_seconds`` of clean evidence -- a voiceprint pooled
      from less than that cannot tell identity apart and must not drive a merge;
    * before each union the *entire current groups* of both sides are checked
      against one another, so accepting A-B and B-C can never smuggle in an
      A-C hard conflict (cannot-link tracklets or simultaneous speech).
    """
    turns = registry.get("turns") or []
    tracklet_by_turn = {turn["turn_id"]: turn.get("local_tracklet_id") for turn in turns}
    owned_by_speaker: Dict[str, set] = {}
    for turn in turns:
        if turn.get("speaker_id"):
            owned_by_speaker.setdefault(turn["speaker_id"], set()).add(turn["turn_id"])
    # An s-norm cohort has to be independent of the pair being scored, otherwise the
    # normalisation is measured against the very same vectors the cosine came from.
    cohort_by_speaker = {
        pool.source_speaker_id: [vector for tid, vector in turn_embeddings.items()
                                 if tid not in owned_by_speaker.get(pool.source_speaker_id, set())]
        for pool in pools}
    cannot = {frozenset((row["left"], row["right"])) for row in registry.get("cannot_links") or []}

    tracklets_of = {index: {tracklet_by_turn.get(tid) for tid in pool.all_turn_ids} - {None}
                    for index, pool in enumerate(pools)}

    pairs: List[Dict[str, Any]] = []
    parent = list(range(len(pools)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def group_members(index: int) -> List[int]:
        root = find(index)
        return [member for member in range(len(pools)) if find(member) == root]

    def group_blockers(left_index: int, right_index: int) -> List[str]:
        blockers: List[str] = []
        members_left, members_right = group_members(left_index), group_members(right_index)
        if any(frozenset((a_tracklet, b_tracklet)) in cannot
               for a in members_left for b in members_right
               for a_tracklet in tracklets_of[a] for b_tracklet in tracklets_of[b]):
            blockers.append("group_cannot_link")
        overlap = temporal_overlap_seconds(
            [span for member in members_left for span in pools[member].intervals],
            [span for member in members_right for span in pools[member].intervals])
        if overlap > max_temporal_overlap:
            blockers.append("group_simultaneous_speech")
        return blockers

    for i, left in enumerate(pools):
        for j, right in enumerate(pools):
            if j <= i or left.embedding is None or right.embedding is None:
                continue
            cosine = round(float(np.dot(unit(left.embedding), unit(right.embedding))), 6)
            overlap = temporal_overlap_seconds(left.intervals, right.intervals)
            normalised = snorm(cosine, cohort_by_speaker[left.source_speaker_id],
                               cohort_by_speaker[right.source_speaker_id],
                               left.embedding, right.embedding)
            acoustic = normalised >= snorm_threshold if use_snorm else cosine >= threshold
            hard_blockers = []
            if overlap > max_temporal_overlap:
                hard_blockers.append("simultaneous_speech")
            if any(frozenset((a, b)) in cannot
                   for a in tracklets_of[i] for b in tracklets_of[j]):
                hard_blockers.append("cannot_link")
            blockers = list(hard_blockers)
            if not acoustic:
                blockers.append("snorm_below_threshold" if use_snorm else "cosine_below_threshold")
            enough_evidence = (min(left.seconds, right.seconds) >= min_speaker_clean_seconds)
            if not enough_evidence:
                blockers.append("insufficient_clean_evidence")
            accepted = not blockers
            group_check: List[str] = []
            if accepted:
                # Edge-level agreement is necessary but not sufficient: only a
                # union that keeps the whole merged group constraint-clean is
                # actually performed.
                group_check = group_blockers(i, j)
                if group_check:
                    accepted = False
                    blockers.extend(group_check)
            # The band always keys on the raw cosine, even under --use-snorm, so it means the
            # same thing in both modes and matches Stage5's own possible_links semantics.
            possible = (not accepted and not hard_blockers and not group_check
                        and possible_threshold <= cosine < threshold)
            pairs.append({
                "left": left.source_speaker_id, "right": right.source_speaker_id,
                "pooled_cosine": cosine, "pooled_snorm": normalised,
                "temporal_overlap_seconds": overlap,
                # Prototype seconds, i.e. bounded by --max-pool-seconds.  A speaker record's
                # `pooled_seconds` is the delivered asset instead, which counts every gated-in
                # turn and can be larger (global_0005 SPK001: 39.05 here, 46.74 there).
                "left_prototype_seconds": left.seconds, "right_prototype_seconds": right.seconds,
                "accepted": accepted, "possible": possible, "blockers": blockers,
                "group_check_blockers": group_check,
            })
            if accepted:
                parent[find(i)] = find(j)

    clusters: Dict[int, List[int]] = {}
    for index in range(len(pools)):
        clusters.setdefault(find(index), []).append(index)
    # Biggest clean-voice pool first: SPK001 should be the person who talks most, which is
    # what the character asset library wants to see at the top.
    ordered = sorted(clusters.values(),
                     key=lambda members: (-sum(pools[m].seconds for m in members),
                                          min((pools[m].intervals[0][0] for m in members if pools[m].intervals),
                                              default=float("inf"))))
    return ordered, pairs


def attach_merge_evidence(speakers: List[Dict[str, Any]], pairs: Sequence[Dict[str, Any]]) -> None:
    """Record, per speaker, the closest other speaker it was compared against."""
    group_of: Dict[str, str] = {}
    for speaker in speakers:
        for source_id in speaker["source_speaker_ids"]:
            group_of[source_id] = speaker["speaker_id"]
    best: Dict[str, Dict[str, Any]] = {}
    for pair in pairs:
        left, right = group_of.get(pair["left"]), group_of.get(pair["right"])
        if left is None or right is None or left == right:
            continue
        for speaker_id, other_id in ((left, right), (right, left)):
            current = best.get(speaker_id)
            if current is None or pair["pooled_cosine"] > current["pooled_cosine"]:
                best[speaker_id] = {"speaker_id": other_id, "pooled_cosine": pair["pooled_cosine"],
                                    "pooled_snorm": pair["pooled_snorm"], "possible": pair["possible"],
                                    "temporal_overlap_seconds": pair["temporal_overlap_seconds"]}
    for speaker in speakers:
        speaker["nearest_other_speaker"] = best.get(speaker["speaker_id"])


# --------------------------------------------------------------------------- assets

def build_voice_assets(group_ids: Sequence[int], pools: Sequence[SpeakerPool],
                       registry: Dict[str, Any], gates: Dict[str, Dict[str, Any]],
                       stem: np.ndarray, embedder: SpeakerEmbedder, speaker_id: str,
                       root: Path, args: argparse.Namespace,
                       competitors: Sequence[np.ndarray],
                       exclusions: OverlapExclusions,
                       origin: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Re-cut one merged speaker's representative audio from the clean stem.

    ``origin`` is the unprocessed mixture at the same sample rate (the foreground
    report's ``origin_16000`` file); when present, each reference candidate is also
    embedded from it for the dual-path audit.
    """
    turns_by_id = {turn["turn_id"]: turn for turn in registry.get("turns") or []}
    tracklet_score_by_id = {row["local_tracklet_id"]: float(row.get("quality_score") or 0.0)
                            for row in registry.get("local_tracklets") or []
                            if row.get("local_tracklet_id")}
    # Membership is every gated-in turn plus the duration-only short turns and the
    # partial-overlap turns: their clean spans are single-speaker foreground audio and
    # count as evidence.  Only the prototype is budget-capped.
    member_turn_ids = ([tid for index in group_ids for tid in pools[index].all_turn_ids]
                       + [tid for index in group_ids for tid in pools[index].short_turn_ids]
                       + [tid for index in group_ids for tid in pools[index].partial_turn_ids])
    short_member_turn_ids = {tid for index in group_ids for tid in pools[index].short_turn_ids}
    partial_member_turn_ids = {tid for index in group_ids for tid in pools[index].partial_turn_ids}
    member_turns = [turns_by_id[tid] for tid in member_turn_ids if tid in turns_by_id]
    # Computed once so the pooled asset, the gallery and every reference cut agree on what
    # counts as this speaker's audio.
    member_spans = {turn["turn_id"]: exclusions.spans(float(turn["start"]), float(turn["end"]))
                    for turn in member_turns}
    asset_intervals = sorted(pair for spans in member_spans.values() for pair in spans)
    pooled_seconds = round(sum(b - a for a, b in asset_intervals), 6)
    raw_seconds = round(sum(float(turn["end"]) - float(turn["start"]) for turn in member_turns), 6)
    prototype_intervals = sorted(pair for index in group_ids for pair in pools[index].intervals)

    directory = root / "speaker_registry" / speaker_id
    directory.mkdir(parents=True, exist_ok=True)
    member_tracklet_ids = sorted({turns_by_id[tid].get("local_tracklet_id")
                                  for tid in member_turn_ids
                                  if turns_by_id.get(tid, {}).get("local_tracklet_id")})
    pooled_waveform = _concat(stem, asset_intervals)
    pooled_path = directory / "pooled_clean_voice.wav"
    # The prototype stays on the budget-capped pool: CAM++ statistics pooling gains nothing
    # from a fourth minute of the same voice, and the cap bounds cost on long monologues.
    prototype_vector = embedder.embed(_concat(stem, prototype_intervals)) if prototype_intervals else None
    if len(pooled_waveform):
        sf.write(str(pooled_path), pooled_waveform, SAMPLE_RATE, subtype="PCM_16")
    prototype = unit(prototype_vector) if prototype_vector is not None else None

    gallery, gallery_ids = [], []
    # Foreign = every turn outside this merged group, not just outside this turn, plus every span
    # where two people talk at once.  Padding into either would put someone else's voice into
    # this speaker's voiceprint.
    group_turn_ids = set(member_turn_ids)
    foreign_all = merge_intervals(list(exclusions.intervals)
                                  + [(other["start"], other["end"])
                                     for other in registry.get("turns") or []
                                     if other["turn_id"] not in group_turn_ids])
    for turn in member_turns:
        spans = srb.subtract_intervals(
            [(max(0.0, turn["start"] - args.turn_embed_padding),
              turn["end"] + args.turn_embed_padding)], foreign_all)
        vector = embedder.embed(_concat(stem, spans))
        if vector is not None:
            gallery.append(vector)
            gallery_ids.append(turn["turn_id"])
    if gallery:
        np.save(directory / "embedding_gallery.npy", np.asarray(gallery, np.float32))
    if prototype is not None:
        np.save(directory / "prototype.npy", prototype.astype(np.float32))

    # Leave-one-out: the pooled voiceprint minus this turn.  Comparing a clip against a
    # prototype it is itself inside returns ~1.00 and admits everything, which is exactly the
    # bug behind speaker_reference_separator._verify_single_speaker_purity false-rejecting a
    # clean monologue.
    def leave_one_out(excluded_turn_id: str) -> List[np.ndarray]:
        rest = sorted(pair for turn in member_turns if turn["turn_id"] != excluded_turn_id
                      for pair in member_spans[turn["turn_id"]])
        waveform = _concat(stem, rest)
        vector = embedder.embed(waveform) if len(waveform) >= int(0.5 * SAMPLE_RATE) else None
        return [vector] if vector is not None else []

    def longest_clean_span(turn: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        spans = member_spans.get(turn["turn_id"]) or []
        return max(spans, key=lambda pair: pair[1] - pair[0]) if spans else None

    def deliverable_seconds(turn: Dict[str, Any]) -> float:
        # A partial-overlap turn can only promise its longest continuous clean span;
        # a regular turn promises its whole window (overlap holes and all).
        if turn["turn_id"] in partial_member_turn_ids:
            span = longest_clean_span(turn)
            return (span[1] - span[0]) if span else 0.0
        return float(turn["end"]) - float(turn["start"])

    ranked = sorted((turn for turn in member_turns
                     if not turn.get("overlap") and turn.get("quality_tier") != "rejected"
                     and deliverable_seconds(turn) >= args.min_reference_duration),
                    key=lambda turn: -float(turn.get("quality_score") or 0.0))
    # pooled_seconds already counts the recovered short and partial-overlap turns, so the
    # evidence floor applies to the speaker's total clean speech, not only to its
    # regular-length turns.
    insufficient = pooled_seconds < args.min_speaker_clean_seconds
    recovery = False
    reference_min_duration = args.min_reference_duration
    if not ranked and not insufficient:
        # Short-turn recovery: no regular-length candidate exists, but the speaker clears the
        # evidence floor once short turns count.  A usable turn that failed only the regular
        # reference-duration bar (or a partial-overlap turn whose clean span does) is still
        # one person talking into a clean microphone, so it may anchor a reference at a lower
        # duration floor -- the identity evidence below is NOT relaxed with it.
        # (Duration-only rejected turns stay evidence-only: they are shorter than this floor
        # by construction.)
        ranked = sorted((turn for turn in member_turns
                         if not turn.get("overlap") and turn.get("quality_tier") != "rejected"
                         and deliverable_seconds(turn) >= args.short_reference_min_duration),
                        key=lambda turn: -float(turn.get("quality_score") or 0.0))
        recovery = bool(ranked)
        if recovery:
            reference_min_duration = args.short_reference_min_duration
    references: List[str] = []
    admitted_vectors: List[np.ndarray] = []
    candidates: List[Dict[str, Any]] = []
    if insufficient:
        ranked = []
    for turn in ranked:
        reasons: List[str] = []
        if not args.disable_asr and not has_lexical_speech(turn.get("text", ""),
                                                           args.min_reference_asr_characters):
            reasons.append("no_lexical_speech")
        gate = gates.get(turn["turn_id"]) or {}
        if not gate.get("has_foreground_speech", False):
            reasons.append("insufficient_foreground_speech")
        contamination = gate.get("music_contamination")
        if contamination is not None and contamination > args.max_reference_music_contamination:
            reasons.append("excessive_music_or_background_contamination")

        is_partial = turn["turn_id"] in partial_member_turn_ids
        if is_partial:
            # A partial-overlap turn delivers ONE continuous single-speaker segment cut
            # from the stem; the rest of the turn stays pooling material and is never
            # stitched into a delivered clip.
            span = longest_clean_span(turn)
            win_start = float(span[0])
            win_end = min(float(span[1]), win_start + args.max_reference_duration)
            waveform = _slice(stem, win_start, win_end)
        else:
            win_start = float(turn["start"])
            win_end = min(float(turn["end"]), win_start + args.max_reference_duration)
            # The clip is the concatenation of the single-speaker spans inside the window,
            # so it can be discontinuous and is shorter than the window when anything was
            # removed.
            waveform = _concat(stem, exclusions.spans(win_start, win_end))
        end = win_end
        delivered_seconds = round(len(waveform) / SAMPLE_RATE, 6)
        # max() is for the sign, not the value: subtract_intervals only returns sub-intervals of
        # the window, so this cannot be negative, but rounding a tiny residue gives -0.0.
        overlap_removed = max(0.0, round(end - win_start - delivered_seconds, 6))
        if delivered_seconds < reference_min_duration:
            reasons.append("insufficient_single_speaker_audio")
        # turn_quality divides by an empty waveform and returns NaN, which json writes as a bare
        # NaN token -- not valid JSON, and a silent corruption of every downstream reader.
        score, tier, quality = (srb.turn_quality(waveform, delivered_seconds, False,
                                                 reference_min_duration,
                                                 args.max_reference_duration)
                                if len(waveform) else (0.0, "rejected", {}))
        if tier == "rejected":
            reasons.append("reference_waveform_quality_rejected")
        candidate_embedding = embedder.embed(waveform)
        own_support = leave_one_out(turn["turn_id"])
        if candidate_embedding is None:
            identity: Dict[str, Any] = {"status": "REJECT", "reasons": ["missing_reference_embedding"]}
        elif not own_support:
            # Leave-one-out with nothing left is a self-match waiting to happen: an empty
            # support set must not lower the identity requirement to zero.
            identity = {"status": "REJECT", "reasons": ["no_independent_support"]}
        else:
            identity = reference_identity_evidence(
                candidate_embedding, own_support, list(competitors), admitted_vectors,
                minimum_similarity=args.min_reference_identity_similarity,
                minimum_support_count=args.min_reference_identity_support,
                minimum_margin=args.min_reference_competitor_margin,
                maximum_competitor_similarity=args.max_reference_competitor_similarity,
                minimum_pairwise_similarity=args.min_reference_pairwise_similarity)
        reasons.extend(identity.get("reasons", []))
        # Dual-path shadow check (P2/A5): the same delivered interval is embedded once from
        # the enhanced stem and once from the original mixture.  The origin path never
        # changes admission unless --dual-path-enforce is set; by default a disagreement is
        # recorded so the false-reject reduction can be measured before any rule acts on it.
        dual_path: Dict[str, Any] = {"status": "unavailable"}
        if origin is not None and candidate_embedding is not None and len(waveform):
            origin_waveform = (_slice(origin, win_start, win_end) if is_partial
                               else _concat(origin, exclusions.spans(win_start, win_end)))
            origin_embedding = embedder.embed(origin_waveform) if len(origin_waveform) else None
            if origin_embedding is not None:
                origin_vec = unit(origin_embedding)
                origin_own = max((float(np.dot(origin_vec, unit(support)))
                                  for support in own_support), default=None)
                origin_competitor = max((float(np.dot(origin_vec, unit(other)))
                                         for other in competitors), default=None)
                origin_margin = (origin_own - origin_competitor
                                 if origin_own is not None and origin_competitor is not None
                                 else None)
                origin_pass = (origin_own is not None
                               and origin_own >= args.min_reference_identity_similarity
                               and (origin_margin is None
                                    or origin_margin >= args.min_reference_competitor_margin))
                stem_pass = identity.get("status") == "PASS"
                dual_path = {
                    "status": "agree" if stem_pass == origin_pass else "conflict",
                    "origin_own_best_similarity": (round(origin_own, 6)
                                                   if origin_own is not None else None),
                    "origin_competitor_best_similarity": (round(origin_competitor, 6)
                                                          if origin_competitor is not None else None),
                    "origin_margin": (round(origin_margin, 6)
                                      if origin_margin is not None else None),
                }
        if dual_path["status"] == "conflict" and args.dual_path_enforce:
            # 两路冲突时转待审：不自动放行，也不按单路结果交付。
            reasons.append("dual_path_conflict")
        if not reasons and len(references) >= args.voice_top_k:
            reasons.append("voice_top_k_limit")
        admitted = not reasons
        destination = None
        if admitted:
            destination = directory / "references" / f"ref_{len(references) + 1:02d}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(destination), waveform, SAMPLE_RATE, subtype="PCM_16")
            references.append(str(destination))
            admitted_vectors.append(candidate_embedding)
        candidates.append({
            "turn_id": turn["turn_id"], "status": "ADMITTED" if admitted else "REJECTED",
            "reasons": reasons,
            # For a partial-overlap turn these describe the delivered continuous segment's
            # source interval, not the parent turn; the segment must never masquerade as
            # the whole parent turn.
            "start": round(win_start, 6), "end": round(end, 6),
            "duration": round(end - win_start, 6),
            "parent_turn_id": turn["turn_id"] if is_partial else None,
            # `duration` is the source window; the file that was written is
            # `single_speaker_seconds` long, because the overlap came out of the middle
            # (zero for a partial-overlap segment, which is continuous by construction).
            "single_speaker_seconds": delivered_seconds,
            "overlap_seconds_removed": overlap_removed,
            "text": turn.get("text", ""),
            "source": "enhanced_foreground_pooled",
            "recovery": ("partial_overlap" if is_partial
                         else "short_turn" if recovery else "standard"),
            "audio_path": str(destination) if destination else None,
            "foreground_gate": {key: gate.get(key) for key in
                                ("music_contamination", "foreground_coverage", "has_foreground_speech")},
            "overlap_gate": {key: gate.get(key) for key in
                             ("overlap_seconds", "overlap_fraction", "single_speaker_seconds")},
            "identity_gate": identity, "dual_path": dual_path,
            "output_quality_score": score,
            "output_quality_tier": tier, "output_quality": quality,
        })

    # A one-turn gallery would compare that turn against a prototype built from itself and
    # report 1.0, which reads as "perfectly pure" but is only a self-match.  Report nothing
    # rather than a number that cannot be distinguished from real evidence.
    consistencies = [round(float(np.dot(unit(prototype), unit(vector))), 6)
                     for vector in gallery] if prototype is not None and len(gallery) >= 2 else []
    return {
        "speaker_id": speaker_id, "character_id": None,
        "source_speaker_ids": [pools[index].source_speaker_id for index in group_ids],
        "local_tracklets": member_tracklet_ids,
        "turns": [turn["turn_id"] for turn in member_turns],
        "embedding_count": len(gallery),
        "embedding_gallery_turns": gallery_ids,
        "embedding_gallery": str(directory / "embedding_gallery.npy") if gallery else None,
        "prototype": str(directory / "prototype.npy") if prototype is not None else None,
        # Single-speaker seconds.  `turn_seconds` is what the member turns spanned before the
        # overlap came out, so the difference is auditable rather than inferred.
        "pooled_seconds": pooled_seconds,
        "turn_seconds": raw_seconds,
        "overlap_seconds_removed": max(0.0, round(raw_seconds - pooled_seconds, 6)),
        "reference_overlap_seconds_removed": round(
            sum(row["overlap_seconds_removed"] for row in candidates
                if row["status"] == "ADMITTED"), 6),
        "pooled_clean_voice": str(pooled_path) if len(pooled_waveform) else None,
        "mean_turn_consistency": round(float(np.mean(consistencies)), 6) if consistencies else None,
        "min_turn_consistency": round(min(consistencies), 6) if consistencies else None,
        "reference_audio": references,
        # Compatibility alias consumed by multimodal_character_registry_builder.
        "representative_audio": references,
        "reference_status": ("INSUFFICIENT_CLEAN_VOICE" if insufficient
                             else "READY" if references else "NO_SAFE_REFERENCE"),
        "insufficient_clean_voice": insufficient,
        "min_speaker_clean_seconds": args.min_speaker_clean_seconds,
        "short_turn_recovery": {
            "short_turn_ids": sorted(short_member_turn_ids),
            "short_turn_seconds": round(
                sum(b - a for tid in short_member_turn_ids for a, b in member_spans.get(tid, [])),
                6),
            "reference_channel": "short_turn" if recovery else "standard",
        },
        "partial_overlap_recovery": {
            "partial_turn_ids": sorted(partial_member_turn_ids),
            "partial_seconds": round(
                sum(b - a for tid in partial_member_turn_ids
                    for a, b in member_spans.get(tid, [])), 6),
        },
        "reference_source": "enhanced_foreground_stem",
        "reference_candidates": candidates,
        "prototype_excluded_turn_ids": sorted(
            {tid for index in group_ids for tid in pools[index].all_turn_ids}
            - {tid for index in group_ids for tid in pools[index].turn_ids}),
        "gate_rejected_turn_ids": sorted({tid for index in group_ids
                                          for tid in pools[index].gate_rejected_turn_ids}),
        # speaker_registry_builder:594 defines this as the mean over member TRACKLET scores, not
        # turn scores, and multimodal_character_registry_builder.copy_audio gates on it.  Keep the
        # builder's definition so the field does not silently change meaning here.
        "confidence": round(float(np.mean([tracklet_score_by_id[tid]
                                           for tid in member_tracklet_ids
                                           if tid in tracklet_score_by_id] or [0.0])), 6),
    }


def assemble_registry(registry: Dict[str, Any], groups: Sequence[Sequence[int]],
                      pools: Sequence[SpeakerPool], speakers: Sequence[Dict[str, Any]],
                      pairs: Sequence[Dict[str, Any]], stem_path: Path, report: Dict[str, Any],
                      report_path: Path, args: argparse.Namespace, root: Path,
                      gate_audit: Dict[str, Any],
                      exclusions: OverlapExclusions) -> Dict[str, Any]:
    """Emit a Stage5-schema registry whose speaker set is the consolidated one."""
    remap: Dict[str, str] = {}
    for speaker in speakers:
        for source_id in speaker["source_speaker_ids"]:
            remap[source_id] = speaker["speaker_id"]

    turns = []
    for turn in registry.get("turns") or []:
        row = dict(turn)
        row["speaker_id"] = remap.get(turn.get("speaker_id"), turn.get("speaker_id"))
        row["stage5_speaker_id"] = turn.get("speaker_id")
        turns.append(row)

    tracklets = []
    for tracklet in registry.get("local_tracklets") or []:
        row = dict(tracklet)
        row["speaker_id"] = remap.get(tracklet.get("speaker_id"), tracklet.get("speaker_id"))
        row["stage5_speaker_id"] = tracklet.get("speaker_id")
        tracklets.append(row)

    assignments = {row["local_tracklet_id"]: (row["speaker_id"] or f"UNKNOWN::{row['local_tracklet_id']}")
                   for row in tracklets}
    timeline = srb.build_multilabel_timeline(
        [types.SimpleNamespace(local_tracklet_id=row["local_tracklet_id"], segments=[tuple(s) for s in row["segments"]])
         for row in tracklets], assignments)
    overlaps = [{**row, "overlap_id": f"OVL{i:05d}"} for i, row in enumerate(
        (x for x in timeline if len(x["active_speakers"]) >= 2
         and x["end"] - x["start"] >= args.min_overlap_duration), 1)]
    simultaneous = [{"simultaneous_speech_id": f"SIM{i:05d}", "start": region["start"],
                     "end": region["end"], "duration": region["duration"],
                     "pyannote_speakers": region["speakers"]}
                    for i, region in enumerate(exclusions.report.get("overlap_regions") or [], 1)]
    unknown = [row for row in turns if row["speaker_id"] is None]

    def pooled_link(relation: str, pair: Dict[str, Any]) -> Dict[str, Any]:
        return {"relation": relation,
                "left": remap.get(pair["left"], pair["left"]),
                "right": remap.get(pair["right"], pair["right"]),
                "stage5_left": pair["left"], "stage5_right": pair["right"],
                "pooled_cosine": pair["pooled_cosine"], "pooled_snorm": pair["pooled_snorm"],
                "temporal_overlap_seconds": pair["temporal_overlap_seconds"]}

    merged_links = [pooled_link("POOLED_MERGE", pair) for pair in pairs if pair["accepted"]]
    pooled_possible = [pooled_link("POOLED_POSSIBLE", pair) for pair in pairs if pair.get("possible")]

    result = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "ok",
        "timeline": {**(registry.get("timeline") or {}), "audio_path": str(stem_path)},
        "speech_window_count": registry.get("speech_window_count"),
        "local_tracklet_count": len(tracklets),
        "speaker_count": len(speakers),
        "turn_count": len(turns),
        "unknown_turn_count": len(unknown),
        "rejected_asset_turn_count": sum(row["quality_tier"] == "rejected" for row in turns),
        "reference_audio_count": sum(len(speaker["reference_audio"]) for speaker in speakers),
        "safe_reference_speaker_count": sum(bool(speaker["reference_audio"]) for speaker in speakers),
        "quality_stats": {tier: sum(row["quality_tier"] == tier for row in turns)
                          for tier in ("high", "medium", "rejected")},
        "speakers": speakers,
        "turns": turns,
        "local_tracklets": tracklets,
        "global_speaker_timeline": timeline,
        "overlap_regions": overlaps,
        # `overlap_regions` above stays empty on any Stage5-produced registry: CAM++ returns a
        # strict time partition, so no segment is ever attributed to two labels at once.  These
        # are the regions stage 4 found and removed, kept separate so the Stage5 field's meaning
        # is not silently changed for readers that already consume it.
        "simultaneous_speech_regions": simultaneous,
        "must_links": list(registry.get("must_links") or []) + merged_links,
        "possible_links": list(registry.get("possible_links") or []) + pooled_possible,
        "cannot_links": registry.get("cannot_links") or [],
        "unknown_turns": unknown,
        "asr_timeline": registry.get("asr_timeline") or [],
        "metadata": {
            **(registry.get("metadata") or {}),
            "produced_by": SCHEMA_VERSION,
            "pipeline_input": str(Path(args.input).expanduser().resolve()),
            "stage5_input_was_enhanced_stem": True,
            "enhanced_stem": str(stem_path),
            "foreground_report": str(report_path),
            "identity_encoder": "CAM++ on enhanced foreground stem",
            "reference_source": "enhanced_foreground_stem",
            "consolidation": {
                "pooled_merge_threshold": args.pooled_merge_threshold,
                "pooled_possible_threshold": args.pooled_possible_threshold,
                "pooled_snorm_threshold": args.pooled_snorm_threshold,
                "use_snorm": args.use_snorm,
                "max_temporal_overlap": args.max_temporal_overlap,
                "max_pool_seconds": args.max_pool_seconds,
                "min_pool_seconds": args.min_pool_seconds,
                "min_speaker_clean_seconds": args.min_speaker_clean_seconds,
                "turn_embed_padding": args.turn_embed_padding,
                "gate_min_coverage": args.min_coverage,
                "gate_max_contamination": args.max_contamination,
                "stage5_speaker_count": len(registry.get("speakers") or []),
                "merged_speaker_count": len(speakers),
                "insufficient_clean_voice_speaker_count": sum(
                    bool(speaker.get("insufficient_clean_voice")) for speaker in speakers),
                "merge_decisions": pairs,
                "gate_tier_change_count": gate_audit.get("tier_change_count"),
                "dual_path_verification": {
                    "mode": ("disabled" if args.disable_dual_path_verification
                             else "enforce" if args.dual_path_enforce else "shadow"),
                    "candidate_count": sum(len(speaker["reference_candidates"])
                                           for speaker in speakers),
                    "agree_count": sum(1 for speaker in speakers
                                       for row in speaker["reference_candidates"]
                                       if row["dual_path"]["status"] == "agree"),
                    "conflict_count": sum(1 for speaker in speakers
                                          for row in speaker["reference_candidates"]
                                          if row["dual_path"]["status"] == "conflict"),
                    "unavailable_count": sum(1 for speaker in speakers
                                             for row in speaker["reference_candidates"]
                                             if row["dual_path"]["status"] == "unavailable"),
                },
            },
            "overlap_exclusion": {
                "status": exclusions.status,
                # Overlap only.  pyannote's labels are recorded in the report for audit but never
                # decide identity here -- it reports 2 speakers on this clip where Stage5 finds 7,
                # and its config is tuned for a different job.
                "detector": "pyannote.audio SpeakerDiarization",
                "report": str(exclusions.path) if exclusions.path else None,
                "model": exclusions.report.get("model"),
                "region_count": len(simultaneous),
                "simultaneous_speech_seconds": round(
                    sum(region["duration"] for region in simultaneous), 6),
                "min_overlap_seconds": args.min_overlap_seconds,
                "max_overlap_fraction": args.max_overlap_fraction,
                "removed_from_pooled_seconds": round(
                    sum(speaker["overlap_seconds_removed"] for speaker in speakers), 6),
                "removed_from_reference_seconds": round(
                    sum(speaker["reference_overlap_seconds_removed"] for speaker in speakers), 6),
            },
            "foreground_global_metrics": report.get("global_metrics"),
        },
        "evaluation": registry.get("evaluation"),
    }
    write_json(root / "registry.json", result)
    write_json(root / "speaker_turns.json", {"turns": turns})
    write_json(root / "local_speaker_tracklets.json", {"tracklets": tracklets})
    write_json(root / "global_speaker_timeline.json", {"segments": timeline})
    write_json(root / "overlap_regions.json", {"overlaps": overlaps})
    write_json(root / "simultaneous_speech_regions.json", {"regions": simultaneous})
    write_json(root / "unknown_turns.json", {"turns": unknown})
    write_json(root / "must_links.json", {"links": result["must_links"]})
    write_json(root / "possible_links.json", {"links": result["possible_links"]})
    write_json(root / "consolidation.json", {
        "schema_version": SCHEMA_VERSION, "pairs": pairs,
        "groups": [[pools[index].source_speaker_id for index in group] for group in groups],
        "pools": [{"speaker_id": pool.source_speaker_id, "prototype_seconds": pool.seconds,
                   "prototype_turn_ids": pool.turn_ids, "all_turn_ids": pool.all_turn_ids,
                   "short_turn_ids": pool.short_turn_ids, "short_seconds": pool.short_seconds,
                   "partial_turn_ids": pool.partial_turn_ids,
                   "partial_seconds": pool.partial_seconds,
                   "gate_rejected_turn_ids": pool.gate_rejected_turn_ids,
                   "has_embedding": pool.embedding is not None} for pool in pools],
        "short_turn_recovery": {speaker["speaker_id"]: speaker["short_turn_recovery"]
                                for speaker in speakers},
        "partial_overlap_recovery": {speaker["speaker_id"]: speaker["partial_overlap_recovery"]
                                     for speaker in speakers},
        "foreground_gate_audit": gate_audit,
    })
    write_json(root / "voice_asset_library.json", {
        "schema_version": SCHEMA_VERSION, "source_clip": report.get("input"),
        "audio_basis": "enhanced_foreground_stem", "sample_rate": SAMPLE_RATE,
        "speaker_count": len(speakers),
        "total_reference_audio": sum(len(s["reference_audio"]) for s in speakers),
        "total_pooled_clean_seconds": round(sum(s["pooled_seconds"] for s in speakers), 3),
        "overlap_exclusion": exclusions.status,
        "total_reference_overlap_seconds_removed": round(
            sum(s["reference_overlap_seconds_removed"] for s in speakers), 3),
        "insufficient_clean_voice_speaker_count": sum(
            bool(s.get("insufficient_clean_voice")) for s in speakers),
        "speakers": [{key: speaker[key] for key in
                      ("speaker_id", "source_speaker_ids", "local_tracklets", "turns",
                       "pooled_seconds", "turn_seconds", "overlap_seconds_removed",
                       "reference_overlap_seconds_removed", "pooled_clean_voice",
                       "reference_audio", "representative_audio", "reference_status",
                       "insufficient_clean_voice", "nearest_other_speaker", "reference_source",
                       "prototype", "embedding_count", "mean_turn_consistency",
                       "min_turn_consistency", "confidence", "short_turn_recovery",
                       "partial_overlap_recovery")}
                     for speaker in speakers],
    })
    return result


# --------------------------------------------------------------------------- compare

def summarize_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    speakers = registry.get("speakers") or []
    turns = registry.get("turns") or []
    usable = [turn for turn in turns if turn.get("quality_tier") != "rejected" and not turn.get("overlap")]
    turns_per_speaker = [len(speaker.get("turns") or []) for speaker in speakers]
    prototypes = []
    for speaker in speakers:
        path = speaker.get("prototype")
        if path and Path(path).is_file():
            prototypes.append((speaker["speaker_id"], np.load(path)))
    # One prototype per speaker, so every pair here is BETWEEN speakers.  Lower is better: it
    # means the registry is not describing the same voice twice under two ids.
    between = [round(float(np.dot(unit(left), unit(right))), 6)
               for left_id, left in prototypes for right_id, right in prototypes
               if left_id != right_id]
    overlap_exclusion = ((registry.get("metadata") or {}).get("overlap_exclusion") or {})
    return {
        "speaker_count": len(speakers),
        "singleton_speaker_count": sum(count == 1 for count in turns_per_speaker),
        "safe_reference_speaker_count": registry.get("safe_reference_speaker_count"),
        "reference_audio_count": registry.get("reference_audio_count"),
        "turn_count": len(turns),
        "usable_turn_count": len(usable),
        "unknown_turn_count": registry.get("unknown_turn_count"),
        "quality_stats": registry.get("quality_stats"),
        "turns_per_speaker": turns_per_speaker,
        # Null on any registry this pipeline did not produce, which is how a reader tells the two
        # zero totals below apart from a measured-and-clean run.
        "overlap_exclusion": overlap_exclusion.get("status"),
        "simultaneous_speech_region_count": len(registry.get("simultaneous_speech_regions") or []),
        "reference_overlap_seconds_removed": round(sum(
            float(speaker.get("reference_overlap_seconds_removed") or 0.0)
            for speaker in speakers), 6),
        "pooled_overlap_seconds_removed": round(sum(
            float(speaker.get("overlap_seconds_removed") or 0.0) for speaker in speakers), 6),
        "mean_between_speaker_prototype_cosine": round(float(np.mean(between)), 6) if between else None,
        "max_between_speaker_prototype_cosine": round(max(between), 6) if between else None,
    }


def compare_registries(named: Sequence[Tuple[str, Path]]) -> Dict[str, Any]:
    rows = {}
    for label, path in named:
        rows[label] = summarize_registry(json.loads(_resolve(path).read_text(encoding="utf-8")))
    return {"schema_version": SCHEMA_VERSION, "registries": rows}


# --------------------------------------------------------------------------- driver

def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = _resolve(args.output)
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{root} is not empty; pass --overwrite")
    root.mkdir(parents=True, exist_ok=True)

    report, report_path = stage_separate(args, root)
    stem, stem_path = load_stem(report)
    duration = len(stem) / float(SAMPLE_RATE)
    if abs(duration - float(report.get("duration") or duration)) > 0.5:
        raise SystemExit(f"stem is {duration:.3f}s but the report says {report.get('duration')}s")

    origin: Optional[np.ndarray] = None
    if args.disable_dual_path_verification:
        print("[dual-path] verification disabled", file=sys.stderr)
    else:
        origin = load_origin(report)
        if origin is None:
            print("[dual-path] origin mixture unavailable; candidates record "
                  "dual_path.status=unavailable", file=sys.stderr)

    registry = stage_build(args, stem_path, report_path, root)
    # srb.run already wrote stage5/registry.json; apply_to_registry mutates in place, so the
    # ungated tiers only survive if they are copied aside first.
    shutil.copy(root / "stage5" / "registry.json", root / "stage5" / "registry.pregate.json")
    registry, gate_audit = apply_to_registry(
        registry, report, report_path=report_path, min_coverage=args.min_coverage,
        max_contamination=args.max_contamination, prune_references=False)
    write_json(root / "stage5" / "registry.gate.json", gate_audit)
    write_json(root / "stage5" / "registry.json", registry)
    write_json(root / "stage5" / "speaker_turns.json", {"turns": registry.get("turns") or []})
    print(f"[gate] tier_change_count={gate_audit['tier_change_count']} "
          f"quality_stats={registry['quality_stats']}", file=sys.stderr)

    exclusions = stage_overlap(args, stem_path, duration, root)

    embedder = SpeakerEmbedder(_resolve(args.embedding_model))
    pools, gates = collect_pools(registry, report, stem, embedder, exclusions,
                                 max_pool_seconds=args.max_pool_seconds,
                                 min_pool_seconds=args.min_pool_seconds,
                                 min_coverage=args.min_coverage,
                                 max_contamination=args.max_contamination,
                                 max_overlap_fraction=args.max_overlap_fraction,
                                 min_turn_duration=args.min_turn_duration,
                                 min_recovered_segment_seconds=args.min_recovered_segment_seconds)
    turn_embeddings = embed_turns(registry, stem, embedder, exclusions, args.turn_embed_padding)
    groups, pairs = plan_merges(pools, turn_embeddings, registry,
                                threshold=args.pooled_merge_threshold,
                                possible_threshold=args.pooled_possible_threshold,
                                snorm_threshold=args.pooled_snorm_threshold,
                                use_snorm=args.use_snorm,
                                max_temporal_overlap=args.max_temporal_overlap,
                                min_speaker_clean_seconds=args.min_speaker_clean_seconds)

    consolidated = root / "consolidated"
    consolidated.mkdir(parents=True, exist_ok=True)
    speakers: List[Dict[str, Any]] = []
    for number, group in enumerate(groups, 1):
        competitors = [pools[other].embedding for other in range(len(pools))
                       if other not in group and pools[other].embedding is not None]
        speakers.append(build_voice_assets(group, pools, registry, gates, stem, embedder,
                                           f"SPK{number:03d}", consolidated, args, competitors,
                                           exclusions, origin=origin))
    attach_merge_evidence(speakers, pairs)
    insufficient = sum(bool(speaker["insufficient_clean_voice"]) for speaker in speakers)
    removed = round(sum(speaker["overlap_seconds_removed"] for speaker in speakers), 6)
    dropped = sum("excessive_simultaneous_speech" in (gate.get("excluded_reasons") or [])
                  for gate in gates.values())
    print(f"[consolidate] {len(pools)} stage5 speakers -> {len(groups)}; merged "
          f"{sum(pair['accepted'] for pair in pairs)}/{len(pairs)} pairs, "
          f"{sum(pair.get('possible') for pair in pairs)} ambiguous, "
          f"{insufficient} below the {args.min_speaker_clean_seconds}s evidence floor",
          file=sys.stderr)
    print(f"[overlap] status={exclusions.status} removed={removed}s from the pooled assets, "
          f"{dropped} turns dropped whole for exceeding "
          f"--max-overlap-fraction {args.max_overlap_fraction}", file=sys.stderr)
    result = assemble_registry(registry, groups, pools, speakers, pairs, stem_path, report,
                               report_path, args, consolidated, gate_audit, exclusions)

    comparison = compare_registries([("stage5_enhanced_stem", root / "stage5" / "registry.json"),
                                     ("consolidated", consolidated / "registry.json")]
                                    + ([(args.baseline_label, Path(args.baseline_registry))]
                                       if args.baseline_registry else []))
    write_json(root / "compare.json", comparison)

    # The registry already carries the full record; the summary repeats it without the verbose
    # `model` sub-block so stdout stays readable.
    overlap_summary = {key: value for key, value
                       in result["metadata"]["overlap_exclusion"].items() if key != "model"}
    overlap_summary["turns_dropped"] = dropped

    summary = {"status": "ok", "schema_version": SCHEMA_VERSION, "output": str(root),
               "duration": round(duration, 6),
               "foreground_seconds": report["activity"]["foreground_seconds"],
               "stage5_speaker_count": len(registry.get("speakers") or []),
               "consolidated_speaker_count": result["speaker_count"],
               "turn_count": result["turn_count"],
               "quality_stats": result["quality_stats"],
               "reference_audio_count": result["reference_audio_count"],
               "safe_reference_speaker_count": result["safe_reference_speaker_count"],
               "insufficient_clean_voice_speaker_count": insufficient,
               "overlap": overlap_summary,
               "accepted_merges": [pair for pair in pairs if pair["accepted"]],
               "possible_pairs": [{"left": pair["left"], "right": pair["right"],
                                   "pooled_cosine": pair["pooled_cosine"]}
                                  for pair in pairs if pair.get("possible")],
               "registry": str(consolidated / "registry.json"),
               "voice_asset_library": str(consolidated / "voice_asset_library.json"),
               "compare": str(root / "compare.json"),
               "comparison": comparison["registries"]}
    write_json(root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stage5 flags are inherited verbatim; --input is the video, --output the pipeline root.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add_help=False: Stage5's parser is inherited as a parent and already registers -h, and
    # argparse refuses to add the same option string twice.  Help still prints this subparser's
    # own usage, because _HelpAction formats through whichever parser is doing the parsing.
    run_parser = sub.add_parser("run", help="separate -> Stage5 on the clean stem -> consolidate",
                                parents=[srb.parser()], add_help=False)
    run_parser.add_argument("--clearvoice-root", type=Path, default=DEFAULT_CLEARVOICE_ROOT)
    run_parser.add_argument("--enhancer-model", default=DEFAULT_MODEL)
    run_parser.add_argument("--enhance-extract-sample-rate", type=int, default=48000)
    run_parser.add_argument("--enhance-extract-channels", type=int, default=1, choices=[1, 2])
    run_parser.add_argument("--enhance-overwrite", action="store_true")
    run_parser.add_argument("--ffmpeg")
    run_parser.add_argument("--max-pool-seconds", type=float, default=DEFAULT_MAX_POOL_SECONDS)
    run_parser.add_argument("--min-pool-seconds", type=float, default=DEFAULT_MIN_POOL_SECONDS)
    run_parser.add_argument("--turn-embed-padding", type=float, default=DEFAULT_TURN_EMBED_PADDING)
    run_parser.add_argument("--pooled-merge-threshold", type=float, default=DEFAULT_POOLED_MERGE_THRESHOLD)
    run_parser.add_argument("--pooled-possible-threshold", type=float,
                            default=DEFAULT_POOLED_POSSIBLE_THRESHOLD,
                            help="pooled cosines in [this, --pooled-merge-threshold) are reported as "
                                 "POOLED_POSSIBLE links rather than silently dropped")
    run_parser.add_argument("--min-speaker-clean-seconds", type=float,
                            default=DEFAULT_MIN_SPEAKER_CLEAN_SECONDS,
                            help="below this much total clean speech a speaker is still reported but "
                                 "gets no voice asset; short clips do not identify anyone")
    run_parser.add_argument("--short-reference-min-duration", type=float,
                            default=DEFAULT_SHORT_REFERENCE_MIN_DURATION,
                            help="duration floor for a reference recovered from a duration-only "
                                 "short turn, used only when no regular-length reference exists")
    run_parser.add_argument("--min-recovered-segment-seconds", type=float,
                            default=DEFAULT_MIN_RECOVERED_SEGMENT_SECONDS,
                            help="a turn rejected only for simultaneous speech is recoverable "
                                 "when its longest continuous single-speaker span reaches this")
    run_parser.add_argument("--pooled-snorm-threshold", type=float, default=2.0)
    run_parser.add_argument("--use-snorm", action="store_true",
                            help="decide merges on the cohort-normalised score instead of raw cosine")
    run_parser.add_argument("--max-temporal-overlap", type=float, default=DEFAULT_MAX_TEMPORAL_OVERLAP,
                            help="two speakers talking at the same time are never the same person")
    run_parser.add_argument("--overlap-report", type=Path,
                            help="reuse a speaker_overlap_detector report instead of running pyannote")
    run_parser.add_argument("--overlap-config", type=Path, default=sod.DEFAULT_PYANNOTE_CONFIG,
                            help="pyannote pipeline config; the weights it points at are read from "
                                 "disk, this stage never downloads anything")
    run_parser.add_argument("--overlap-device",
                            help="torch device for pyannote; defaults to cuda when present")
    run_parser.add_argument("--min-overlap-seconds", type=float,
                            default=sod.DEFAULT_MIN_OVERLAP_SECONDS,
                            help="ignore simultaneous speech shorter than this")
    run_parser.add_argument("--max-overlap-fraction", type=float,
                            default=DEFAULT_MAX_OVERLAP_FRACTION,
                            help="a turn with more than this share of simultaneous speech is "
                                 "dropped whole instead of shaved down to a fragment")
    run_parser.add_argument("--disable-overlap-exclusion", action="store_true",
                            help="cut audio on Stage5's turn boundaries alone, overlap and all")
    run_parser.add_argument("--disable-dual-path-verification", action="store_true",
                            help="skip embedding each reference candidate from the original "
                                 "mixture as well as from the enhanced stem")
    run_parser.add_argument("--dual-path-enforce", action="store_true",
                            help="reject a candidate when the stem path and the mixture path "
                                 "disagree on identity; by default the disagreement is only "
                                 "recorded (shadow mode)")
    run_parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    run_parser.add_argument("--max-contamination", type=float, default=DEFAULT_MAX_CONTAMINATION)
    run_parser.add_argument("--baseline-registry", type=Path,
                            help="an existing registry.json to compare against, e.g. a mixture-based run")
    run_parser.add_argument("--baseline-label", default="baseline")
    run_parser.add_argument("--overwrite", action="store_true")

    compare_parser = sub.add_parser("compare", help="summarise several registries side by side")
    compare_parser.add_argument("--registry", action="append", required=True, metavar="LABEL=PATH")
    compare_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        named = []
        for entry in args.registry:
            if "=" not in entry:
                raise SystemExit(f"--registry expects LABEL=PATH, got {entry!r}")
            label, path = entry.split("=", 1)
            named.append((label, Path(path)))
        payload = compare_registries(named)
        if args.output:
            write_json(_resolve(args.output), payload)
        print(json.dumps(payload["registries"], ensure_ascii=False, indent=2))
        return 0

    for name in ("min_coverage", "max_contamination", "pooled_merge_threshold",
                 "pooled_possible_threshold", "max_overlap_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be within [0, 1]")
    for name in ("max_pool_seconds", "min_pool_seconds", "turn_embed_padding",
                 "max_temporal_overlap", "min_speaker_clean_seconds", "min_overlap_seconds",
                 "short_reference_min_duration", "min_recovered_segment_seconds"):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must not be negative")
    if args.short_reference_min_duration > args.min_reference_duration:
        raise SystemExit("--short-reference-min-duration must not exceed --min-reference-duration")
    if args.min_pool_seconds > args.max_pool_seconds:
        raise SystemExit("--min-pool-seconds must not exceed --max-pool-seconds")
    if args.pooled_possible_threshold > args.pooled_merge_threshold:
        raise SystemExit("--pooled-possible-threshold must not exceed --pooled-merge-threshold")
    if args.dual_path_enforce and args.disable_dual_path_verification:
        raise SystemExit("--dual-path-enforce is meaningless with --disable-dual-path-verification")
    summary = run(args)
    print(json.dumps({key: summary[key] for key in
                      ("status", "duration", "stage5_speaker_count", "consolidated_speaker_count",
                       "insufficient_clean_voice_speaker_count", "turn_count", "quality_stats",
                       "reference_audio_count", "safe_reference_speaker_count", "overlap",
                       "possible_pairs", "registry", "voice_asset_library")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
