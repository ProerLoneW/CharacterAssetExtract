#!/usr/bin/env python3
"""Turn-level foreground gate for Stage 5, kept outside ``speaker_registry_builder.py``.

``audio_foreground_separator.py`` answers "is there enhanced human voice here, and how
much background bled through".  This module turns that answer into Stage-5 tier
decisions.  It never modifies ``speaker_registry_builder.py``; it runs either side of it.

``prefilter``  BEFORE Stage5 -- drop diarization segments the enhancement model says
               contain no speech.  A dropped segment never becomes a turn, so it can
               never contribute an embedding, seed a global cluster, or be cut into a
               reference clip.  This is the only non-invasive way to reproduce what an
               in-builder gate would do, because the builder computes the embedding
               inside the same loop that assigns the tier: by the time a registry exists
               on disk, a false-positive turn is already baked into a speaker prototype.

``apply``      AFTER Stage5 -- re-tier an existing registry and record the evidence.
               Turns stay visible as ``rejected`` instead of disappearing, which is what
               you want for auditing.  It cannot retroactively withdraw an embedding, so
               it also emits an audit naming exactly the speakers whose prototype is
               carrying gated-out audio.

Both modes share one rule set:

    no enhanced speech at all     -> tier becomes ``rejected``
    contamination above the cap   -> ``high`` demoted to ``medium``

``medium`` is not cosmetic.  ``associate_global_speakers`` seeds global clusters only
from ``high`` turns, so a demotion means "may attach to a speaker, may not found one".

The gate never touches ``quality_score``.  That score feeds ``gallery_weights`` and the
weighted cluster prototype, so moving it would silently shift speaker identities.  The
tier is the field that already exists for admitting or excluding evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from audio_foreground_separator import SCHEMA_VERSION as FOREGROUND_SCHEMA_VERSION
from audio_foreground_separator import foreground_gate_for_turn

SCHEMA_VERSION = "speaker-turn-foreground-gate-v1"

# Looser than the reference gate in speaker_registry_builder (--min-reference-foreground-coverage
# defaults to 0.5): a turn only needs enough enhanced speech to prove it is real, whereas a
# reference clip has to be clean enough to identify someone by.
DEFAULT_MIN_COVERAGE = 0.3
# Not empirically calibrated.  The region-level leakage bracket measured so far is 0.481-0.491
# (migration doc section 5.3), the pre-existing reference cap is 0.5, and the highest turn-level
# value seen is 0.4403 -- so 0.6 has never fired on real material.  It is set above all of those
# on purpose: this demotion should only trigger on a turn that is clearly mostly background.
DEFAULT_MAX_CONTAMINATION = 0.6
DURATION_TOLERANCE_SECONDS = 0.5


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _guard_overwrite(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists() and not overwrite]
    if existing:
        raise SystemExit(f"refusing to overwrite {', '.join(existing)}; pass --overwrite")


def load_report(path: Path) -> Dict[str, Any]:
    report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if report.get("schema_version") != FOREGROUND_SCHEMA_VERSION:
        raise SystemExit(f"{path}: expected schema {FOREGROUND_SCHEMA_VERSION}, "
                         f"got {report.get('schema_version')!r}")
    if report.get("status") != "HAS_SPEECH":
        # A clip with no enhanced speech at all would reject every turn.  That is a legitimate
        # outcome but almost always means the wrong report was passed, so make it explicit.
        print(f"warning: {path} reports status={report.get('status')!r}; every turn will be "
              f"rejected", file=sys.stderr)
    return report


def _check_duration(report: Dict[str, Any], duration: Optional[float]) -> None:
    expected = report.get("duration")
    if duration is None or expected is None:
        return
    if abs(float(expected) - float(duration)) > DURATION_TOLERANCE_SECONDS:
        raise SystemExit(f"foreground report covers {float(expected):.3f}s but the Stage5 "
                         f"timeline is {float(duration):.3f}s; they are not the same clip")


def classify_tier(tier: Optional[str], gate: Dict[str, Any],
                  max_contamination: float) -> Tuple[str, List[str]]:
    """Apply the gate rules to one tier.  Returns the new tier and why it moved."""
    if not gate.get("has_foreground_speech", True):
        return "rejected", ["no_foreground_speech"]
    contamination = gate.get("music_contamination")
    if contamination is not None and contamination > max_contamination and tier == "high":
        return "medium", ["excessive_music_contamination"]
    return tier or "rejected", []


def _quality_fields(gate: Dict[str, Any]) -> Dict[str, float]:
    contamination = gate.get("music_contamination")
    coverage = gate.get("foreground_coverage")
    return {
        # -1.0 is the sentinel speaker_registry_builder.turn_quality already writes when it has
        # no foreground evidence, so downstream readers keep one meaning for "unknown".
        "music_contamination": round(float(contamination), 6) if contamination is not None else -1.0,
        "foreground_coverage": round(float(coverage), 6) if coverage is not None else -1.0,
        "no_foreground_speech": 1.0 if not gate.get("has_foreground_speech", True) else 0.0,
    }


def apply_to_registry(registry: Dict[str, Any], report: Dict[str, Any], *,
                      report_path: Optional[Path] = None,
                      min_coverage: float = DEFAULT_MIN_COVERAGE,
                      max_contamination: float = DEFAULT_MAX_CONTAMINATION,
                      prune_references: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Re-tier a finished Stage5 registry in place and return it with an audit report."""
    if (registry.get("metadata") or {}).get("foreground_turn_gate"):
        # Idempotent for tiers, but the audit degrades: original_tier is now the already-gated
        # tier, so `changes` comes back empty and the gallery audits can no longer see anything.
        print("warning: this registry was already gated; the audit will report changes relative "
              "to the gated tiers, not the ones Stage5 produced. Re-run from the pre-gate "
              "registry if you need the original provenance.", file=sys.stderr)
    _check_duration(report, (registry.get("timeline") or {}).get("duration"))
    turns: List[Dict[str, Any]] = registry.get("turns") or []
    rows: List[Dict[str, Any]] = []
    # A turn reached the gallery iff it was not overlap and its BUILD-TIME tier was not
    # "rejected" (see extract_tracklet_evidence).  Capture that before re-tiering, otherwise
    # the audit would be measuring the gate's own output.
    gallery_by_speaker: Dict[str, List[str]] = {}
    gated_out_of_gallery: Dict[str, List[str]] = {}
    demoted_in_gallery: Dict[str, List[str]] = {}

    for turn in turns:
        original_tier = turn.get("quality_tier")
        gate = foreground_gate_for_turn(report, turn["start"], turn["end"], min_coverage)
        new_tier, reasons = classify_tier(original_tier, gate, max_contamination)
        turn.setdefault("quality", {}).update(_quality_fields(gate))
        turn["quality_tier"] = new_tier
        speaker_id = turn.get("speaker_id")
        # speaker_id is the right proxy for "inside a prototype": an unassigned turn still sits
        # in its local tracklet gallery, but a tracklet that never joined a global cluster
        # contributes nothing to any speaker prototype, so it cannot pollute an identity.
        in_gallery = speaker_id is not None and not turn.get("overlap") and original_tier != "rejected"
        if in_gallery:
            gallery_by_speaker.setdefault(speaker_id, []).append(turn["turn_id"])
            if new_tier == "rejected":
                gated_out_of_gallery.setdefault(speaker_id, []).append(turn["turn_id"])
            elif new_tier == "medium" and original_tier == "high":
                demoted_in_gallery.setdefault(speaker_id, []).append(turn["turn_id"])
        rows.append({
            "turn_id": turn.get("turn_id"), "speaker_id": speaker_id,
            "start": turn.get("start"), "end": turn.get("end"),
            "original_tier": original_tier, "gated_tier": new_tier,
            "changed": new_tier != original_tier, "reasons": reasons,
            "reached_speaker_gallery": in_gallery, **gate,
        })

    registry["quality_stats"] = {tier: sum(t["quality_tier"] == tier for t in turns)
                                 for tier in ("high", "medium", "rejected")}
    registry["rejected_asset_turn_count"] = sum(t["quality_tier"] == "rejected" for t in turns)

    pruned: List[Dict[str, Any]] = []
    if prune_references:
        rejected_ids = {row["turn_id"] for row in rows if row["gated_tier"] == "rejected"}
        for speaker in registry.get("speakers") or []:
            keep, dropped = [], []
            for candidate in speaker.get("reference_candidates") or []:
                if candidate.get("status") == "ADMITTED" and candidate.get("turn_id") in rejected_ids:
                    dropped.append(candidate.get("turn_id"))
                elif candidate.get("audio_path"):
                    keep.append(candidate["audio_path"])
            if not dropped:
                continue
            speaker["reference_audio"] = keep
            speaker["representative_audio"] = keep
            speaker["reference_status"] = "READY" if keep else "NO_SAFE_REFERENCE"
            pruned.append({"speaker_id": speaker.get("speaker_id"), "dropped_turn_ids": dropped,
                           "remaining_reference_count": len(keep)})
        registry["reference_audio_count"] = sum(len(s.get("reference_audio") or [])
                                                for s in registry.get("speakers") or [])
        registry["safe_reference_speaker_count"] = sum(bool(s.get("reference_audio"))
                                                       for s in registry.get("speakers") or [])

    metadata = registry.setdefault("metadata", {})
    metadata["foreground_turn_gate"] = {
        "enabled": True,
        "mode": "apply",
        "min_coverage": min_coverage,
        "max_music_contamination": max_contamination,
        "prune_references": prune_references,
        "turns_without_foreground_speech": sum(not r["has_foreground_speech"] for r in rows),
        "tier_change_count": sum(r["changed"] for r in rows),
        "foreground_report": str(Path(report_path).expanduser().resolve()) if report_path else None,
    }

    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": "apply",
        "foreground_report": str(Path(report_path).expanduser().resolve()) if report_path else None,
        "source_clip": report.get("input"),
        "parameters": {"min_coverage": min_coverage, "max_music_contamination": max_contamination,
                       "prune_references": prune_references},
        "gate_parameters": report.get("gate_parameters"),
        "turn_count": len(rows),
        "tier_change_count": sum(r["changed"] for r in rows),
        "changes": [r for r in rows if r["changed"]],
        "turns": rows,
        # A non-empty gated_out_of_gallery means the speaker prototype on disk was built with
        # audio this gate rejects.  Re-run Stage5 through `prefilter` to actually remove it;
        # `apply` can only report it.
        "speakers_with_contaminated_prototype": [
            {"speaker_id": speaker_id,
             "gallery_turn_count": len(gallery_by_speaker.get(speaker_id, [])),
             "gated_out_turn_ids": gated_out_of_gallery[speaker_id]}
            for speaker_id in sorted(gated_out_of_gallery)
        ],
        # Clustering already ran before `apply`, so demoting high -> medium cannot retract a
        # seed the way it would inside the builder.  This list is the only trace of that: a
        # speaker here was founded on evidence the gate considers too contaminated to seed with.
        "speakers_seeded_from_contaminated_turn": [
            {"speaker_id": speaker_id,
             "gallery_turn_count": len(gallery_by_speaker.get(speaker_id, [])),
             "demoted_turn_ids": demoted_in_gallery[speaker_id],
             "sole_gallery_evidence": len(gallery_by_speaker.get(speaker_id, []))
             == len(demoted_in_gallery[speaker_id])}
            for speaker_id in sorted(demoted_in_gallery)
        ],
        "pruned_references": pruned,
    }
    return registry, audit


def prefilter_diarization(diarization: Any, report: Dict[str, Any], duration: Optional[float], *,
                          min_coverage: float = DEFAULT_MIN_COVERAGE) -> Dict[str, Any]:
    """Drop diarization segments with no enhanced speech, before Stage5 ever sees them.

    Only rule 1 applies here.  A heavily contaminated segment is still real speech, so it is
    kept -- dropping it would lose content.  What gets removed is segments that are entirely
    room tone or music bed, which is what CAM++ turns into phantom singleton speakers.
    """
    # Reuse the builder's own normalizer so every input shape it accepts stays accepted here,
    # including {"segments": ...}, {"diarization": ...} and active_speakers lists.
    from speaker_registry_builder import _seconds_if_needed, normalize_diarization

    rows = _seconds_if_needed(normalize_diarization(diarization), duration or 0.0)
    _check_duration(report, duration)
    kept: List[List[Any]] = []
    dropped: List[Dict[str, Any]] = []
    for start, end, label in rows:
        gate = foreground_gate_for_turn(report, start, end, min_coverage)
        if not gate["has_foreground_speech"]:
            dropped.append({"start": round(start, 6), "end": round(end, 6), "label": label,
                            "reason": "no_foreground_speech", **gate})
        else:
            kept.append([round(start, 6), round(end, 6), label])
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "prefilter",
        "parameters": {"min_coverage": min_coverage},
        "segments": kept,
        "prefilter": {
            "input_segment_count": len(rows),
            "kept_segment_count": len(kept),
            "dropped_segment_count": len(dropped),
            "dropped_seconds": round(sum(r["end"] - r["start"] for r in dropped), 6),
            "dropped": dropped,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="prefilter runs before speaker_registry_builder.py, apply runs after it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--foreground-report", required=True,
                            help="audio_foreground_separator foreground_report.json")
        target.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                            help="fraction of a turn that must be backed by enhanced speech")
        target.add_argument("--overwrite", action="store_true")

    pre = sub.add_parser("prefilter", help="drop speechless diarization segments before Stage5")
    pre.add_argument("--diarization-json", required=True, help="diarization to filter")
    pre.add_argument("--duration", type=float, help="clip duration in seconds; used to reject a "
                                                    "report that belongs to a different clip")
    pre.add_argument("--output", required=True, help="filtered diarization, feed to Stage5 "
                                                     "--diarization-json")
    add_common(pre)

    app = sub.add_parser("apply", help="re-tier an existing Stage5 registry")
    app.add_argument("--registry", required=True, help="Stage5 registry.json")
    app.add_argument("--output", required=True, help="where to write the gated registry")
    app.add_argument("--report", help="gate audit json; defaults to <output>.gate.json")
    app.add_argument("--max-contamination", type=float, default=DEFAULT_MAX_CONTAMINATION,
                     help="above this background energy ratio a high turn is demoted to medium; "
                          "note this cannot retract a cluster seed that already exists")
    app.add_argument("--prune-references", action="store_true",
                     help="also drop reference clips whose source turn was gated to rejected")
    add_common(app)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("min_coverage", "max_contamination"):
        value = getattr(args, name, None)
        if value is not None and not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be within [0, 1]")
    foreground_path = Path(args.foreground_report).expanduser()
    report = load_report(foreground_path)

    if args.command == "prefilter":
        source = Path(args.diarization_json).expanduser()
        output = Path(args.output).expanduser()
        _guard_overwrite([output], args.overwrite)
        payload = prefilter_diarization(
            json.loads(source.read_text(encoding="utf-8")), report, args.duration,
            min_coverage=args.min_coverage)
        _atomic_json(output, payload)
        summary = {k: payload["prefilter"][k] for k in
                   ("input_segment_count", "kept_segment_count", "dropped_segment_count",
                    "dropped_seconds")}
        print(json.dumps({"status": "ok", "mode": "prefilter", "output": str(output), **summary},
                         ensure_ascii=False))
        return 0

    registry_path = Path(args.registry).expanduser()
    output = Path(args.output).expanduser()
    audit_path = Path(args.report).expanduser() if args.report else output.with_suffix(".gate.json")
    # registry.json and speaker_turns.json hold the same turn rows in a Stage5 output directory;
    # rewriting one without the other leaves the two disagreeing about quality_tier.
    sidecar = output.parent / "speaker_turns.json"
    targets = [output, audit_path] + ([sidecar] if sidecar.exists() else [])
    _guard_overwrite(targets, args.overwrite)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    gated, audit = apply_to_registry(
        registry, report, report_path=foreground_path,
        min_coverage=args.min_coverage,
        max_contamination=args.max_contamination, prune_references=args.prune_references)
    _atomic_json(output, gated)
    _atomic_json(audit_path, audit)
    if sidecar.exists():
        _atomic_json(sidecar, {"turns": gated.get("turns") or []})
    print(json.dumps({
        "status": "ok", "mode": "apply", "output": str(output), "report": str(audit_path),
        "sidecar_updated": str(sidecar) if sidecar.exists() else None,
        "turn_count": audit["turn_count"], "tier_change_count": audit["tier_change_count"],
        "quality_stats": gated["quality_stats"],
        "speakers_with_contaminated_prototype": len(audit["speakers_with_contaminated_prototype"]),
        "speakers_seeded_from_contaminated_turn": [
            row["speaker_id"] for row in audit["speakers_seeded_from_contaminated_turn"]],
        "changes": [{"turn_id": r["turn_id"], "from": r["original_tier"], "to": r["gated_tier"],
                     "reasons": r["reasons"]} for r in audit["changes"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
