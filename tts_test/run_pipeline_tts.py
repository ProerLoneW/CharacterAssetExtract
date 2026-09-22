#!/usr/bin/env python3
"""Run the local VoxCPM2/Fish-S2-Pro comparison on mainline voice assets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError):
        return 0.0


def _discover(workspace: Path, limit: int, minimum_seconds: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role_path in sorted((workspace / "characters").glob("*/*/role_character.json")):
        payload = json.loads(role_path.read_text(encoding="utf-8"))
        scope = str(payload.get("semantic_scope_id") or role_path.parents[1].name)
        subject = str(payload.get("subject_id") or role_path.parent.name)
        ref_dir = role_path.parent / "voice" / "valid"
        clips = sorted(ref_dir.glob("*.wav")) if ref_dir.is_dir() else []
        duration = sum(_wav_seconds(path) for path in clips)
        if not clips or duration < minimum_seconds:
            continue
        rows.append({
            "scope": scope,
            "subject": subject,
            "speaker_key": f"{scope}_{subject}",
            "ref_dir": ref_dir,
            "clip_count": len(clips),
            "reference_seconds": round(duration, 3),
            "tracklet_count": int(payload.get("tracklet_count") or 0),
        })
    rows.sort(key=lambda row: (row["reference_seconds"], row["tracklet_count"],
                               row["speaker_key"]), reverse=True)
    return rows if limit == 0 else rows[:limit]


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_command(model: str, row: dict[str, Any], workspace: Path,
                   output_root: Path, candidates: int) -> tuple[list[str], Path]:
    transcripts = workspace / "captions" / "global" / "speaker_attributed_asr.jsonl"
    out_dir = output_root / model / row["speaker_key"]
    if model == "voxcpm2":
        command = [
            str(HERE / ".venvs" / "voxcpm" / "bin" / "python"),
            str(HERE / "tts_voxcpm2.py"),
            "--ref-dir", str(row["ref_dir"]),
            "--model-dir", str(HERE / "local_models" / "VoxCPM2"),
            "--out-dir", str(out_dir),
            "--num-candidates", str(candidates),
        ]
    elif model == "s2pro":
        command = [
            str(HERE / ".venvs" / "fish" / "bin" / "python"),
            str(HERE / "tts_fish_s2pro.py"),
            "--ref-dir", str(row["ref_dir"]),
            "--model-dir", str(HERE / "local_models" / "s2-pro"),
            "--out-dir", str(out_dir),
            "--num-candidates", str(candidates),
        ]
    else:  # guarded by argparse
        raise ValueError(model)
    if transcripts.is_file():
        command += ["--transcripts", str(transcripts)]
    return command, out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=("voxcpm2", "s2pro"),
                        default=["voxcpm2", "s2pro"])
    parser.add_argument("--max-speakers", type=int, default=3, help="0 means all")
    parser.add_argument("--min-reference-seconds", type=float, default=5.0)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_speakers < 0 or args.min_reference_seconds < 0 or args.num_candidates < 1:
        parser.error("invalid speaker/duration/candidate limit")

    workspace = args.workspace.expanduser().resolve()
    root = (args.output_root or workspace / "tts_test").expanduser().resolve()
    output_root = root / "outputs"
    speakers = _discover(workspace, args.max_speakers, args.min_reference_seconds)
    plans: list[dict[str, Any]] = []
    for speaker in speakers:
        for model in args.models:
            command, out_dir = _model_command(
                model, speaker, workspace, output_root, args.num_candidates)
            plans.append({**speaker, "model": model, "command": command,
                          "out_dir": out_dir})
    if args.dry_run:
        print(json.dumps({
            "workspace": str(workspace),
            "tasks": [{k: (str(v) if isinstance(v, Path) else v)
                       for k, v in row.items() if k != "command"}
                      | {"command": [str(v) for v in row["command"]]}
                      for row in plans],
        }, ensure_ascii=False, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    failures = 0
    for row in plans:
        meta = row["out_dir"] / "meta.json"
        result = {
            "model": row["model"], "scope": row["scope"],
            "subject": row["subject"], "speaker_key": row["speaker_key"],
            "ref_dir": str(row["ref_dir"]), "clip_count": row["clip_count"],
            "reference_seconds": row["reference_seconds"], "out_dir": str(row["out_dir"]),
        }
        if meta.is_file() and not args.overwrite:
            result["status"] = "skipped_existing"
            results.append(result)
            continue
        row["out_dir"].mkdir(parents=True, exist_ok=True)
        log_path = row["out_dir"] / "run.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    row["command"], cwd=str(HERE), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.task_timeout, check=False,
                )
            if completed.returncode != 0 or not meta.is_file():
                raise RuntimeError(
                    f"exit={completed.returncode}, meta_exists={meta.is_file()} (see {log_path})")
            result["status"] = "succeeded"
            print(f"[tts] {row['model']} {row['speaker_key']} -> {row['out_dir']}", flush=True)
        except Exception as exc:  # independent model/speaker boundary
            failures += 1
            result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:2000]})
            print(f"[tts] failed {row['model']} {row['speaker_key']}: {result['error']}",
                  file=sys.stderr, flush=True)
        results.append(result)

    manifest = {
        "schema_version": "pipeline-tts-review-v1",
        "workspace": str(workspace),
        "speaker_count": len(speakers),
        "task_count": len(results),
        "succeeded_count": sum(row["status"] in {"succeeded", "skipped_existing"} for row in results),
        "failed_count": failures,
        "tasks": results,
    }
    _write_manifest(root / "manifest.json", manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
