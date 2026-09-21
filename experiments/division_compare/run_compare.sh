#!/usr/bin/env bash
# MiMo-division vs sliding-window-division comparison on a single video.
#
# Both chains share an identical base so the ONLY variable is the segment-level
# division semantics:
#
#   preprocessed 480p/15fps video            (unified downsampling — same
#                                             normalize_video.py stage the
#                                             main pipeline runs)
#     └─ shot_splitter (TransNetV2 / adaptive)        ─ shared
#         └─ global_clip_aggregator (SCRL / metadata) ─ shared
#             └─ mimo_global_captioner_speaker_v2     ─ shared (ASR optional)
#                 └─ mimo_shot_captioner_v3           ─ shared
#                     ├─ chain A (MiMo division):
#                     │    mimo_segment_captioner_v2  (MiMo plans boundaries)
#                     │    mimo_segment_cutter        (export segment clips)
#                     └─ chain B (window division):
#                          segmentation/three_way --impls window (duration
#                          greedy, no semantics)
#                          mimo_segment_captioner_fixed_segments_v4 (captions
#                          for the fixed windows)
#
# Usage (server, repo root as cwd):
#   bash experiments/division_compare/run_compare.sh <video.mp4> \
#        [--asr <asr.jsonl>] [--out <run_dir>]
#
# Optional env overrides:
#   PY FFMPEG NORM_HEIGHT NORM_FPS SHOT_DETECTOR MIN_GLOBAL_SEC TARGET_GLOBAL_SEC
#   MAX_GLOBAL_SEC CONCURRENCY TRANSNET_MODEL TRANSNET_SOURCE_DIR SCRL_ENCODER
#   SCRL_BOUNDARY SCRL_SOURCE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"
FFMPEG="${FFMPEG:-ffmpeg}"

VIDEO=""
ASR=""
RUN_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr) ASR="$2"; shift 2 ;;
    --out) RUN_ROOT="$2"; shift 2 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) VIDEO="$1"; shift ;;
  esac
done
[[ -n "$VIDEO" && -f "$VIDEO" ]] || { echo "usage: $0 <video.mp4> [--asr a.jsonl] [--out dir]" >&2; exit 2; }

STEM="$(basename "$VIDEO")"
STEM="${STEM%.*}"
RUN_ROOT="${RUN_ROOT:-$ROOT/experiments/division_compare/runs/${STEM}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/logs"
TIMINGS="$RUN_ROOT/timings.tsv"
echo -e "stage\tseconds" > "$TIMINGS"

log()  { echo "[compare] $*"; }

run_stage() {  # run_stage <name> <done_marker> <cmd...>
  local name="$1" marker="$2"; shift 2
  if [[ -e "$marker" ]]; then log "skip $name (exists: $marker)"; return 0; fi
  local t0=$SECONDS
  log "run  $name"
  if ! "$@" > "$RUN_ROOT/logs/$name.log" 2>&1; then
    echo "[compare] stage $name FAILED — see $RUN_ROOT/logs/$name.log" >&2
    tail -n 20 "$RUN_ROOT/logs/$name.log" >&2 || true
    exit 1
  fi
  echo -e "$name\t$((SECONDS - t0))" >> "$TIMINGS"
}

# ---------------------------------------------------------------- step 0
# Unified downsampling: 480p, 15 fps (NORM_HEIGHT / NORM_FPS). Everything
# downstream consumes ONLY this file, so both chains see identical frames and
# the MiMo payload shrink is controlled for. normalize_video.py also writes
# "$NORM.info.json" with probed input/output metadata and the exact ffmpeg
# command line it ran.
NORM="$RUN_ROOT/preprocessed.mp4"
run_stage preprocess "$NORM" \
  "$PY" "$ROOT/preprocessing/normalize_video.py" \
  --input "$VIDEO" --output "$NORM" \
  --height "${NORM_HEIGHT:-480}" --fps "${NORM_FPS:-15}" \
  --ffmpeg "$FFMPEG"

SHOTS_DIR="$RUN_ROOT/shots"
GLOBAL_DIR="$RUN_ROOT/global"
CAP_GLOBAL="$RUN_ROOT/captions/global"
CAP_SHOT="$RUN_ROOT/captions/shot"

# ---------------------------------------------------------------- shared base
run_stage split_shots "$SHOTS_DIR/shots.jsonl" \
  "$PY" "$ROOT/shot_splitter.py" \
  --input "$NORM" --output-dir "$SHOTS_DIR" \
  --detector "${SHOT_DETECTOR:-auto}" --flat \
  $( [[ -n "${TRANSNET_MODEL:-}" ]] && echo --transnet-model "$TRANSNET_MODEL" ) \
  $( [[ -n "${TRANSNET_SOURCE_DIR:-}" ]] && echo --transnet-source-dir "$TRANSNET_SOURCE_DIR" )

run_stage aggregate_global "$GLOBAL_DIR/global_clips.jsonl" \
  "$PY" "$ROOT/global_clip_aggregator.py" \
  --input "$SHOTS_DIR" --output-dir "$GLOBAL_DIR" \
  --min-sec "${MIN_GLOBAL_SEC:-60}" --target-sec "${TARGET_GLOBAL_SEC:-90}" \
  --max-sec "${MAX_GLOBAL_SEC:-120}" --flat \
  $( [[ -n "${SCRL_ENCODER:-}" ]] && echo --scrl-encoder "$SCRL_ENCODER" ) \
  $( [[ -n "${SCRL_BOUNDARY:-}" ]] && echo --scrl-boundary "$SCRL_BOUNDARY" ) \
  $( [[ -n "${SCRL_SOURCE_DIR:-}" ]] && echo --scrl-source-dir "$SCRL_SOURCE_DIR" )

ASR_ARGS=()
[[ -n "$ASR" ]] && ASR_ARGS=(--asr-input "$ASR")

run_stage global_caption "$CAP_GLOBAL/global_captions.jsonl" \
  "$PY" "$ROOT/mimo_global_captioner_speaker_v2.py" \
  --input "$GLOBAL_DIR" --output-dir "$CAP_GLOBAL" \
  "${ASR_ARGS[@]}" --concurrency "${CONCURRENCY:-2}" --flat

run_stage shot_caption "$CAP_SHOT/shot_captions.jsonl" \
  "$PY" "$ROOT/mimo_shot_captioner_v3.py" \
  --input "$SHOTS_DIR" --global-input "$CAP_GLOBAL" --output-dir "$CAP_SHOT" \
  --concurrency "${CONCURRENCY:-2}" --flat

# ---------------------------------------------------------------- chain A: MiMo division
A_CAP="$RUN_ROOT/mimo_division/segment_captions"
A_SEG="$RUN_ROOT/mimo_division/segments"
run_stage mimo_segment_caption "$A_CAP/segment_captions.jsonl" \
  "$PY" "$ROOT/mimo_segment_captioner_v2.py" \
  --global-input "$CAP_GLOBAL" --shot-input "$CAP_SHOT" \
  --output-dir "$A_CAP" --concurrency "${CONCURRENCY:-2}" --flat

run_stage mimo_segment_cut "$A_SEG/segments.jsonl" \
  "$PY" "$ROOT/mimo_segment_cutter.py" \
  --input "$A_CAP" --global-input "$GLOBAL_DIR" \
  --output-dir "$A_SEG" --flat

# ---------------------------------------------------------------- chain B: window division
B_PLAN="$RUN_ROOT/window_division/segment_plan"
B_CAP="$RUN_ROOT/window_division/segment_captions"
run_stage window_segment_plan "$B_PLAN/segments_window/segments.jsonl" \
  "$PY" "$ROOT/segmentation/three_way.py" \
  --global-clips "$GLOBAL_DIR/global_clips.jsonl" \
  --shots-jsonl "$SHOTS_DIR/shots.jsonl" \
  --out-dir "$B_PLAN" --impls window

run_stage window_segment_caption "$B_CAP/segment_captions.jsonl" \
  "$PY" "$ROOT/mimo_segment_captioner_fixed_segments_v4.py" \
  --global-input "$CAP_GLOBAL" --shot-input "$CAP_SHOT" \
  --segment-input "$B_PLAN/segments_window/segments.jsonl" \
  --output-dir "$B_CAP" --concurrency "${CONCURRENCY:-2}" --flat

# ---------------------------------------------------------------- report
run_stage report "$RUN_ROOT/comparison_report.md" \
  "$PY" "$ROOT/experiments/division_compare/report_compare.py" \
  --run-dir "$RUN_ROOT" --video-stem "$STEM"

log "done -> $RUN_ROOT/comparison_report.md"
