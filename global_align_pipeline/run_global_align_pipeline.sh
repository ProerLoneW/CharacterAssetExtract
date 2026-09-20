#!/usr/bin/env bash
# Global-scope tracklet → character alignment test run.
# Reuses the v4 workspace's tracklets, captions and voice previews; writes
# everything into a NEW result folder. Nothing in v4 is modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_WS="${SRC_WS:-$ROOT/movie_full_pipeline_20260912_v4}"
OUT_WS="${OUT_WS:-$ROOT/movie_full_pipeline_20260915_global_align_test}"
PY="${PY:-python}"

mkdir -p "$OUT_WS"

# Step 1: stitch per-shot tracklet overlays back into one video per global
# clip, then let MiMo assign every tracklet directly to an S-ID at global
# scale. --dry-run first is a cheap sanity check of stitching + packaging.
"$PY" "$ROOT/global_align_pipeline/global_tracklet_aligner.py" \
  --tracklets "$SRC_WS/assets/visual_person_v2/tracklets" \
  --global-manifest "$SRC_WS/middle/global_clips.jsonl" \
  --shots "$SRC_WS/middle/shots.jsonl" \
  --global-captions "$SRC_WS/captions_speaker/global" \
  --shot-captions "$SRC_WS/captions_speaker/shot" \
  --segment-captions "$SRC_WS/captions_speaker/segment" \
  --output "$OUT_WS/global_alignment" \
  --concurrency 2

# Step 2: materialise characters/<scope>/<S-ID>/{visual,voice}
# --identity-check reruns Stage 3's InsightFace pack on the gallery faces and
# drops face crops of tracklets that do not match the character's face core
# (e.g. a bystander MiMo absorbed into an S-ID). Fail-open: if the model is
# unavailable the check is skipped, not fatal.
"$PY" "$ROOT/global_align_pipeline/global_character_materializer.py" \
  --alignment "$OUT_WS/global_alignment/global_tracklet_alignment.jsonl" \
  --tracklets "$SRC_WS/assets/visual_person_v2/tracklets" \
  --voice-root "$SRC_WS/voice_by_role_src" \
  --output "$OUT_WS" \
  --face-top-k 8 --body-top-k 8 \
  --identity-check \
  --face-model "$ROOT/local_models/insightface/models/buffalo_l/w600k_r50.onnx"

echo "done -> $OUT_WS"
