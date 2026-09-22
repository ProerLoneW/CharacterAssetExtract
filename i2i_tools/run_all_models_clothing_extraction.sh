#!/usr/bin/env bash
# Run single-stage and two-stage clothing extraction across all 5 supported
# models (gemini-3-pro-image, gemini-3.1-flash-image,
# gemini-3.1-flash-lite-image, gpt-image-2, gpt-image-2.5-sunburst).
#
# This script assumes:
#   1. `UNIAPI_API_KEY` is exported in the calling shell.
#   2. You're cd'd into the project root
#      (general_video_annotation_level3_v2/).
#   3. The three full-body crops already exist under
#      workspace/i2i_test/.
#
# Outputs go to:
#   workspace/i2i_test_single_stage/<model>/CHR004_top.png
#   workspace/i2i_test_single_stage/<model>/CHR004_pants.png
#   workspace/i2i_test_two_stage/<model>/CHR004_{three_view,top,pants}.png

set -euo pipefail

cd "$(dirname "$0")/.."  # i2i_tools/run_*.sh -> project root

CROPS=(workspace/i2i_test/CHR004_BODY_0007.jpg
       workspace/i2i_test/CHR004_BODY_0009.jpg
       workspace/i2i_test/CHR004_BODY_0010.jpg)

# Single-stage clothing extraction (one model per invocation).
SINGLE_ROOT=workspace/i2i_test_single_stage
mkdir -p "$SINGLE_ROOT"

for model in gemini-3-pro-image gemini-3.1-flash-image gemini-3.1-flash-lite-image; do
    out_dir="$SINGLE_ROOT/$model"
    mkdir -p "$out_dir"
    echo "==> Single stage: $model"
    python i2i_tools/extract_character_clothing.py \
        --provider gemini --model "$model" \
        --images "${CROPS[@]}" \
        --output "$out_dir/CHR004.png" \
        --bake-alpha \
        --overwrite 2>&1 | tail -20
done

for model in gpt-image-2 gpt-image-2.5-sunburst; do
    out_dir="$SINGLE_ROOT/$model"
    mkdir -p "$out_dir"
    echo "==> Single stage: $model"
    python i2i_tools/extract_character_clothing.py \
        --provider openai --model "$model" \
        --images "${CROPS[@]}" \
        --output "$out_dir/CHR004.png" \
        --bake-alpha \
        --overwrite 2>&1 | tail -20
done

# Two-stage pipeline (turnaround first, then garment extraction).
TWO_STAGE_ROOT=workspace/i2i_test_two_stage
echo "==> Two-stage pipeline: all 5 models"
python i2i_tools/two_stage_clothing_pipeline.py \
    --images "${CROPS[@]}" \
    --character-label CHR004 \
    --output-root "$TWO_STAGE_ROOT" \
    --bake-alpha \
    --overwrite 2>&1 | tail -60

echo "Done."
echo "Single stage outputs: $SINGLE_ROOT/<model>/CHR004_{top,pants}.png"
echo "Two stage outputs:    $TWO_STAGE_ROOT/<model>/CHR004_{three_view,top,pants}.png"
