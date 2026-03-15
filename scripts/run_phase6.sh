#!/usr/bin/env bash
# Phase 6 — Baseline Training (Pod GPU required)
#
# Trains one condition at a time. Run three times for the three training baselines.
# frozen and rag require no training (handled at inference in Phase 7).
#
# Usage:
#   bash scripts/run_phase6.sh --condition naive_lora
#   bash scripts/run_phase6.sh --condition unfiltered_lora
#   bash scripts/run_phase6.sh --condition gold_lora
#   bash scripts/run_phase6.sh --condition naive_lora --persona alice   # single persona
#   bash scripts/run_phase6.sh --condition naive_lora --sanity          # Day 3 only
#   bash scripts/run_phase6.sh --condition naive_lora --resume          # skip done cycles
#
# Requires:
#   • CUDA GPU with >= 10 GB VRAM
#   • HUGGING_FACE_TOKEN in .env
#   • Phase 5 complete (checkpoints/main/ must exist)
#   • For unfiltered_lora: data/memories_unfiltered/ must exist
#     (re-run: python -m src.extractor.run --config configs/extract_unfiltered_config.json)
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env"
    exit 1
fi

CONDITION=""
ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)  CONDITION="$2"; shift 2 ;;
        --persona)    ARGS="$ARGS --persona $2"; shift 2 ;;
        --sanity)     ARGS="$ARGS --sanity"; shift ;;
        --resume)     ARGS="$ARGS --resume"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CONDITION" ]; then
    echo "Usage: bash scripts/run_phase6.sh --condition [naive_lora|unfiltered_lora|gold_lora]"
    exit 1
fi

echo "=== MemLoRA Phase 6: Baseline — ${CONDITION} ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

python3 -m src.baselines.run \
    --condition       "$CONDITION" \
    --config          configs/train_config.json \
    --memories-dir    data/memories \
    --unfiltered-dir  data/memories_unfiltered \
    --dialogue-dir    data/dialogue \
    --personas-dir    data/personas \
    --checkpoints-dir checkpoints \
    --logs-dir        logs \
    $ARGS

echo ""
echo "Checkpoints → checkpoints/${CONDITION}/"
echo "Run 'bash scripts/sync_local.sh' before stopping pod."
