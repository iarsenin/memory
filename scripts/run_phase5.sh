#!/usr/bin/env bash
# Phase 5 — Sleep Phase Consolidation Training (Pod GPU required)
#
# Usage:
#   bash scripts/run_phase5.sh                    # full run, both personas
#   bash scripts/run_phase5.sh --persona alice    # one persona only
#   bash scripts/run_phase5.sh --sanity           # 1-cycle debug (Day 3 only)
#   bash scripts/run_phase5.sh --resume           # skip completed checkpoints
#
# Requires:
#   • CUDA GPU (RTX 4090 recommended, 24 GB VRAM)
#   • HUGGING_FACE_TOKEN in .env (meta-llama/Meta-Llama-3-8B-Instruct access)
#   • Phase 4 complete (data/memories/*_memories.jsonl with salience_score)
set -e
cd "$(dirname "$0")/.."

# Load environment variables (HF token, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env"
    echo "  1. Accept the Llama-3 license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    echo "  2. Add HUGGING_FACE_TOKEN=hf_... to your .env file"
    exit 1
fi

ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --persona)  ARGS="$ARGS --persona $2"; shift 2 ;;
        --sanity)   ARGS="$ARGS --sanity";     shift ;;
        --resume)   ARGS="$ARGS --resume";     shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== MemLoRA Phase 5: Sleep Phase Consolidation ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

python3 -m src.trainer.run \
    --config         configs/train_config.json \
    --memories-dir   data/memories \
    --dialogue-dir   data/dialogue \
    --checkpoints-dir checkpoints \
    --logs-dir        logs \
    $ARGS

echo ""
echo "Checkpoints saved to checkpoints/"
echo "Training logs in logs/*_train_day*.jsonl"
echo ""
echo "Recommended: run 'bash scripts/sync_local.sh' to back up checkpoints and logs."
