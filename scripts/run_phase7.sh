#!/usr/bin/env bash
# Phase 7 — Zero-Context Evaluation Suite (Pod GPU required)
#
# Usage:
#   bash scripts/run_phase7.sh                          # full run
#   bash scripts/run_phase7.sh --sanity                 # 3 probes/persona, debug
#   bash scripts/run_phase7.sh --persona alice          # one persona
#   bash scripts/run_phase7.sh --condition main         # one condition
#   bash scripts/run_phase7.sh --skip-inference         # re-run judge only
#   bash scripts/run_phase7.sh --skip-judge             # inference only
#   bash scripts/run_phase7.sh --force                  # rerun even if files exist

set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env"; exit 1
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set in .env"; exit 1
fi

# Forward any extra args (--sanity, --persona, --condition, --skip-*, --force)
ARGS="$@"

echo "=== MemLoRA Phase 7: Zero-Context Evaluation ==="
echo "  GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Args: ${ARGS:-<none>}"
echo ""

python3 -m src.eval.run \
    --train-config    configs/train_config.json \
    --eval-config     configs/eval_config.json \
    --personas-dir    data/personas \
    --memories-dir    data/memories \
    --checkpoints-dir checkpoints \
    --results-dir     results \
    --eval-probes-dir data/eval_probes \
    $ARGS

echo ""
echo "Eval results in results/*_eval.json"
echo "Summary     in results/eval_summary.json"
echo ""
echo "Recommended: run 'bash scripts/sync_local.sh' to pull results before stopping pod."
