#!/usr/bin/env bash
# Phase 3 — Extraction Evaluation (Gatekeeper)
# Usage:
#   bash scripts/run_phase3.sh
#   bash scripts/run_phase3.sh --persona alice
set -e
cd "$(dirname "$0")/.."

PERSONA_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --persona) PERSONA_ARG="--persona $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== MemLoRA Phase 3: Extraction Evaluation ==="
echo ""

python3 -m src.extractor.eval \
    --personas-dir data/personas \
    --memories-dir data/memories \
    --results-dir  results \
    $PERSONA_ARG

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Gate passed. Ready for Phase 4."
else
    echo "Gate failed. Review results/extraction_eval_*.json for details."
fi

exit $EXIT_CODE
