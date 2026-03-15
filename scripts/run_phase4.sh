#!/usr/bin/env bash
# Phase 4 — Salience Scoring
# Usage:
#   bash scripts/run_phase4.sh
#   bash scripts/run_phase4.sh --persona alice
#   bash scripts/run_phase4.sh --dry-run          # inspect scores without modifying memories
set -e
cd "$(dirname "$0")/.."

ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --persona)  ARGS="$ARGS --persona $2"; shift 2 ;;
        --dry-run)  ARGS="$ARGS --dry-run"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== MemLoRA Phase 4: Salience Scoring ==="
echo ""

python3 -m src.salience.run \
    --config        configs/salience_config.json \
    --memories-dir  data/memories \
    --dialogue-dir  data/dialogue \
    --results-dir   results \
    $ARGS
