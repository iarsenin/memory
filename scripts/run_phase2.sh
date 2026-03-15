#!/usr/bin/env bash
# Phase 2: Memory Extraction Pipeline
# Distills daily dialogue into structured memory items via OpenAI API.
# Runs locally (no GPU needed — OpenAI API only).
#
# Usage:
#   bash scripts/run_phase2.sh                    # full run
#   bash scripts/run_phase2.sh --days 3           # quick sanity check (3 days)
#   bash scripts/run_phase2.sh --resume           # resume interrupted run

set -e
cd "$(dirname "$0")/.."

echo "=== Phase 2: Memory Extraction ==="
python3 -m src.extractor.run --config configs/extract_config.json "$@"
echo "=== Phase 2 complete ==="
