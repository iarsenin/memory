#!/usr/bin/env bash
# Phase 1: Simulator Engine
# Generates ground truth persona JSON and 20-day dialogue JSONL for all personas.
#
# Usage:
#   bash scripts/run_phase1.sh                    # full run
#   bash scripts/run_phase1.sh --days 2           # quick sanity check (2 days)
#   bash scripts/run_phase1.sh --resume           # resume interrupted run

set -e
cd "$(dirname "$0")/.."

echo "=== Phase 1: Simulator Engine ==="
python -m src.simulator.run --config configs/sim_config.json "$@"
echo "=== Phase 1 complete ==="
