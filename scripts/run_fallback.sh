#!/usr/bin/env bash
# =============================================================================
# run_fallback.sh — Fallback: retrain 'main' only with expanded LoRA surface
#
# Context
# -------
# The initial paper run (q_proj + v_proj, r=16) showed 'unfiltered_lora' beating
# 'main' on Updated facts, suggesting the salience-filtered batches are too sparse
# for the small adapter to absorb. This script expands LoRA to all 7 attention +
# MLP projections (q/k/v/o/gate/up/down) and retrains only the 'main' condition
# so we don't pay the cost of rerunning the 5.5-hour full baseline suite.
#
# What it does
# ------------
#   For each seed (42, 123, 456):
#     1. Reset consolidated flags so Phase 5 sees all memories as new.
#     2. Train Phase 5 (main MemLoRA) with the expanded 7-module config.
#        Checkpoints → checkpoints/fallback/seed{S}/main/
#     3. Prune intermediate checkpoints (keep day_18 only).
#     4. Run Phase 7 (eval, 'main' condition only) with --force to overwrite
#        the old main_*.json files in results/paper/seed{S}/.
#        Baseline eval files (frozen/rag/naive_lora/etc.) are left untouched.
#     5. Delete fallback checkpoints (results already saved).
#     6. Write .fallback_main_done sentinel to allow safe reruns.
#
# After all seeds finish, run locally:
#   python analysis/summarize.py
# This reads the updated main_*.json plus the existing baseline files and
# produces a new analysis/paper_results.md with the correct merged table.
#
# State-tracking notes
# --------------------
# • Sentinel: results/paper/seed{S}/.fallback_main_done — skip if present.
# • Baseline eval files are NEVER touched; only main_{pid}_eval/responses.json
#   are overwritten (--force flag passed to Phase 7).
# • eval_summary.json in results/paper/seed{S}/ is rewritten by Phase 7 to
#   contain 'main' metrics only. summarize.py does NOT read eval_summary.json
#   (it reads individual condition files), so this is safe.
# • Consolidated flags in data/memories/*.jsonl are reset at the start of each
#   seed loop. This is the same approach used in run_paper.sh.
#
# Usage
# -----
#   bash scripts/run_fallback.sh               # all 3 seeds
#   bash scripts/run_fallback.sh --seeds "42"  # single seed (debug/test)
# =============================================================================

set -e
cd "$(dirname "$0")/.."

# ── Environment ───────────────────────────────────────────────────────────────
if [ -f .env ]; then set -a; source .env; set +a; fi

if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env"; exit 1
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set in .env"; exit 1
fi

# ── Args ──────────────────────────────────────────────────────────────────────
SEEDS=(42 123 456)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) shift; IFS=' ' read -r -a SEEDS <<< "$1" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Banner ────────────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
TARGET_MODULES=$(python3 -c "
import json; c=json.load(open('configs/train_config.json'))
print(','.join(c['lora']['target_modules']))
")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MemLoRA v1 — Fallback: Expanded LoRA Surface        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  GPU            : ${GPU_NAME}"
echo "  Seeds          : ${SEEDS[*]}"
echo "  LoRA modules   : ${TARGET_MODULES}"
echo "  Condition      : main only (baselines reused from paper run)"
echo "  Results dir    : results/paper/seed{S}/ (main files overwritten)"
echo "  Checkpoint dir : checkpoints/fallback/seed{S}/main/"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
_prune_checkpoints() {
    local base="$1"
    [ -d "$base" ] || return 0
    for pid_dir in "$base"/*/; do
        [ -d "$pid_dir" ] || continue
        last=$(ls -d "$pid_dir"day_*/ 2>/dev/null | sort | tail -1)
        [ -n "$last" ] || continue
        for day_dir in "$pid_dir"day_*/; do
            [ "$day_dir" != "$last" ] && rm -rf "$day_dir"
        done
    done
    echo "    pruned intermediate checkpoints — kept final day only in ${base}"
}

_clear_gpu_cache() {
    python3 -c "import torch; torch.cuda.empty_cache(); print('    GPU cache cleared')" 2>/dev/null || true
}

TOTAL_START=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    SEED_START=$(date +%s)

    CKPT_BASE="checkpoints/fallback/seed${SEED}"
    RESULTS_DIR="results/paper/seed${SEED}"   # shared with baseline results
    LOGS_DIR="logs/fallback/seed${SEED}"
    SENTINEL="${RESULTS_DIR}/.fallback_main_done"

    mkdir -p "${CKPT_BASE}/main" "${LOGS_DIR}" "${RESULTS_DIR}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SEED ${SEED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Fallback-done guard ────────────────────────────────────────────────
    if [ -f "${SENTINEL}" ]; then
        echo "  .fallback_main_done exists — seed ${SEED} already complete, skipping."
        continue
    fi

    # ── Reset consolidated flags ───────────────────────────────────────────
    echo ""
    echo "  Resetting consolidated flags for seed ${SEED} …"
    python3 - <<'PYEOF'
import json, glob

reset_count = 0
for pattern in ["data/memories/*.jsonl", "data/memories_unfiltered/*.jsonl"]:
    for path in sorted(glob.glob(pattern)):
        lines = [json.loads(l) for l in open(path) if l.strip()]
        if not lines:
            continue
        reset = [{**l, "consolidated": False} for l in lines]
        open(path, "w").write("\n".join(json.dumps(r) for r in reset) + "\n")
        reset_count += len(reset)
        print(f"    reset {len(reset):3d} items  ←  {path}")
print(f"  Total reset: {reset_count} items")
PYEOF

    # ── Phase 5: main MemLoRA (expanded LoRA surface) ─────────────────────
    echo ""
    echo "  [P5] Training main MemLoRA — expanded surface (seed=${SEED}) …"
    python3 -m src.trainer.run \
        --config          configs/train_config.json \
        --seed            "${SEED}" \
        --memories-dir    data/memories \
        --dialogue-dir    data/dialogue \
        --checkpoints-dir "${CKPT_BASE}/main" \
        --logs-dir        "${LOGS_DIR}" \
        --resume

    _prune_checkpoints "${CKPT_BASE}/main"
    _clear_gpu_cache
    echo "  [P5] Done."

    # ── Phase 7: eval 'main' only, overwrite old main results ─────────────
    # Passes --force so inference+judge reruns even if old main_*.json exist.
    # Baseline files (frozen/rag/naive_lora/unfiltered_lora/gold_lora) are
    # untouched because we pass --condition main.
    echo ""
    echo "  [P7] Evaluating 'main' condition (seed=${SEED}) …"
    python3 -m src.eval.run \
        --train-config    configs/train_config.json \
        --eval-config     configs/eval_config.json \
        --personas-dir    data/personas \
        --memories-dir    data/memories \
        --checkpoints-dir "${CKPT_BASE}" \
        --results-dir     "${RESULTS_DIR}" \
        --eval-probes-dir data/eval_probes \
        --condition       main \
        --force
    echo "  [P7] Eval done."

    # ── Cleanup ───────────────────────────────────────────────────────────
    echo "  Removing fallback checkpoints for seed ${SEED} …"
    rm -rf "${CKPT_BASE}"
    DISK_USED=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $3" used / "$2" total"}' || echo "n/a")
    echo "  Checkpoint cleanup done. Disk: ${DISK_USED}"

    # ── Sentinel ──────────────────────────────────────────────────────────
    touch "${SENTINEL}"

    SEED_END=$(date +%s)
    SEED_ELAPSED=$(( (SEED_END - SEED_START) / 60 ))
    echo ""
    echo "  Seed ${SEED} complete in ${SEED_ELAPSED} min."
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Fallback run complete in ${TOTAL_ELAPSED} min."
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps (run locally after syncing results):"
echo "  1. Sync results:"
echo "       scp -P <port> -i /tmp/cursor_runpod -r \\"
echo "         root@<ip>:/workspace/memory/results/paper ."
echo "  2. Aggregate (merges new main with existing baselines):"
echo "       python analysis/summarize.py"
echo ""
