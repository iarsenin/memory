#!/usr/bin/env bash
# =============================================================================
# run_paper.sh — Full 3-seed paper run (Phase 5 → 6 → 7)
#
# Checkpoint layout (per seed, isolated):
#   checkpoints/paper/seed{S}/main/{pid}/day_{N}/     ← Phase 5
#   checkpoints/paper/seed{S}/{cond}/{pid}/day_{N}/   ← Phase 6
#
# Results layout:
#   results/paper/seed{S}/{condition}_{pid}_eval.json
#
# Aggregation (run locally after syncing results):
#   python analysis/summarize.py
#
# Usage:
#   bash scripts/run_paper.sh                       # full run, all 3 seeds
#   bash scripts/run_paper.sh --seeds 42            # single seed (debug)
#   bash scripts/run_paper.sh --skip-p5             # skip Phase 5 (resume)
#   bash scripts/run_paper.sh --skip-p6             # skip Phase 6 baselines
#   bash scripts/run_paper.sh --skip-p7             # skip Phase 7 eval
#   bash scripts/run_paper.sh --condition oracle_data_lora  # Phase 6: one condition
# =============================================================================

set -e
cd "$(dirname "$0")/.."

# ── Environment ──────────────────────────────────────────────────────────────
if [ -f .env ]; then set -a; source .env; set +a; fi

if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env"; exit 1
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set in .env"; exit 1
fi

# ── Parse args ───────────────────────────────────────────────────────────────
SEEDS=(42 123 456)
SKIP_P5=false
SKIP_P6=false
SKIP_P7=false
SINGLE_CONDITION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)       shift; IFS=' ' read -r -a SEEDS <<< "$*"; break ;;
        --skip-p5)     SKIP_P5=true ;;
        --skip-p6)     SKIP_P6=true ;;
        --skip-p7)     SKIP_P7=true ;;
        --condition)   shift; SINGLE_CONDITION="$1" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

BASELINE_CONDITIONS=("naive_lora" "unfiltered_lora" "oracle_data_lora" "ablation_no_salience" "ablation_no_replay" "ablation_no_negative")
if [ -n "$SINGLE_CONDITION" ]; then
    BASELINE_CONDITIONS=("$SINGLE_CONDITION")
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MemLoRA v1 — 3-Seed Paper Run                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  GPU      : ${GPU_NAME}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Baselines: ${BASELINE_CONDITIONS[*]}"
echo "  Skip P5  : ${SKIP_P5}  | Skip P6: ${SKIP_P6}  | Skip P7: ${SKIP_P7}"
echo ""

# ── Helpers ──────────────────────────────────────────────────────────────────
# Delete all day_* subdirs except the highest-numbered one (the final adapter).
# Phase 7 find_latest_checkpoint picks the highest day, so earlier ones are safe
# to remove once training for that condition is complete.
_prune_checkpoints() {
    local base="$1"   # e.g. checkpoints/paper/seed42/main
    [ -d "$base" ] || return 0
    for pid_dir in "$base"/*/; do
        [ -d "$pid_dir" ] || continue
        # Find the last day dir by sort order and keep it; delete the rest
        last=$(ls -d "$pid_dir"day_*/ 2>/dev/null | sort | tail -1)
        [ -n "$last" ] || continue
        for day_dir in "$pid_dir"day_*/; do
            [ "$day_dir" != "$last" ] && rm -rf "$day_dir"
        done
    done
    echo "    pruned intermediate checkpoints → kept day_18 only in ${base}"
}

TOTAL_START=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    SEED_START=$(date +%s)

    CKPT_BASE="checkpoints/paper/seed${SEED}"
    LOGS_DIR="logs/paper/seed${SEED}"
    RESULTS_DIR="results/paper/seed${SEED}"

    mkdir -p "${CKPT_BASE}/main" "${LOGS_DIR}" "${RESULTS_DIR}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SEED ${SEED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Seed-done guard ───────────────────────────────────────────────────
    # eval_summary.json is written last by Phase 7. If it exists, all phases
    # for this seed completed successfully — skip the entire seed.
    if [ -f "${RESULTS_DIR}/eval_summary.json" ]; then
        echo "  eval_summary.json exists — seed ${SEED} already complete, skipping."
        continue
    fi

    # ── Restore clean memory snapshot ────────────────────────────────────
    # Trainers write consolidated=True using open(path,"w") which truncates
    # files before writing. A crash mid-write leaves 0-byte files. We keep
    # clean backups from the post-Phase-4 state and restore them at each seed
    # start, ensuring each seed trains on a full, uncorrupted memory set.
    echo ""
    echo "  Restoring clean memory snapshot for seed ${SEED} …"
    python3 - <<'PYEOF'
import json, glob, shutil, sys, os

restored = 0
for src_dir, dst_dir in [
    ("data/memories_clean",            "data/memories"),
    ("data/memories_unfiltered_clean", "data/memories_unfiltered"),
]:
    if not os.path.isdir(src_dir):
        print(f"  WARNING: backup dir {src_dir!r} missing — falling back to in-place reset")
        for path in sorted(glob.glob(dst_dir + "/*.jsonl")):
            lines = [json.loads(l) for l in open(path) if l.strip()]
            if not lines:
                continue
            reset = [{**l, "consolidated": False} for l in lines]
            open(path, "w").write("\n".join(json.dumps(r) for r in reset) + "\n")
            restored += len(reset)
        continue
    for src_path in sorted(glob.glob(src_dir + "/*.jsonl")):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        count = sum(1 for l in open(dst_path) if l.strip())
        print(f"    restored {count:3d} items  ←  {dst_path}")
        restored += count

print(f"  Total restored: {restored} items")
PYEOF

    # ── Phase 5: main MemLoRA ─────────────────────────────────────────────
    if [ "$SKIP_P5" = false ]; then
        echo ""
        echo "  [P5] Training main MemLoRA (seed=${SEED}) …"
        python3 -m src.trainer.run \
            --config          configs/train_config.json \
            --seed            "${SEED}" \
            --memories-dir    data/memories \
            --dialogue-dir    data/dialogue \
            --checkpoints-dir "${CKPT_BASE}/main" \
            --logs-dir        "${LOGS_DIR}" \
            --resume
        # Keep only final day; intermediate days only needed for accumulation
        _prune_checkpoints "${CKPT_BASE}/main"
        echo "  [P5] Done."
    else
        echo "  [P5] Skipped."
    fi

    # ── Phase 6: baselines ────────────────────────────────────────────────
    if [ "$SKIP_P6" = false ]; then
        for COND in "${BASELINE_CONDITIONS[@]}"; do
            echo ""
            echo "  [P6] Training ${COND} (seed=${SEED}) …"
            python3 -m src.baselines.run \
                --condition       "${COND}" \
                --config          configs/train_config.json \
                --seed            "${SEED}" \
                --memories-dir    data/memories \
                --unfiltered-dir  data/memories_unfiltered \
                --dialogue-dir    data/dialogue \
                --personas-dir    data/personas \
                --checkpoints-dir "${CKPT_BASE}" \
                --logs-dir        "${LOGS_DIR}" \
                --resume
            _prune_checkpoints "${CKPT_BASE}/${COND}"
            echo "  [P6] ${COND} done."
        done
    else
        echo "  [P6] Skipped."
    fi

    # ── Phase 7: evaluation ───────────────────────────────────────────────
    # frozen/rag are deterministic — only run them for seed 42 to save API calls.
    # For seeds 123/456, skip non-adapter conditions; summarize.py reuses seed-42
    # scores for frozen/rag (std=0 is the correct finding for those baselines).
    if [ "$SKIP_P7" = false ]; then
        echo ""
        if [ "${SEED}" = "42" ]; then
            CONDITIONS_ARG=""
            echo "  [P7] Evaluating ALL conditions (seed=${SEED}) …"
        else
            CONDITIONS_ARG="--condition main,naive_lora,unfiltered_lora,oracle_data_lora,ablation_no_salience,ablation_no_replay,ablation_no_negative"
            echo "  [P7] Evaluating LoRA conditions only (seed=${SEED}, frozen/rag reused from seed 42) …"
        fi

        python3 -m src.eval.run \
            --train-config    configs/train_config.json \
            --eval-config     configs/eval_config.json \
            --personas-dir    data/personas \
            --memories-dir    data/memories \
            --checkpoints-dir "${CKPT_BASE}" \
            --results-dir     "${RESULTS_DIR}" \
            --eval-probes-dir data/eval_probes \
            ${CONDITIONS_ARG}
        echo "  [P7] Eval done."

        # ── Post-eval checkpoint cleanup ──────────────────────────────────
        # Eval results are now in results/. Adapter weights no longer needed.
        echo "  Removing seed ${SEED} checkpoints (eval complete, results saved) …"
        rm -rf "${CKPT_BASE}"
        echo "  Checkpoint cleanup done. Disk: $(df -h /workspace | awk 'NR==2{print $3\" used / \"$2\" total\"}')"
    else
        echo "  [P7] Skipped."
    fi

    SEED_END=$(date +%s)
    SEED_ELAPSED=$(( (SEED_END - SEED_START) / 60 ))
    echo ""
    echo "  Seed ${SEED} complete in ${SEED_ELAPSED} min."
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Paper run complete in ${TOTAL_ELAPSED} min."
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Sync results locally:"
echo "       bash scripts/sync_local.sh"
echo "  2. Aggregate & produce paper table:"
echo "       python analysis/summarize.py"
echo ""
