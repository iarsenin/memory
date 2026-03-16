#!/usr/bin/env bash
# =============================================================================
# run_reviewer_revisions.sh  —  TMLR Reviewer Revisions (pod-side)
#
# Objective 1: Re-run the 3 ablation conditions for seed 456 (their sentinels
#              were written by a duplicate process without actual training/eval).
# Objective 3: Re-run RAG evaluation for all 3 seeds using the new BM25 Top-k
#              retrieval (delete stale rag_*_responses.json so inference reruns).
#
# Prerequisites:
#   • git pull (so judge.py / infer.py contain the reviewer-revision changes).
#   • pip install rank_bm25  (for BM25 RAG retrieval).
#   • Clean memory backups already exist: data/memories_clean/ etc.
# =============================================================================
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

[ -z "$HUGGING_FACE_TOKEN" ] && { echo "ERROR: HUGGING_FACE_TOKEN not set"; exit 1; }
[ -z "$OPENAI_API_KEY"     ] && { echo "ERROR: OPENAI_API_KEY not set";     exit 1; }

# Install BM25 library (lightweight, seconds).
pip install rank_bm25 --quiet || true

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MemLoRA — Reviewer Revisions (Objectives 1 + 3)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  GPU: ${GPU_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Retry helper (same as run_paper.sh)
# ---------------------------------------------------------------------------
_run_with_retry() {
    local max_retries=3 attempt=1
    while [ $attempt -le $max_retries ]; do
        "$@"; local rc=$?
        [ $rc -eq 0 ] && return 0
        echo "  [retry] exit $rc (attempt $attempt/$max_retries)"
        attempt=$(( attempt + 1 ))
        [ $attempt -le $max_retries ] && sleep 10
    done
    echo "  [ERROR] failed after $max_retries attempts"; return 1
}

# ---------------------------------------------------------------------------
# OBJECTIVE 1 — Fix ablation n=2 anomaly: re-run seed 456 ablation conditions
# ---------------------------------------------------------------------------
echo "══════════════════════════════════════════════════════════════"
echo "  OBJECTIVE 1 — Ablation seed 456 fix"
echo "══════════════════════════════════════════════════════════════"

SEED=456
CKPT_BASE="checkpoints/paper/seed${SEED}"
LOGS_DIR="logs/paper/seed${SEED}"
RESULTS_DIR="results/paper/seed${SEED}"
ABLATION_CONDS=("ablation_no_salience" "ablation_no_replay" "ablation_no_negative")

mkdir -p "${CKPT_BASE}" "${LOGS_DIR}" "${RESULTS_DIR}"

# Remove stale sentinels written by duplicate process.
for cond in "${ABLATION_CONDS[@]}"; do
    sentinel="${RESULTS_DIR}/${cond}_done.sentinel"
    if [ -f "$sentinel" ]; then
        rm "$sentinel"
        echo "  Removed stale sentinel: $sentinel"
    fi
done
# Remove premature seed_complete.json (written before ablations finished).
if [ -f "${RESULTS_DIR}/seed_complete.json" ]; then
    rm "${RESULTS_DIR}/seed_complete.json"
    echo "  Removed premature seed_complete.json for seed ${SEED}"
fi

# Restore clean memory snapshot for this seed.
echo ""
echo "  Restoring clean memory snapshot for seed ${SEED} …"
python3 - <<'PYEOF'
import json, glob, shutil, os

for src_dir, dst_dir in [
    ("data/memories_clean",            "data/memories"),
    ("data/memories_unfiltered_clean", "data/memories_unfiltered"),
]:
    if not os.path.isdir(src_dir):
        print(f"  WARNING: backup dir {src_dir!r} missing — falling back to in-place reset")
        for path in sorted(glob.glob(dst_dir + "/*.jsonl")):
            lines = [json.loads(l) for l in open(path) if l.strip()]
            reset = [{**l, "consolidated": False} for l in lines]
            open(path, "w").write("\n".join(json.dumps(r) for r in reset) + "\n")
        continue
    for src_path in sorted(glob.glob(src_dir + "/*.jsonl")):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        count = sum(1 for l in open(dst_path) if l.strip())
        print(f"    restored {count:3d} items  ←  {dst_path}")
PYEOF

# Train and evaluate each missing ablation condition.
for COND in "${ABLATION_CONDS[@]}"; do
    echo ""
    echo "  [P6] Training ${COND} (seed=${SEED}) …"
    _run_with_retry python3 -m src.baselines.run \
        --condition       "${COND}" \
        --config          configs/train_config.json \
        --seed            "${SEED}" \
        --memories-dir    data/memories \
        --unfiltered-dir  data/memories_unfiltered \
        --dialogue-dir    data/dialogue \
        --personas-dir    data/personas \
        --checkpoints-dir "${CKPT_BASE}/${COND}" \
        --logs-dir        "${LOGS_DIR}" \
        --resume
    echo "  [P6] ${COND} training done."

    echo "  [P7] Evaluating ${COND} (seed=${SEED}) …"
    python3 -m src.eval.run \
        --train-config    configs/train_config.json \
        --eval-config     configs/eval_config.json \
        --personas-dir    data/personas \
        --memories-dir    data/memories \
        --checkpoints-dir "${CKPT_BASE}" \
        --results-dir     "${RESULTS_DIR}" \
        --eval-probes-dir data/eval_probes \
        --condition       "${COND}"
    echo "  [P7] ${COND} eval done."

    # Free disk immediately.
    if [ -d "${CKPT_BASE}/${COND}" ]; then
        rm -rf "${CKPT_BASE}/${COND}"
        echo "  Deleted ${COND} checkpoints. Disk: $(df -h /workspace | awk 'NR==2{print $3" used / "$2" total"}')"
    fi

    touch "${RESULTS_DIR}/${COND}_done.sentinel"
done

# Mark seed 456 complete.
python3 -c "
import json, pathlib
p = pathlib.Path('${RESULTS_DIR}/seed_complete.json')
p.write_text(json.dumps({'seed': 456, 'ablation_fix': True}))
print('  seed_complete.json written for seed 456')
"

echo ""
echo "  ✓ Objective 1 complete."

# ---------------------------------------------------------------------------
# OBJECTIVE 3 — Top-k BM25 RAG re-evaluation (all 3 seeds)
# ---------------------------------------------------------------------------
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  OBJECTIVE 3 — BM25 Top-k RAG re-evaluation (seeds 42 123 456)"
echo "══════════════════════════════════════════════════════════════"

for SEED in 42 123 456; do
    CKPT_BASE="checkpoints/paper/seed${SEED}"
    RESULTS_DIR="results/paper/seed${SEED}"
    echo ""
    echo "  Seed ${SEED}: deleting stale RAG response files …"
    rm -f "${RESULTS_DIR}"/rag_*_responses.json \
          "${RESULTS_DIR}"/rag_*_eval.json 2>/dev/null || true
    echo "  Seed ${SEED}: running RAG inference + eval …"
    python3 -m src.eval.run \
        --train-config    configs/train_config.json \
        --eval-config     configs/eval_config.json \
        --personas-dir    data/personas \
        --memories-dir    data/memories \
        --checkpoints-dir "${CKPT_BASE}" \
        --results-dir     "${RESULTS_DIR}" \
        --eval-probes-dir data/eval_probes \
        --condition       rag
    echo "  Seed ${SEED}: RAG re-eval done."
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ All pod-side objectives complete."
echo "  Next: on your local machine:"
echo "    python3 analysis/rescore_deterministic.py"
echo "    python3 analysis/summarize.py --results-dir results/paper --seeds 42 123 456 ..."
echo "    python3 analysis/plot_results.py"
echo "══════════════════════════════════════════════════════════════"
