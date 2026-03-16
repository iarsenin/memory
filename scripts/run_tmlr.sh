#!/usr/bin/env bash
# =============================================================================
# run_tmlr.sh — TMLR Revision: Full Self-Contained Pipeline (Phases 1–7)
#
# Runs the complete experiment end-to-end:
#   Phase 1  — Simulate 10 personas (new 8; alice/bob reused via --resume)
#   Phase 2a — Extract memories to data/memories/ (filtered, all 10 personas)
#   Phase 2b — Extract memories to data/memories_unfiltered/ (all 10 personas)
#   Phase 4  — Salience scoring (all 10 personas)
#   Phase 5–7 — 3-seed paper run (main + 6 baseline/ablation conditions + eval)
#
# Self-recovery features:
#   • Sentinel files guard each upstream phase so restarts skip completed work.
#   • Phase 1 and 2 use --resume for per-day granularity.
#   • Phase 5–7 use run_paper.sh's own checkpoint/sentinel system.
#   • API failures in Phase 1/2 are retried once before aborting that phase.
#   • All output is tee'd to logs/tmlr_run.log.
#
# Usage (from /workspace/memory):
#   tmux new-session -d -s tmlr 'bash scripts/run_tmlr.sh 2>&1 | tee logs/tmlr_run.log'
# =============================================================================

set -eo pipefail
cd "$(dirname "$0")/.."

# ── Env ──────────────────────────────────────────────────────────────────────
if [ -f .env ]; then set -a; source .env; set +a; fi

[[ -z "$OPENAI_API_KEY"    ]] && echo "ERROR: OPENAI_API_KEY not set"    && exit 1
[[ -z "$HUGGING_FACE_TOKEN" ]] && echo "ERROR: HUGGING_FACE_TOKEN not set" && exit 1

mkdir -p logs data/memories data/memories_unfiltered data/dialogue data/personas

TOTAL_START=$(date +%s)
ALL_PERSONAS="alice bob charlie diana ethan fiona george hannah ian julia"
NEW_PERSONAS="charlie diana ethan fiona george hannah ian julia"

# ── Helpers ──────────────────────────────────────────────────────────────────
_elapsed() { echo $(( ($(date +%s) - TOTAL_START) / 60 )) ; }
_log() { echo "[TMLR $(date '+%H:%M:%S')] $*"; }
_sentinel() { echo "data/.tmlr_${1}_done"; }

_run_with_retry() {
    # Usage: _run_with_retry "phase label" cmd [args...]
    local label="$1"; shift
    _log "${label}: starting …"
    if "$@"; then
        _log "${label}: OK"
        return 0
    fi
    _log "${label}: FAILED — retrying in 30 s …"
    sleep 30
    if "$@"; then
        _log "${label}: OK (retry succeeded)"
        return 0
    fi
    _log "${label}: FAILED after retry — see log above"
    return 1
}

# ── Banner ────────────────────────────────────────────────────────────────────
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
TARGET_MODULES=$(python3 -c "
import json; c=json.load(open('configs/train_config.json'))
print(','.join(c['lora']['target_modules']))" 2>/dev/null || echo "unknown")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       MemLoRA TMLR Revision — Full 10-Persona Pipeline      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  GPU            : ${GPU}"
echo "  LoRA modules   : ${TARGET_MODULES}"
echo "  Personas       : ${ALL_PERSONAS}"
echo "  Seeds          : 42 123 456"
echo "  Conditions     : main + naive_lora + unfiltered_lora +"
echo "                   oracle_data_lora + ablation_no_salience +"
echo "                   ablation_no_replay + ablation_no_negative"
echo ""

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — Simulate all 10 personas
# ═══════════════════════════════════════════════════════════════════
P1_SENTINEL=$(_sentinel "phase1")

if [ -f "$P1_SENTINEL" ]; then
    _log "Phase 1: sentinel found — skipping (all dialogue already generated)"
else
    _log "Phase 1: generating ground truth + dialogue for all 10 personas …"
    _log "  (alice/bob dialogue is reused via --resume; only new 8 are generated)"

    _run_with_retry "Phase 1" python3 -m src.simulator.run \
        --config configs/sim_config.json --resume

    # Verify all 10 dialogue files exist
    MISSING_P1=""
    for pid in $ALL_PERSONAS; do
        [ -f "data/dialogue/${pid}_dialogue.jsonl" ] || MISSING_P1="$MISSING_P1 $pid"
    done
    if [ -n "$MISSING_P1" ]; then
        _log "Phase 1 WARNING: missing dialogue for:${MISSING_P1} — retrying those …"
        for pid in $MISSING_P1; do
            _run_with_retry "Phase 1 retry ${pid}" python3 -m src.simulator.run \
                --config configs/sim_config.json --persona "$pid"
        done
    fi

    # Final check
    STILL_MISSING=""
    for pid in $ALL_PERSONAS; do
        [ -f "data/dialogue/${pid}_dialogue.jsonl" ] || STILL_MISSING="$STILL_MISSING $pid"
    done
    if [ -n "$STILL_MISSING" ]; then
        _log "Phase 1 FATAL: still missing dialogue for:${STILL_MISSING}"
        exit 1
    fi

    touch "$P1_SENTINEL"
    _log "Phase 1 complete ($(_elapsed) min elapsed)"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 2a — Extract memories (filtered → data/memories/)
# ═══════════════════════════════════════════════════════════════════
P2A_SENTINEL=$(_sentinel "phase2a")

if [ -f "$P2A_SENTINEL" ]; then
    _log "Phase 2a: sentinel found — skipping (filtered memories already extracted)"
else
    _log "Phase 2a: extracting memories to data/memories/ (all 10 personas, fresh + dedup) …"

    # Clear old alice/bob memories so dedup runs fresh for them too
    for pid in $ALL_PERSONAS; do
        rm -f "data/memories/${pid}_memories.jsonl"
    done

    _run_with_retry "Phase 2a" python3 -m src.extractor.run \
        --config configs/extract_config.json

    # Per-persona retry for any that failed
    for pid in $ALL_PERSONAS; do
        if [ ! -f "data/memories/${pid}_memories.jsonl" ]; then
            _log "  Phase 2a: retrying ${pid} …"
            _run_with_retry "Phase 2a ${pid}" python3 -m src.extractor.run \
                --config configs/extract_config.json --persona "$pid"
        fi
    done

    MISSING_P2A=""
    for pid in $ALL_PERSONAS; do
        [ -f "data/memories/${pid}_memories.jsonl" ] || MISSING_P2A="$MISSING_P2A $pid"
    done
    [ -n "$MISSING_P2A" ] && _log "Phase 2a FATAL: missing memories:${MISSING_P2A}" && exit 1

    touch "$P2A_SENTINEL"
    _log "Phase 2a complete ($(_elapsed) min elapsed)"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 2b — Extract memories (unfiltered → data/memories_unfiltered/)
# ═══════════════════════════════════════════════════════════════════
P2B_SENTINEL=$(_sentinel "phase2b")

if [ -f "$P2B_SENTINEL" ]; then
    _log "Phase 2b: sentinel found — skipping (unfiltered memories already extracted)"
else
    _log "Phase 2b: extracting memories to data/memories_unfiltered/ (all 10 personas) …"

    for pid in $ALL_PERSONAS; do
        rm -f "data/memories_unfiltered/${pid}_memories.jsonl"
    done

    _run_with_retry "Phase 2b" python3 -m src.extractor.run \
        --config configs/extract_unfiltered_config.json

    for pid in $ALL_PERSONAS; do
        if [ ! -f "data/memories_unfiltered/${pid}_memories.jsonl" ]; then
            _log "  Phase 2b: retrying ${pid} …"
            _run_with_retry "Phase 2b ${pid}" python3 -m src.extractor.run \
                --config configs/extract_unfiltered_config.json --persona "$pid"
        fi
    done

    MISSING_P2B=""
    for pid in $ALL_PERSONAS; do
        [ -f "data/memories_unfiltered/${pid}_memories.jsonl" ] || MISSING_P2B="$MISSING_P2B $pid"
    done
    [ -n "$MISSING_P2B" ] && _log "Phase 2b FATAL: missing unfiltered memories:${MISSING_P2B}" && exit 1

    touch "$P2B_SENTINEL"
    _log "Phase 2b complete ($(_elapsed) min elapsed)"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — Salience scoring (writes scores back into data/memories/)
# ═══════════════════════════════════════════════════════════════════
P4_SENTINEL=$(_sentinel "phase4")

if [ -f "$P4_SENTINEL" ]; then
    _log "Phase 4: sentinel found — skipping (salience scores already applied)"
else
    _log "Phase 4: applying salience scoring to all 10 personas …"

    _run_with_retry "Phase 4" python3 -m src.salience.run \
        --config       configs/salience_config.json \
        --memories-dir data/memories \
        --dialogue-dir data/dialogue \
        --results-dir  results

    # Verify salience_score is populated (spot check first persona)
    python3 - <<'PYEOF'
import json, glob, sys
ok = True
for path in sorted(glob.glob("data/memories/*_memories.jsonl")):
    lines = [json.loads(l) for l in open(path) if l.strip()]
    if lines and lines[0].get("salience_score") is None:
        print(f"  WARNING: salience_score missing in {path}")
        ok = False
    else:
        print(f"  OK: {path} ({len(lines)} items, salience populated)")
if not ok:
    sys.exit(1)
PYEOF

    touch "$P4_SENTINEL"
    _log "Phase 4 complete ($(_elapsed) min elapsed)"
fi

# ═══════════════════════════════════════════════════════════════════
# PRE-FLIGHT: clear stale 2-persona results so Phase 5–7 run fresh
# ═══════════════════════════════════════════════════════════════════
_log "Pre-flight: clearing stale 2-persona paper results and eval probes …"

# Remove old seed-done sentinels and eval_summary.json so run_paper.sh
# doesn't skip seeds that were completed with only alice/bob.
for seed in 42 123 456; do
    rm -f "results/paper/seed${seed}/eval_summary.json"
    rm -f "results/paper/seed${seed}/.fallback_main_done"
done

# Remove old eval probes (2-persona) so Phase 7 regenerates for all 10.
rm -f data/eval_probes/probes.json

_log "Pre-flight done."

# ═══════════════════════════════════════════════════════════════════
# PHASES 5–7 — Full 3-seed paper run (run_paper.sh)
# ═══════════════════════════════════════════════════════════════════
_log "Launching Phase 5→7 paper run (run_paper.sh) …"
_log "  Conditions: main + 6 baselines/ablations × 10 personas × 3 seeds"
_log "  Estimated: ~8–10 hours"

bash scripts/run_paper.sh

# ═══════════════════════════════════════════════════════════════════
# POST-RUN: aggregate results
# ═══════════════════════════════════════════════════════════════════
_log "Aggregating results …"
python3 analysis/summarize.py 2>&1

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TMLR full pipeline complete in ${TOTAL_ELAPSED} min.            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps (run locally):"
echo "  scp -P 40013 -i /tmp/cursor_runpod -r \\"
echo "    root@<ip>:/workspace/memory/results ."
echo "  python analysis/summarize.py"
