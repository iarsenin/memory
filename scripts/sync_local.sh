#!/usr/bin/env bash
# Pull critical data from the pod to a local backup directory.
# Run this BEFORE stopping or terminating the pod.
#
# Usage:
#   bash scripts/sync_local.sh <LOCAL_BACKUP_DIR>
#   bash scripts/sync_local.sh ~/memlora_backup   (default)
#
# What gets synced (selective — not raw dialogue or model base weights):
#   logs/            training telemetry
#   results/         eval metrics
#   data/personas/   ground truth (small JSON)
#   data/memories/   extracted + scored memories (small JSONL)
#   data/eval_probes/ generated eval questions
#   checkpoints/     LoRA adapter weights (small, ~0.5 MB each)
#   configs/         experiment configs
#
# What is NOT synced (too large or regenerable):
#   data/dialogue/   raw dialogue transcripts (regenerate via Phase 1 if needed)
#   Base model weights (downloaded by HuggingFace on the pod)

set -e

LOCAL_DIR="${1:-$HOME/memlora_backup}"
POD_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== MemLoRA sync: pod → local ==="
echo "  Source : $POD_DIR"
echo "  Target : $LOCAL_DIR"
echo ""

mkdir -p "$LOCAL_DIR"

sync_dir() {
    local src="$POD_DIR/$1/"
    local dst="$LOCAL_DIR/$1/"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        rsync -av --progress "$src" "$dst"
        echo ""
    else
        echo "  Skipping $1/ (not found)"
    fi
}

sync_dir "logs"
sync_dir "results"
sync_dir "data/personas"
sync_dir "data/memories"
sync_dir "data/eval_probes"
sync_dir "checkpoints"
sync_dir "configs"

echo "=== Sync complete ==="
echo "  Local backup: $LOCAL_DIR"
echo ""
echo "  NOT synced: data/dialogue/ (large; regenerable via Phase 1)"
echo "  NOT synced: base model weights (re-downloaded by HuggingFace)"
