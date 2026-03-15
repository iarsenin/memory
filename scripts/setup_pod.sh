#!/usr/bin/env bash
# Run once on a fresh RunPod pod to install dependencies and restore environment.
set -e

cd "$(dirname "$0")/.."
echo "=== MemLoRA Pod Setup ==="

# Install base Python packages (CPU torch; will be overridden by CUDA below)
pip install -r requirements.txt --quiet

# Install PyTorch with CUDA (pod has CUDA; skips on machines without GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# Create required data directories (in case they weren't restored)
mkdir -p data/personas data/dialogue data/memories data/eval_probes
mkdir -p logs results checkpoints

# Verify .env exists
if [ ! -f .env ]; then
    echo ""
    echo "WARNING: .env not found."
    echo "  Copy your .env file here or set environment variables manually."
    echo "  Required: OPENAI_API_KEY, OPENAI_MODEL"
fi

echo ""
echo "=== Setup complete ==="
echo "  Run 'bash scripts/run_phase1.sh' to start Phase 1."
