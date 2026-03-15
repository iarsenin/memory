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

# Verify .env exists and has required keys
if [ ! -f .env ]; then
    echo ""
    echo "WARNING: .env not found."
    echo "  Copy your .env file here or set environment variables manually."
    echo "  Required: OPENAI_API_KEY, OPENAI_MODEL, HUGGING_FACE_TOKEN"
else
    set -a; source .env; set +a
    if [ -z "$HUGGING_FACE_TOKEN" ]; then
        echo ""
        echo "WARNING: HUGGING_FACE_TOKEN not set in .env"
        echo "  Phase 5+ requires access to meta-llama/Meta-Llama-3-8B-Instruct."
        echo "  1. Accept the license: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
        echo "  2. Add HUGGING_FACE_TOKEN=hf_... to your .env"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "  Phase 1-4: run on local Mac or pod (CPU/OpenAI API)."
echo "  Phase 5+:  run on pod GPU — bash scripts/run_phase5.sh"
