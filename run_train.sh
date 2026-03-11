#!/bin/bash
# ── RunPod Training Script ──────────────────────────────────────────────────
# Run this from inside the runpod_train/ directory:
#   bash run_train.sh

set -e
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install Pillow matplotlib numpy -q

echo "=== Starting training ==="
export PYTHONPATH=.
python -u -m src.train 2>&1 | tee training_log.txt

echo "=== Training complete. Checkpoint saved to outputs/checkpoints/ ==="
