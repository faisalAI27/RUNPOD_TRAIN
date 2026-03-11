#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  ONE-PASTE RunPod Setup + Train Script
#  Paste this entire block into the RunPod terminal and hit Enter.
#  It will: install deps → download Kaggle data → train the model
# ═══════════════════════════════════════════════════════════════════
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1 — Clone repository"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd /root
git clone https://github.com/faisalAI27/RUNPOD_TRAIN.git
cd RUNPOD_TRAIN

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2 — Install Python dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install Pillow matplotlib numpy kagglehub -q

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3 — Download & prepare Kaggle ASL dataset"
echo "  (requires KAGGLE_USERNAME and KAGGLE_KEY env vars)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export PYTHONPATH=.
python prepare_kaggle_data.py

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4 — Start training (logs saved to training_log.txt)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -u -m src.train 2>&1 | tee training_log.txt

echo ""
echo "✅ DONE — checkpoint saved to outputs/checkpoints/frcnn_best.pth"
echo ""
echo "To download it to your Mac, run this ON YOUR MAC:"
echo "  scp root@<runpod-ip>:/root/RUNPOD_TRAIN/outputs/checkpoints/frcnn_best.pth \\"
echo "      ~/Desktop/DNN_MID/DNN_MID/sign_language_frcnn/outputs/checkpoints/"
