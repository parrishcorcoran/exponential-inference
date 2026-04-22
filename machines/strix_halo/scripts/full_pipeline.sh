#!/bin/bash
# Full pipeline: generate corpus + train + evaluate
# Monitors GPU/memory throughout
set -e
cd /home/cpinchington/Exponential-Inference
PYTHON=/home/cpinchington/MedusaBitNet/.venv/bin/python
LOG=machines/strix_halo/results/pipeline_$(date +%Y%m%d_%H%M%S).log

echo "=== FULL PIPELINE ===" | tee "$LOG"
echo "$(date): Starting" | tee -a "$LOG"

# Step 1: Generate corpus
echo "--- Step 1: Corpus ---" | tee -a "$LOG"
$PYTHON -u machines/strix_halo/scripts/generate_corpus_14b.py 2>&1 | tee -a "$LOG"

# Step 2: Train Holographic Matryoshka
echo "--- Step 2: Training ---" | tee -a "$LOG"
$PYTHON -u machines/strix_halo/scripts/train_matryoshka.py \
    --teacher Qwen/Qwen3-14B \
    --corpus machines/strix_halo/scratch/corpora/corpus.pt \
    --k-min 64 --k-max 128 \
    --steps 8000 \
    --lr 1e-4 \
    --out machines/strix_halo/results/matryoshka_qwen3_14b_r64_128.json \
    --save-student machines/strix_halo/scratch/student_14b_r64_128/ \
    2>&1 | tee -a "$LOG"

echo "$(date): Complete" | tee -a "$LOG"
