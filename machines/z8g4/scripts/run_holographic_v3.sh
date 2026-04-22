#!/bin/bash
export TOKENIZERS_PARALLELISM=false
source /home/supercomputerz8/MedusaBitNet/.venv/bin/activate
numactl -N 0-1 -m 0-1 \
python -u machines/z8g4/scripts/train_holographic.py \
  --d-model 384 \
  --n-views 8 \
  --n-blocks 3 \
  --n-heads 6 \
  --head-dim 64 \
  --d-int 1536 \
  --n-kv-heads 3 \
  --seq-len 256 \
  --batch-size 8 \
  --lr 3e-4 \
  --epochs 5 \
  --max-steps-per-epoch 2000 \
  --max-tokens 20000000 \
  --baseline-layers 8 \
  --dataset wikitext-103-raw-v1 \
  --out machines/z8g4/results/holographic_train_v3.json
