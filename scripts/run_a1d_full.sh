#!/usr/bin/env bash
# A1d: run full benchmark suite on base / A0 / A1 sequentially
set -euo pipefail
cd /home/cpinchington/Exponential-Inference

PY=/home/cpinchington/MedusaBitNet/.venv/bin/python
TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,wikitext"
COMMON_ENV="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

mkdir -p logs results/benchmarks

echo "=== A1d run 1/3: base Qwen3-0.6B ==="
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="Qwen/Qwen3-0.6B" TAG="base" TASKS="$TASKS" BATCH_SIZE=8 \
  $PY -u scripts/run_benchmarks.py 2>&1 | tee logs/a1d_base.log

echo ""
echo "=== A1d run 2/3: A0 (lossless conversion) ==="
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="model_package/Qwen3-0.6B-nGPT-form" TAG="a0" TASKS="$TASKS" BATCH_SIZE=8 \
  $PY -u scripts/run_benchmarks.py 2>&1 | tee logs/a1d_a0.log

echo ""
echo "=== A1d run 3/3: A1 (perfect-nGPT) ==="
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL="model_package/Qwen3-0.6B-nGPT-perfect" TAG="a1" TASKS="$TASKS" BATCH_SIZE=8 \
  $PY -u scripts/run_benchmarks.py 2>&1 | tee logs/a1d_a1.log

echo ""
echo "=== A1d done ==="
ls -la results/benchmarks/qwen3_06b_*.json
