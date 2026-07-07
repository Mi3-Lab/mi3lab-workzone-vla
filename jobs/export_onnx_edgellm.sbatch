#!/bin/bash
#SBATCH --job-name=alp_onnx
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/export_onnx_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/export_onnx_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Export ONNX (LLM + visual encoder) — TensorRT Edge-LLM ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

tensorrt-edgellm-export \
    $BASE/models/workzone-2b-stage6-int4awq \
    $BASE/models/workzone-2b-stage6-onnx

echo "=== DONE ==="
du -sh $BASE/models/workzone-2b-stage6-onnx
find $BASE/models/workzone-2b-stage6-onnx -maxdepth 2
