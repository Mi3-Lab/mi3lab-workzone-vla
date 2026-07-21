#!/bin/bash
#SBATCH --job-name=c3e_onnx_v3
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_onnx_v3_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_onnx_v3_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -10
~/.local/bin/uv pip install boto3 qwen-vl-utils onnx onnxscript 2>&1 | tail -10

echo
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/try_onnx_export_v3.py
