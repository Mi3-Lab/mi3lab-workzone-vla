#!/bin/bash
#SBATCH --job-name=c3e_build_qwen3vl
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_build_qwen3vl_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_build_qwen3vl_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -10
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -5

echo
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/build_cosmos3edge_qwen3vl_hf.py

echo
echo "[ok] conteudo final:"
ls -la $BASE/models/cosmos3edge-lm-hf/
du -sh $BASE/models/cosmos3edge-lm-hf/
