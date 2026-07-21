#!/bin/bash
#SBATCH --job-name=c3e_torchao_quant
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_torchao_quant_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_torchao_quant_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128 (ambiente base que ja funciona)..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -15
echo "[setup] instalando so' torchao direto (evita puxar o grupo -train inteiro, "
echo "        que tem transformer-engine/flash-attn com risco de build nvcc)..."
~/.local/bin/uv pip install boto3 qwen-vl-utils torchao 2>&1 | tail -20

echo
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/quantize_cosmos3edge_torchao.py
