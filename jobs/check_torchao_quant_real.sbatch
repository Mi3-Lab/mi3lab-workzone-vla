#!/bin/bash
#SBATCH --job-name=c3e_check_quant
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_check_quant_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_check_quant_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
~/.local/bin/uv sync --group=cu128 2>&1 | tail -5
~/.local/bin/uv pip install boto3 qwen-vl-utils torchao 2>&1 | tail -5
echo
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/check_torchao_quant_real.py
