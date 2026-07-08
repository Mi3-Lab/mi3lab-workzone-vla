#!/bin/bash
#SBATCH --job-name=alp_bf16_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/bf16_bench_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/bf16_bench_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Bench latencia BF16 (HuggingFace generate()) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/diagnostics
python bench_bf16_latency.py
echo "=== DONE ==="
