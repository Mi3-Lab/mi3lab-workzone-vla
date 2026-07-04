#!/bin/bash
#SBATCH --job-name=alp_smoke_kd
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/smoke_kd_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/smoke_kd_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

echo "=== Smoke test: KD puro (teacher 10B original + student 2B) em PhysicalAI-AV ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/diagnostics
python smoke_test_kd_general.py
echo "=== DONE ==="
