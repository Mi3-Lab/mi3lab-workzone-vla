#!/bin/bash
#SBATCH --job-name=alp_evaltrj2
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_traj_kd_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_traj_kd_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "=== Eval trajetoria: student KD 2B vs teacher 10B (PhysicalAI-AV val) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/diagnostics
python eval_traj_kd_v2.py
echo "=== DONE ==="
