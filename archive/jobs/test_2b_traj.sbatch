#!/bin/bash
#SBATCH --job-name=alp_test2b
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/test_2b_traj_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/test_2b_traj_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=true

echo "=== Alpamayo 2B trajectory test ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/archive/diagnostics
python test_2b_traj_v3.py
echo "=== DONE ==="
