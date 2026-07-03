#!/bin/bash
#SBATCH --job-name=alp_diag
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/diag_traj_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/diag_traj_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=true

echo "=== Alpamayo 10B trajectory diagnostic ==="
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/archive/diagnostics
python diag_traj.py
echo "=== DONE ==="
