#!/bin/bash
#SBATCH --job-name=alp_cosmos_eval
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/cosmos_eval_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/cosmos_eval_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Alpamayo-1.5-10B BASE (sem fine-tuning) nos 5 videos de benchmark ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python $BASE/mi3lab-workzone-vla/diagnostics/eval_cosmos_base_workzone.py
echo "=== BASE EVAL DONE ==="
