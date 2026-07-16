#!/bin/bash
#SBATCH --job-name=alp_smoke2b
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/smoke2b_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/smoke2b_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python $BASE/mi3lab-workzone-vla/diagnostics/smoke_test_alpamayo2b.py
