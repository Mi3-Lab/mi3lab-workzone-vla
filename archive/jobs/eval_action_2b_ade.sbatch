#!/bin/bash
#SBATCH --job-name=alp_eval_ade
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_ade_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_ade_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
EVAL_MAX_CLIPS=${EVAL_MAX_CLIPS:-999999} python $BASE/mi3lab-workzone-vla/diagnostics/eval_action_2b_ade.py
