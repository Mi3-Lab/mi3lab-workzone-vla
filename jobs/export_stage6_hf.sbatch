#!/bin/bash
#SBATCH --job-name=alp_export
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/export_hf_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/export_hf_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "=== Export stage6 VLM -> HF Qwen3-VL padrao ==="
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/data_prep
python export_stage6_hf.py
echo "=== DONE ==="
