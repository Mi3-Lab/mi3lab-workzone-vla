#!/bin/bash
#SBATCH --job-name=alp_gendata
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/gen_general_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/gen_general_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

echo "=== Self-distillation: Cosmos-2B base gera Q&A de direção geral (retenção) ==="
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/data_prep
python gen_general_driving_data.py
echo "=== DONE ==="
