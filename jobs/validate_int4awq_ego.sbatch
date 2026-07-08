#!/bin/bash
#SBATCH --job-name=alp_valego
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/valego_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/valego_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Validacao BF16 vs INT4-v1(512 txt) vs INT4-ego(530 multimodal) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/diagnostics
python validate_int4awq_ego.py
echo "=== DONE ==="
