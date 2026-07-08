#!/bin/bash
#SBATCH --job-name=alp_quant_ego
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/quant_ego_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/quant_ego_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Quantizacao INT4 AWQ com calibracao multimodal foco EGO ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/data_prep
python quantize_int4awq_ego.py

echo "=== DONE ==="
