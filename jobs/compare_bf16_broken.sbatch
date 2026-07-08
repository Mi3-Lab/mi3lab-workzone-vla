#!/bin/bash
#SBATCH --job-name=alp_cmp_broken
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/cmp_broken_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/cmp_broken_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "=== Comparando frames quebrados no TRT vs BF16 ==="
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/diagnostics
python compare_bf16_broken_frames.py
echo "=== DONE ==="
