#!/bin/bash
#SBATCH --job-name=alp_fp16diag
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/fp16diag_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/fp16diag_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python $BASE/mi3lab-workzone-vla/diagnostics/diag_fp16_overflow.py
