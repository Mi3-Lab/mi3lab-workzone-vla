#!/bin/bash
#SBATCH --job-name=traj_wz
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/traj_wz_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/traj_wz_%j.err

# Professor 10B (~22GB) + nosso 2B (~8.5GB) = ~31GB, so inferencia -> cabe numa L40S (48GB).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python $BASE/mi3lab-workzone-vla/diagnostics/compare_traj_workzone.py
