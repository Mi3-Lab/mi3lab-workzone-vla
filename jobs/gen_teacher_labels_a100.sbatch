#!/bin/bash
#SBATCH --job-name=alp_tlabels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/tlabels_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/tlabels_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | SHARD=$SHARD/$NUM_SHARDS"
python $BASE/mi3lab-workzone-vla/data_prep/gen_teacher_traj_labels.py
