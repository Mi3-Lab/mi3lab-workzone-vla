#!/bin/bash
#SBATCH --job-name=alpamayo_traj
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/alpamayo_traj_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/alpamayo_traj_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

python $BASE/mi3lab-workzone-vla/diagnostics/extract_traj_boston.py
