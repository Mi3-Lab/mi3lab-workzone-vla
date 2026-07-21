#!/bin/bash
#SBATCH --job-name=plot_traj_cmp
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/plot_traj_cmp_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/plot_traj_cmp_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python $BASE/mi3lab-workzone-vla/diagnostics/plot_trajectory_comparison.py \
    "$BASE/outputs/cosmos3edge_trajectory_test"
