#!/bin/bash
#SBATCH --job-name=diag_wz10b
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/diag_wz10b_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/diag_wz10b_%j.err

# Diagnostico do loss=0.0: inspeciona labels_mask e loss crua, comparando
# lingoqa_stage7_1 (dado que ja funcionou) vs lingoqa_combined (dado novo),
# com o MESMO checkpoint. So' inferencia, GPU generica, nao consome fila H200.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

python $BASE/mi3lab-workzone-vla/diagnostics/diag_wz10b_zero_loss.py
