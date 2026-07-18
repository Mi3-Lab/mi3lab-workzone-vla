#!/bin/bash
#SBATCH --job-name=scale_curve
# 467M cabe folgado em qualquer GPU daqui. Vamos de L40S porque e' o que de fato
# tem GPU ociosa: as H200 estao todas alocadas e as A100 "livres" estao em
# DRAIN+REBOOT. Conferir com AllocTRES (nao CfgTRES) antes de escolher.
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/scale_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/scale_%j.err

# Preditor compacto de trajetoria (467M: ResNet-101 + decoder DETR multimodal).
# Alternativa ao VLA: sem VLM, so visao -> trajetoria. Mesmos dados, mesmo action
# space e mesma metrica (minADE no chunk 3126) dos baselines.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=$BASE/.hf_cache

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python $BASE/mi3lab-workzone-vla/training/scale_curve.py
