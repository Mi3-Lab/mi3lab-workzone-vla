#!/bin/bash
#SBATCH --job-name=kin_base
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kinbase_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kinbase_%j.err

# Baseline cinematico: SEM modelo, SEM GPU. So integra 6 acoes constantes pelo
# action space e mede minADE no chunk 3126. Roda como job mesmo sendo leve
# (PAIDataset decodifica video — nao e' pro no de login).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

python $BASE/mi3lab-workzone-vla/diagnostics/kinematic_baseline.py
