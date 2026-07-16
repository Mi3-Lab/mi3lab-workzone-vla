#!/bin/bash
#SBATCH --job-name=transplant
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/transplant_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/transplant_%j.err

# Transplanta o expert de acao pre-treinado do professor 10B pro nosso 2B.
# So CPU/RAM (carrega ~30GB de pesos: professor ~22GB + nosso ~8.5GB) -- por isso
# roda como job e nao no no de login, onde foi morto por consumo de memoria.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

python $BASE/mi3lab-workzone-vla/training/transplant_expert.py
