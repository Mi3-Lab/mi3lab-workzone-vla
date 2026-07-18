#!/bin/bash
#SBATCH --job-name=dl_roadwork
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/dl_roadwork_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/dl_roadwork_%j.err

# Download + extracao seletiva dos videos ROADWork que faltam. So CPU/rede/disco
# (download HF + zip -FF + extracao seletiva), sem GPU. Roda como job porque e'
# uma operacao longa (rede + I/O de ~174GB no pico) — nao e' pro no de login.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1

df -h /data | tail -1
python $BASE/mi3lab-workzone-vla/data_prep/download_roadwork_videos_selective.py
echo "--- disco apos ---"
df -h /data | tail -1
