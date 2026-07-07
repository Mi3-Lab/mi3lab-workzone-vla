#!/bin/bash
#SBATCH --job-name=alp_vtraj
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_traj_kd_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_traj_kd_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "=== Video traj KD student 2B — 4 cidades ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

mkdir -p $BASE/outputs/traj_kd
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference
for CITY in seattle_unseen boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_traj_kd.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/traj_kd/traj_${CITY}.mp4"
    echo ">>> [$CITY] done"
    echo ""
done
echo "=== ALL DONE ==="
