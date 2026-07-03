#!/bin/bash
#SBATCH --job-name=alp_v4_video
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v4_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v4_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft
CKPT=$BASE/checkpoints/sft_stage4_v4/checkpoint-8622

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "=== Alpamayo 2B v4 — Video Inference ==="
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CKPT : $CKPT"
echo ""

cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference/archive
for CITY in boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_inference_v4.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/v4/v4_${CITY}.mp4" \
        "$CKPT" \
        3
    echo ">>> [$CITY] done"
    echo ""
done

echo "=== ALL DONE ==="
