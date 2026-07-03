#!/bin/bash
#SBATCH --job-name=alp_v5b_video
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v5b_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v5b_%j.err

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

echo "=== Alpamayo 2B v5b — Description-grounded State Classification ==="
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

mkdir -p $BASE/outputs/v5b
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/archive/inference
for CITY in boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_inference_v5b.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/v5b/v5b_${CITY}.mp4" \
        "$CKPT" \
        3
    echo ">>> [$CITY] done"
done

echo "=== ALL DONE ==="
