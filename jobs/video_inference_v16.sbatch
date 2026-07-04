#!/bin/bash
#SBATCH --job-name=alp_v16
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v16_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v16_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft
CKPT=$BASE/checkpoints/sft_stage6_general

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "=== Alpamayo v16 — Stage 6 (general driving retention + workzone) ==="
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

mkdir -p $BASE/outputs/v16
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference
for CITY in seattle_unseen boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_inference_v16.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/v16/v16_${CITY}.mp4" \
        "$CKPT"
    echo ">>> [$CITY] done"
    echo ""
done

echo "=== ALL DONE ==="
