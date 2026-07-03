#!/bin/bash
#SBATCH --job-name=alp_v14
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v14_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v14_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft
CKPT=$BASE/checkpoints/sft_stage5_1_signs

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "=== Alpamayo v14 — sign OCR (Stage 5.1) + display policy ==="
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

mkdir -p $BASE/outputs/v14
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference/archive
for CITY in seattle_unseen boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_inference_v14.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/v14/v14_${CITY}.mp4" \
        "$CKPT"
    echo ">>> [$CITY] done"
    echo ""
done

echo "=== ALL DONE ==="
