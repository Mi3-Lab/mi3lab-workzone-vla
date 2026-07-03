#!/bin/bash
#SBATCH --job-name=alp_v10_video
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v10_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v10_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft
# Diretório de treino: o script escolhe o checkpoint-* mais recente sozinho
CKPT=$BASE/checkpoints/sft_stage5_egostate

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "=== Alpamayo v10 — Learned ego-centric state (Stage 5) + Bayes filter ==="
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CKPTs : $(ls -d $CKPT/checkpoint-* 2>/dev/null | tail -3 | tr '\n' ' ')"
echo ""

mkdir -p $BASE/outputs/v10
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference/archive
for CITY in boston denver chicago; do
    echo ">>> [$CITY] starting..."
    python video_inference_v10.py \
        "$BASE/videos/${CITY}.mp4" \
        "$BASE/outputs/v10/v10_${CITY}.mp4" \
        "$CKPT"
    echo ">>> [$CITY] done"
    echo ""
done

echo "=== ALL DONE ==="
