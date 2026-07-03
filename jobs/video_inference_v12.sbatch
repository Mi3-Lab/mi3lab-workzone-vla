#!/bin/bash
#SBATCH --job-name=alp_v12_video
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v12_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v12_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft
CKPT=$BASE/checkpoints/sft_stage5_egostate

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "=== Alpamayo v12 — video NUNCA visto em nenhum treino (verificado por hash) ==="
echo "Fonte: seattle_11a6d1c0ae1944ec9bfe1ef71377be7a (0/9 segmentos sem overlap)"
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

mkdir -p $BASE/outputs/v12
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference/archive
python video_inference_v12.py \
    "$BASE/videos/seattle_unseen.mp4" \
    "$BASE/outputs/v12/v12_seattle_unseen.mp4" \
    "$CKPT"
echo "=== DONE ==="
