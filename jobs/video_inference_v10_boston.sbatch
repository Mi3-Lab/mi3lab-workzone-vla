#!/bin/bash
#SBATCH --job-name=alp_v10_bos
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v10_boston_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/video_v10_boston_%j.err

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

echo "=== v10 Boston retry (job anterior falhou por erro transiente CUDA/MPS) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p $BASE/outputs/v10
cd /data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference/archive
python video_inference_v10.py \
    "$BASE/videos/boston.mp4" \
    "$BASE/outputs/v10/v10_boston.mp4" \
    "$CKPT"
echo "=== DONE ==="
