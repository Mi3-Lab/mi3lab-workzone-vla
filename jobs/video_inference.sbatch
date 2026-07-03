#!/bin/bash
#SBATCH --job-name=alpamayo_video
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/video_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/video_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate

export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

VIDEO_IN=${1:-$BASE/data/roadwork/videos/boston_042e1caf93114d3286c11ba14ddaa759_000001_00300_snippet.mp4}
VIDEO_OUT=${2:-$BASE/logs/alpamayo_video_v2.mp4}
CKPT=${3:-$BASE/checkpoints/sft_stage1_roadwork_v2/checkpoint-4500}

echo "=== Alpamayo Video Inference ==="
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Video : $VIDEO_IN"
echo "Out   : $VIDEO_OUT"
echo "CKPT  : $CKPT"
echo "================================="

STRIDE=${4:-3}
python $BASE/mi3lab-workzone-vla/inference/archive/video_inference_v1.py "$VIDEO_IN" "$VIDEO_OUT" "$CKPT" "$STRIDE"
