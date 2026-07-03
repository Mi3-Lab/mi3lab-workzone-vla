#!/bin/bash
#SBATCH --job-name=alp_s5ego
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage5ego_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage5ego_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate

export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

echo "=== Alpamayo Stage 5 — Ego-centric WZ state (continua do checkpoint-8622) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Data: lingoqa_egomix (8k roadwork retention + egostate x4)"
echo "=============================================================================="

mkdir -p $BASE/checkpoints/sft_stage5_egostate

cd $SFT_DIR

torchrun \
    --nproc_per_node=2 \
    --master_port=29507 \
    train_hf.py \
    --config-path configs \
    --config-name sft_stage5_egostate
