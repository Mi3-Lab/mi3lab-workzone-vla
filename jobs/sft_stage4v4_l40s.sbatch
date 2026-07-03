#!/bin/bash
#SBATCH --job-name=alp_s4v4
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage4v4_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage4v4_%j.err

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

echo "=== Alpamayo Stage 4 v4 — Student SFT (Cosmos-2B pretrained init, embed_tokens FROZEN) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Init: student_2b_clean_init (Cosmos-2B from_pretrained + resize)"
echo "Fix: vlm.model.language_model.embed_tokens LR = 0.0"
echo "=============================================================================="

mkdir -p $BASE/checkpoints/sft_stage4_v4

cd $SFT_DIR

torchrun \
    --nproc_per_node=2 \
    --master_port=29506 \
    train_hf.py \
    --config-path configs \
    --config-name sft_stage4_v4
