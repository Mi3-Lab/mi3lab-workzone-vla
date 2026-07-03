#!/bin/bash
#SBATCH --job-name=alp_teach_v2
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/teacher_v2_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/teacher_v2_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0

echo "=== Alpamayo Teacher 10B — SFT pathways only (lingoqa_pathways) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

mkdir -p $BASE/checkpoints/sft_teacher_v2_pathways
cd $SFT_DIR

torchrun \
    --nproc_per_node=2 \
    --master_port=29508 \
    train_hf.py \
    --config-path configs \
    --config-name sft_teacher_v2_pathways
