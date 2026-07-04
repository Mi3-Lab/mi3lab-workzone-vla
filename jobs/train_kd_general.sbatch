#!/bin/bash
#SBATCH --job-name=alp_kd_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kd_general_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kd_general_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== KD geral: teacher Alpamayo-10B puro -> student 2B | PhysicalAI-AV ==="
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

mkdir -p $BASE/checkpoints/kd_general_driving
cd $SFT_DIR
python train_kd.py \
    --config-path configs \
    --config-name sft_kd_general_driving

echo "=== DONE ==="
