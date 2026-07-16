#!/bin/bash
#SBATCH --job-name=kdvel_l40s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kdvel_l40s_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kdvel_l40s_%j.err

# KD de campo de velocidade em 2x L40S (48GB cada), via particionamento de modelo:
#   GPU0 = aluno + grads + optimizer (~26GB)   <- unico processo de treino
#   GPU1 = professor 10B, so inferencia (~22GB)
# Juntos nao cabem numa L40S so (48.4GB > 48GB), mas separados sobra folga.
# Batch efetivo 16 (1 x accum 16), identico ao run de A100.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== KD de velocidade — 2x L40S (aluno GPU0 / professor GPU1) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b_kd_velocity.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_kd_velocity.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_action_2b_kd_velocity
cd $SFT_DIR
torchrun \
    --nproc_per_node=1 \
    --master_port=29519 \
    $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_kd_velocity \
    trainer.gradient_accumulation_steps=16 \
    model.teacher_device=cuda:1
