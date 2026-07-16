#!/bin/bash
#SBATCH --job-name=kdvel_smoke
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:50:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kdvel_smoke_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/kdvel_smoke_%j.err

# Stage AR1-2B-KDvel: destila o CAMPO DE VELOCIDADE do professor 10B pro expert
# de acao do aluno 2B. Professor e aluno residem juntos na GPU (por isso A100-80GB,
# nao L40S-48GB): ~20GB professor + ~8.5GB aluno + ~18GB optimizer/grads.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Stage AR1-2B-KDvel — destilacao de campo de velocidade (professor 10B) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b_kd_velocity.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_kd_velocity.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_action_2b_kd_velocity
cd $SFT_DIR
torchrun \
    --nproc_per_node=1 \
    --master_port=29516 \
    $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_kd_velocity +trainer.max_steps=3
