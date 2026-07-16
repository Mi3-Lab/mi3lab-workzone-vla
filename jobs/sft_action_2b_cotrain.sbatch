#!/bin/bash
#SBATCH --job-name=alp_cotrain
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cotrain_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cotrain_%j.err

# Opcao 3: descongela o VLM. Modelo inteiro (4.23B) treina, com o gradiente da
# loss de acao retroalimentando as features visuais (stop_grad_from_vlm=false).
# ZeRO-2 particiona optimizer+grads entre as 2 L40S; sem isso nao caberia.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Stage AR1-2B-cotrain — VLM DESCONGELADO (ZeRO-2 + grad ckpt) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b_cotrain.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_cotrain.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_action_2b_cotrain
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29520 \
    $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_cotrain
