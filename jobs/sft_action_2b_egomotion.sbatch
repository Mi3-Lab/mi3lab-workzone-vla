#!/bin/bash
#SBATCH --job-name=alp_act2b
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/action2b_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/action2b_%j.err

# Stage AR1-2B: treina a cabeca de acao real do Alpamayo (flow matching,
# pacote alpamayo_r1 da NVIDIA) em cima do nosso VLM 2B (workzone stage7_1),
# GT direto contra egomotion real (PhysicalAI-AV, 39 chunks locais). VLM
# congelado (cotrain_vlm: false) -- so o expert de acao (~1.77B) treina.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Stage AR1-2B — cabeca de acao real (flow matching) sobre VLM 2B ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

# 1) Copiar os configs (modelo + estagio) pro diretorio de configs do treino
cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_egomotion.yaml $SFT_DIR/configs/

# 2) Treinar (wrapper aplica o patch de vocabulario antes de chamar train_hf.train())
mkdir -p $BASE/checkpoints/sft_action_2b_egomotion
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29511 \
    $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_egomotion
