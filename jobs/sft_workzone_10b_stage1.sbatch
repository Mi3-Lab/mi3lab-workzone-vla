#!/bin/bash
#SBATCH --job-name=wz10b_s1
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_s1_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_s1_%j.err

# Fine-tune de workzone do Alpamayo-1.5-10B real (Cosmos-Reason2-8B backbone).
# 2x H200 (141GB cada), ZeRO-2 padrao SEM offload: o cluster nao tem nvcc/CUDA
# toolkit carregavel (so embutido em pacotes como GROMACS), entao o offload do
# optimizer pra CPU (DeepSpeedCPUAdam) nao compila. Sem offload, ZeRO-2 exige
# ~79GB/GPU (params 17.6GB + grads 8.8GB + AdamW particionado 52.8GB) -- cabe
# folgado nos 141GB da H200, mas nao cabe com margem nos 80GB de uma A100.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Workzone 10B Stage 1 — Alpamayo-1.5-10B + lingoqa_combined ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/alpamayo10b_workzone_init.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_workzone_10b_stage1.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_workzone_10b_stage1
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29531 \
    train_hf.py \
    --config-path configs \
    --config-name sft_workzone_10b_stage1 \
    ${EXTRA_OVERRIDES}
