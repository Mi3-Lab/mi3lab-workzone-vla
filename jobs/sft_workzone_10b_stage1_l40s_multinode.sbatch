#!/bin/bash
#SBATCH --job-name=wz10b_l40s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_l40s_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_l40s_%j.err

# Fine-tune de workzone do Alpamayo-1.5-10B em 4x L40S (2 nos x 2 GPUs), ZeRO-3.
#
# Por que ZeRO-3 e nao ZeRO-2 aqui: ZeRO-2 so particiona grads+optimizer, os
# PARAMS ficam replicados inteiros em cada GPU (17.6GB fixos, nao cai com mais
# GPUs). Com N=4 isso ainda da ~48GB/GPU -- no limite exato da L40S, sem
# margem pra ativacoes. ZeRO-3 particiona TAMBEM os parametros: com N=4,
# (17.6+17.6+105.6)/4 = ~35GB/GPU, com folga real pra ativacoes.
#
# Por que 2 nos e nao 1: nenhum no deste cluster tem mais de 2 L40S: pra
# chegar em 4 GPUs precisa de 2 nos. srun+torchrun com rendezvous c10d e' o
# padrao SLURM pra isso -- cada no roda sua propria instancia de torchrun
# (nproc_per_node=2), c10d faz o handshake entre elas via MASTER_ADDR:PORT.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Workzone 10B Stage 1 — 4x L40S multi-node (ZeRO-3) ==="
echo "Nos: $SLURM_JOB_NODELIST"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    bash -c 'echo "  $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr "\n" " ")"'

cp $BASE/mi3lab-workzone-vla/configs/models/alpamayo10b_workzone_init.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_workzone_10b_stage1.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_workzone_10b_stage1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29532

cd $SFT_DIR
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_hf.py \
    --config-path configs \
    --config-name sft_workzone_10b_stage1 \
    trainer.deepspeed=configs/deepspeed/zero3_fast.json \
    ${EXTRA_OVERRIDES}
