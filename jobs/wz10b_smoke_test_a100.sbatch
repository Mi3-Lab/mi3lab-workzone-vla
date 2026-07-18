#!/bin/bash
#SBATCH --job-name=wz10b_ta100
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_ta100_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_ta100_%j.err

# Validacao pre-H200: MESMA config exata do job de producao (H200, ZeRO-2,
# sem offload), so trocando a GPU. A100 (80GB) e' mais apertada que H200
# (141GB) pra essa conta (~79GB/GPU esperado) -- se passar aqui, passa com
# folga na H200. Se estourar por memoria aqui, e' inconclusivo pra H200 (mas
# qualquer bug de CODIGO (checkpoint loading, trainer loop, dataset) que
# aparecer aqui TAMBEM apareceria na H200, entao vale a pena rodar antes.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Validacao pre-H200 em 2x A100 (mesma config: ZeRO-2, sem offload) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/alpamayo10b_workzone_init.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_workzone_10b_stage1.yaml $SFT_DIR/configs/

cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29534 \
    train_hf.py \
    --config-path configs \
    --config-name sft_workzone_10b_stage1 \
    +trainer.max_steps=3
