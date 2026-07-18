#!/bin/bash
#SBATCH --job-name=wz10b_t4
#SBATCH --partition=test
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_t4_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/wz10b_t4_%j.err

# Smoke test do ZeRO-3 em 4x L40S NUM SO NO (gnode016, achado via `test`
# partition -- idle, fora do pool gpu/cenvalarc.gpu que estavamos monitorando).
# Valida o config antes de esperar o multi-node (2 nos) entrar na fila cenvalarc.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Smoke ZeRO-3 4x L40S single-node (test partition) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/alpamayo10b_workzone_init.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_workzone_10b_stage1.yaml $SFT_DIR/configs/

cd $SFT_DIR
torchrun \
    --nproc_per_node=4 \
    --master_port=29533 \
    train_hf.py \
    --config-path configs \
    --config-name sft_workzone_10b_stage1 \
    trainer.deepspeed=configs/deepspeed/zero3_fast.json \
    +trainer.max_steps=3
