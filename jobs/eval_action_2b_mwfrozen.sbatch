#!/bin/bash
#SBATCH --job-name=eval_mwfroz
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_mwfroz_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/eval_mwfroz_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b_mwfrozen_eval.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/eval_action_2b_mwfrozen.yaml $SFT_DIR/configs/

cd $SFT_DIR
torchrun \
    --nproc_per_node=1 \
    --master_port=29530 \
    $BASE/mi3lab-workzone-vla/training/eval_action_2b.py \
    --config-path configs \
    --config-name eval_action_2b_mwfrozen
