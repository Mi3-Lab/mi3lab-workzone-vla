#!/bin/bash
#SBATCH --job-name=alp_act2b_dry
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/act2b_dry_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/act2b_dry_%j.err

# Dry run: mesma config do treino real, mas 1 GPU + poucos steps, so pra
# validar que Hydra/dataset/collate_fn/Trainer/modelo fecham juntos antes
# de comprometer o job longo (2x L40S, 1d12h).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_egomotion.yaml $SFT_DIR/configs/

mkdir -p /tmp/act2b_dryrun_$SLURM_JOB_ID
cd $SFT_DIR
python $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_egomotion \
    paths.output_dir=/tmp/act2b_dryrun_$SLURM_JOB_ID \
    +trainer.max_steps=3 \
    trainer.save_steps=1000000 \
    trainer.logging_steps=1 \
    trainer.per_device_train_batch_size=1 \
    trainer.gradient_accumulation_steps=1 \
    trainer.dataloader_num_workers=0

echo "=== DRY RUN CONCLUIDO ==="
