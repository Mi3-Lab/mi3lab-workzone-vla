#!/bin/bash
#SBATCH --job-name=alp_s7_1
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage7_1_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage7_1_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Stage 7 — hard negatives + restauracao do gate ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

# 1) Init a partir do checkpoint final do Stage 6
LATEST=$(ls -d $BASE/checkpoints/sft_stage6_general/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERRO: nenhum checkpoint do Stage 6 encontrado"; exit 1
fi
echo "Init : $LATEST"
cat > $SFT_DIR/configs/models/student_2b_v8.yaml <<EOF
_target_: alpamayo1_5_sft.models.sft_base_model.TrainableReasoningVLA.from_alpamayo_checkpoint
checkpoint_path: $LATEST
vlm_name_or_path: $BASE/models/Cosmos-Reason2-2B
EOF

# 2) Copiar o config do stage 7 pro diretorio de configs do treino
cp $BASE/mi3lab-workzone-vla/configs/sft_stage7_1.yaml $SFT_DIR/configs/

# 3) Treinar
mkdir -p $BASE/checkpoints/sft_stage7_1
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29510 \
    train_hf.py \
    --config-path configs \
    --config-name sft_stage7_1
