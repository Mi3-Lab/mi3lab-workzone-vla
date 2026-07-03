#!/bin/bash
#SBATCH --job-name=alp_s6gen
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage6gen_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/stage6gen_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Stage 6 — retenção de direção geral + especialização work zone ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

# 1) Detectar o checkpoint final do Stage 5.1 e gerar o model config
LATEST=$(ls -d $BASE/checkpoints/sft_stage5_1_signs/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LATEST" ]; then
    echo "ERRO: nenhum checkpoint do Stage 5.1 encontrado"; exit 1
fi
echo "Init : $LATEST"
cat > $SFT_DIR/configs/models/student_2b_v7.yaml <<EOF
_target_: alpamayo1_5_sft.models.sft_base_model.TrainableReasoningVLA.from_alpamayo_checkpoint
checkpoint_path: $LATEST
vlm_name_or_path: $BASE/models/Cosmos-Reason2-2B
EOF

# 2) Montar o mix v3 (requer lingoqa_general já gerado pelo job de self-distill)
cd $BASE/data/roadwork
python merge_egomix_v3.py || exit 1

# 3) Treinar
mkdir -p $BASE/checkpoints/sft_stage6_general
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29508 \
    train_hf.py \
    --config-path configs \
    --config-name sft_stage6_general
