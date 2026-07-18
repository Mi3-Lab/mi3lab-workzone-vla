#!/bin/bash
#SBATCH --job-name=multiwin
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=1-12:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/multiwin_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/multiwin_%j.err

# Opcao 3 (cotrain do VLM) numa unica A100-80GB, SEM DeepSpeed.
#
# Na L40S (48GB) o modelo inteiro nao cabia (OOM: 42.6/44.4GB usados), e as saidas
# eram todas ruins: gradient checkpointing e' arquiteturalmente impossivel aqui (o
# expert precisa do KV-cache do VLM, por isso a NVIDIA marca
# supports_gradient_checkpointing=False), e o DeepSpeed desliga a torch.autocast
# de que o action_in_proj depende (ele faz upcast fp32 interno de proposito).
#
# Na A100-80GB tudo isso desaparece: params 8.5 + grads 8.5 + AdamW fp32 33.8 +
# ativacoes ~12 = ~63GB, cabe sem particionar. E sem DeepSpeed o HF religa a
# autocast sozinho -- volta o caminho exato dos 3 treinos que ja funcionaram.
# Batch efetivo 16 (1 x accum 16), igual aos runs anteriores.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== Cotrain do VLM — 1x A100-80GB, sem DeepSpeed ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cp $BASE/mi3lab-workzone-vla/configs/models/ar1_5_expert_2b_cotrain.yaml $SFT_DIR/configs/models/
cp $BASE/mi3lab-workzone-vla/configs/sft_action_2b_multiwindow.yaml $SFT_DIR/configs/

mkdir -p $BASE/checkpoints/sft_action_2b_multiwindow
cd $SFT_DIR
torchrun \
    --nproc_per_node=2 \
    --master_port=29527 \
    $BASE/mi3lab-workzone-vla/training/train_action_2b.py \
    --config-path configs \
    --config-name sft_action_2b_multiwindow \
    trainer.deepspeed=null \
    trainer.gradient_accumulation_steps=8 \
    ${EXTRA_OVERRIDES}
