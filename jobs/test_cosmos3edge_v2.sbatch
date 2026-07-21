#!/bin/bash
#SBATCH --job-name=cosmos3e_v2
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_v2_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_v2_%j.err

# Teste v2 do Cosmos3-Edge: ambiente completo do cosmos_framework (uv sync
# --group=cu128, SEM -train pra evitar transformer-engine/torchao que podem
# ter o mesmo problema de compilacao JIT que o DeepSpeed teve), usando a
# classe nativa Cosmos3EdgeForConditionalGeneration em vez do pacote leve
# incompleto. Internet necessaria (download de wheels + checkpoint ~3.86GB).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
# so' inferencia -- evita que o framework tente registrar o backend de storage
# de treino (S3/multi-storage-client), que exigiria instalar o grupo -train
# inteiro so' por causa de um import incondicional em registry_utils.py
export COSMOS_TRAINING=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

cd $CF
echo "[setup] uv sync --group=cu128 (ambiente completo, sem -train)..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -60

echo "[setup] boto3 e qwen-vl-utils nao estao no grupo cu128 (so' no 'train'),"
echo "        mas sao imports incondicionais em pretrained_models_downloader.py"
echo "        e nemotron3densevl_processor.py -- instalando DEPOIS do uv sync"
echo "        pra nao serem removidos por ele..."
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -15

echo
echo "[run] rodando o teste de raciocinio..."
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/test_cosmos3edge_v2.py
