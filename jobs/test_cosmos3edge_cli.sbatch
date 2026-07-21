#!/bin/bash
#SBATCH --job-name=cosmos3e_cli
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_cli_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_cli_%j.err

# Teste do Cosmos3-Edge via CLI OFICIAL do framework (cosmos_framework.scripts.inference),
# em vez de model.generate() manual -- o README/docs/inference.md documenta esta como a
# forma testada e suportada, com --checkpoint-path Cosmos3-Edge resolvendo pro mesmo
# nvidia/Cosmos3-Edge que ja temos em cache local. A tentativa manual anterior (v2)
# carregou modelo+processador corretos mas gerou lixo -- suspeita de que falta algo
# no caminho de geracao que so o pipeline oficial monta certo (system prompt, sampling
# params, etc.)

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

cd $CF
echo "[setup] uv sync --group=cu128 (ambiente completo, sem -train)..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -30

echo "[setup] boto3 + qwen-vl-utils (imports incondicionais fora do grupo cu128)..."
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -15

echo
echo "[run] rodando via CLI oficial (cosmos_framework.scripts.inference)..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/workzone/gate_test.json \
    -o outputs/workzone_gate_test \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "[result] sample_outputs.json:"
find outputs/workzone_gate_test -name "sample_outputs.json" -exec cat {} \;
echo
echo "[result] reasoner_text.txt (se existir):"
find outputs/workzone_gate_test -name "reasoner_text.txt" -exec cat {} \;
