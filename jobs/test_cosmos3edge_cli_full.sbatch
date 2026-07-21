#!/bin/bash
#SBATCH --job-name=cosmos3e_full
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_full_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_full_%j.err

# Teste completo (5 perguntas v16, mesma cena Boston) via CLI oficial do
# Cosmos3-Edge, validado no job 194663 (GATE isolado -> resposta correta "Yes"
# com raciocinio coerente). Roda todas de uma vez -- o modelo carrega 1x so'.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -20
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -10

echo
echo "[run] rodando as 5 perguntas..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/workzone/*_test.json" \
    -o outputs/workzone_full_test \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "=== RESULTADOS ==="
for f in gate_test ego_test active_test sign_test desc_test; do
    echo "--- [$f] ---"
    python3 -c "
import json
p = 'outputs/workzone_full_test/$f/sample_outputs.json'
try:
    d = json.load(open(p))
    print('prompt:', d['args']['prompt'])
    print('status:', d['status'])
    print('resposta:', d['outputs'][0]['content']['reasoner_text'])
except Exception as e:
    print('ERRO lendo', p, ':', e)
"
    echo
done
