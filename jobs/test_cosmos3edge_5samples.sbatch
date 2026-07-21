#!/bin/bash
#SBATCH --job-name=cosmos3e_5s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_5s_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_5s_%j.err

# Roda 5 seeds x 2 modos (inverse_dynamics + policy) numa carga so' do modelo,
# igual ao K=5 usado no Alpamayo (pedido explicito do usuario pra comparacao
# justa). Tambem re-roda as 5 perguntas de raciocinio (reasoner) pra ter tudo
# fresco pro video final.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -20
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -10

echo
echo "[run 1/3] 5 seeds x inverse_dynamics + policy (10 amostras, 1 carga do modelo)..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/workzone/trajectory_test_seed*.json" "inputs/workzone/policy_test_seed*.json" \
    -o outputs/workzone_5samples \
    --checkpoint-path Cosmos3-Edge \
    --no-guardrails

echo
echo "[run 2/3] 5 perguntas de raciocinio (reasoner, fresco)..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/workzone/*_test.json" \
    -o outputs/workzone_full_test_fresh \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "[run 3/3] extraindo e organizando resultados..."
python3 -c "
import json, glob, os

DELIVER = '$DELIVER'

for mode, prefix in [('trajectory_test', 'inverse_dynamics'), ('policy_test', 'policy')]:
    samples = []
    for i in range(5):
        p = f'outputs/workzone_5samples/{mode}_seed{i}/sample_outputs.json'
        d = json.load(open(p))
        samples.append(d['outputs'][0]['content']['action'])
    json.dump(samples, open(f'{DELIVER}/cosmos3edge_{prefix}_5samples.json', 'w'))
    print(f'{prefix}: {len(samples)} amostras salvas')

# perguntas de raciocinio frescas
qa = {}
for name in ['gate_test', 'ego_test', 'active_test', 'sign_test', 'desc_test']:
    p = f'outputs/workzone_full_test_fresh/{name}/sample_outputs.json'
    d = json.load(open(p))
    qa[name] = {
        'prompt': d['args']['prompt'],
        'answer': d['outputs'][0]['content']['reasoner_text'],
    }
json.dump(qa, open(f'{DELIVER}/cosmos3edge_qa_fresh.json', 'w'), indent=2, ensure_ascii=False)
print('QA salvo')
"
echo "[ok] tudo em $DELIVER"
ls -la "$DELIVER"
