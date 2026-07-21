#!/bin/bash
#SBATCH --job-name=cosmos3e_cascade
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_cascade_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_cascade_%j.err

# Roda o Cosmos3-Edge reasoner em 25 frames reais do boston.mp4 (cadencia
# 0.6s, mesma calibracao da CascadeStateMachine do pipeline 2B) x 4 perguntas
# (GATE/EGO/DESC/SIGN) = 100 amostras, 1 carga do modelo. Pedido explicito do
# usuario: texto tem que ir "gerando o output" conforme o video passa, igual
# ao pipeline 2B ja fazia -- aqui reusamos a MESMA CascadeStateMachine, so'
# trocando a fonte das respostas (Cosmos3-Edge em vez do 2B).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -15
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -5

echo
echo "[run] 100 amostras (25 frames x 4 perguntas), 1 carga do modelo..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/workzone/frames/*.json" \
    -o outputs/workzone_cascade_frames \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "[extract] organizando respostas por frame..."
python3 -c "
import json, glob, re, os

CF = '$CF'
DELIVER = '$DELIVER'
os.makedirs(DELIVER, exist_ok=True)

results = {}
pat = re.compile(r'f(\d{3})_(\w+)')
for d in glob.glob(f'{CF}/outputs/workzone_cascade_frames/*/sample_outputs.json'):
    name = d.split('/')[-2]
    m = pat.match(name)
    if not m:
        print(f'AVISO: nome inesperado {name}')
        continue
    frame_idx, tag = int(m.group(1)), m.group(2)
    data = json.load(open(d))
    answer = data['outputs'][0]['content']['reasoner_text']
    results.setdefault(frame_idx, {})[tag] = answer

print(f'{len(results)} frames com respostas coletadas')
json.dump(results, open(f'{DELIVER}/cosmos3edge_cascade_frames_raw.json', 'w'), indent=2, ensure_ascii=False)
print('salvo em cosmos3edge_cascade_frames_raw.json')
"
echo "[ok]"
