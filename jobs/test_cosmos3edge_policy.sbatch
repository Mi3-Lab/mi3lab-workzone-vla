#!/bin/bash
#SBATCH --job-name=cosmos3e_policy
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_policy_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_policy_%j.err

# model_mode=policy: planejamento futuro a partir de 1 frame de observacao
# (sample_frame.jpg, mesma cena boston) -- mesma TAREFA que o Alpamayo faz
# (prever trajetoria futura), diferente do inverse_dynamics (recupera
# movimento PASSADO a partir de video real) testado antes.

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
echo "[run] policy mode (planejamento futuro) na cena boston..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/workzone/policy_test.json \
    -o outputs/workzone_policy_test \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
python3 -c "
import json
d = json.load(open('outputs/workzone_policy_test/policy_test/sample_outputs.json'))
action = d['outputs'][0]['content']['action']
json.dump(action, open('$DELIVER/cosmos3edge_policy_action_60x9.json', 'w'), indent=2)
print('acao (policy) salva:', len(action), 'x', len(action[0]))
"
echo "[ok] salvo em $DELIVER/cosmos3edge_policy_action_60x9.json"
