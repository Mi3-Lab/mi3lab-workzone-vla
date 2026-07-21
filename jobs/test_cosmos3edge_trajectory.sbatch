#!/bin/bash
#SBATCH --job-name=cosmos3e_traj
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_traj_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_traj_%j.err

# Teste de trajetoria (inverse_dynamics, domain_name=av, view_point=ego_view)
# do Cosmos3-Edge no nosso video real videos/boston.mp4 -- mesma cena usada
# nos testes de raciocinio (194663/194667). So' inferencia, formato de acao
# AV = 9D (translacao+rotacao), nao steering/throttle cru.

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
echo "[run] rodando inverse_dynamics em videos/boston.mp4..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/workzone/trajectory_test.json \
    -o outputs/workzone_trajectory_test \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "=== RESULTADO COMPLETO (sample_outputs.json) ==="
find outputs/workzone_trajectory_test -name "sample_outputs.json" -exec cat {} \;
echo
