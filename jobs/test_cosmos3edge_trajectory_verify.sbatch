#!/bin/bash
#SBATCH --job-name=cosmos3e_traj_v
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_traj_v_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_traj_v_%j.err

# Verificacao mais a fundo da trajetoria prevista no job 194709: usa a propria
# acao prevista (inverse_dynamics) como entrada do forward_dynamics do mesmo
# modelo, gerando um rollout de video. Se o video gerado parecer com o
# boston.mp4 real (mesma direcao/velocidade aparente de movimento), e' evidencia
# de autoconsistencia -- nao e' validacao contra odometria real (nao temos),
# mas e' o teste mais forte que da pra fazer sem dado externo.
#
# Tambem gera o plot da trajetoria (top-down) e organiza tudo em
# outputs/cosmos3edge_trajectory_test/ pro usuario conferir.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -20
~/.local/bin/uv pip install boto3 qwen-vl-utils matplotlib 2>&1 | tail -10

echo
echo "[run 1/2] forward_dynamics rollout usando a acao prevista..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/workzone/trajectory_rollout.json \
    -o outputs/workzone_trajectory_rollout \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

ROLLOUT_MP4=$(find outputs/workzone_trajectory_rollout -name "vision.mp4")
if [ -n "$ROLLOUT_MP4" ]; then
    cp "$ROLLOUT_MP4" "$DELIVER/forward_dynamics_rollout_from_predicted_action.mp4"
    echo "[ok] rollout copiado pra $DELIVER/forward_dynamics_rollout_from_predicted_action.mp4"
else
    echo "[ERRO] nao achei vision.mp4 gerado pelo forward_dynamics"
fi

echo
echo "[run 2/2] plot da trajetoria (top-down + componentes)..."
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/plot_trajectory.py \
    "$DELIVER/predicted_action_60x9.json" \
    "$DELIVER/trajectory_plot.png"

echo
echo "=== ARQUIVOS FINAIS EM $DELIVER ==="
ls -la "$DELIVER"
