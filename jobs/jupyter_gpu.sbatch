#!/bin/bash
#SBATCH --job-name=alpamayo_jupyter
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/jupyter_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/jupyter_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate

export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

NODE=$(hostname)
PORT=8888

echo "================================================"
echo "  Jupyter Lab — Alpamayo Inference"
echo "  Node : $NODE"
echo "  Port : $PORT"
echo "  GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "  SSH tunnel (rodar no seu terminal local):"
echo "  ssh -L ${PORT}:${NODE}:${PORT} wesleyferreiramaia@login.rc.ucmerced.edu"
echo ""
echo "  Depois abra no browser: http://localhost:${PORT}"
echo "================================================"

# Instalar jupyter se necessário
pip install jupyterlab ipywidgets -q 2>/dev/null

jupyter lab \
    --no-browser \
    --port=$PORT \
    --ip=0.0.0.0 \
    --notebook-dir=$BASE \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.token='' \
    --ServerApp.password=''
