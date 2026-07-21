#!/bin/bash
#SBATCH --job-name=cosmos3e_test
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_test_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/cosmos3e_test_%j.err

# Teste do Cosmos3-Edge (novo, lancado 15/07/2026) nas mesmas perguntas de
# workzone do nosso 2B. Ambiente EFEMERO via `uv run --script` (torch 2.11 +
# transformers 5.8, bem mais novo que nosso venv do alpamayo-recipes) -- nao
# mexe em nada do que ja temos. PRECISA de internet (download do checkpoint
# ~3.86GB do HF + wheels do PyPI) -- SEM HF_HUB_OFFLINE de proposito.

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

~/.local/bin/uv run --script $BASE/mi3lab-workzone-vla/diagnostics/test_cosmos3edge_reasoning.py
