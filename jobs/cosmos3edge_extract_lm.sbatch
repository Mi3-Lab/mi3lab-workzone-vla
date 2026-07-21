#!/bin/bash
#SBATCH --job-name=c3e_extract_lm
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_extract_lm_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_extract_lm_%j.err

# Passo 1/4 do pipeline de quantizacao do Cosmos3-Edge: extrai a torre de
# texto (autoregressiva) do checkpoint OmniMoT pro formato puro
# Qwen3VLForConditionalGeneration (oficialmente suportado pelo
# TensorRT-Edge-LLM). A torre de difusao/acao NAO entra aqui -- fica de fora
# da quantizacao por enquanto (ver investigacao: nao ha caminho pronto pra
# exportar ela pro TensorRT-Edge-LLM).

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -15
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -5

echo
echo "[run] extraindo torre de texto do Cosmos3-Edge -> Qwen3VL HF safetensors..."
.venv/bin/python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
    --checkpoint-path Cosmos3-Edge \
    -o $BASE/models/cosmos3edge-lm-hf

echo
echo "[ok] conteudo extraido:"
ls -la $BASE/models/cosmos3edge-lm-hf/
du -sh $BASE/models/cosmos3edge-lm-hf/
echo
echo "[config] verificando arquitetura/config.json:"
python3 -c "
import json
c = json.load(open('$BASE/models/cosmos3edge-lm-hf/config.json'))
print('architectures:', c.get('architectures'))
print('model_type:', c.get('model_type'))
print('tie_word_embeddings:', c.get('tie_word_embeddings'))
tc = c.get('text_config', {})
print('text_config.tie_word_embeddings:', tc.get('tie_word_embeddings'))
print('text_config.hidden_size:', tc.get('hidden_size'))
print('text_config.num_hidden_layers:', tc.get('num_hidden_layers'))
"
