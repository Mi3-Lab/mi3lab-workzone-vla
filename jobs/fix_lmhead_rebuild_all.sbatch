#!/bin/bash
#SBATCH --job-name=alp_lmhead_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/lmhead_fix_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/lmhead_fix_%j.err

# Fix da causa raiz do bug do 1o token: tie_word_embeddings=True fazia o
# loader do Edge-LLM sobrescrever o lm_head treinado com a embedding
# (checkpoint/loader.py:189). Configs ja corrigidos para false; este job:
#   1. re-exporta o ONNX do LLM (agora com o lm_head verdadeiro)
#   2. recompila llm_inference limpo (binario atual tem supressao antiga do
#      token 52728 embutida)
#   3. reconstroi o engine do LLM (visual inalterado)
#   4. re-roda o batch dos 685 requests dos 4 videos

set -e
module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ONNX=$BASE/models/workzone-2b-stage6-onnx
ENG=$BASE/models/workzone-2b-stage6-engines

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo "=== [1/4] Re-export ONNX do LLM (lm_head verdadeiro) ==="
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
grep -o '"tie_word_embeddings": [a-z]*' $BASE/models/workzone-2b-stage6-int4awq/config.json
tensorrt-edgellm-export --skip-visual \
    $BASE/models/workzone-2b-stage6-int4awq \
    $ONNX
deactivate

echo "=== [2/4] Recompilar llm_inference (fonte limpo) ==="
singularity exec --nv $SIF bash -c "cd $PROJ/build && make -j16 llm_inference" | tail -3
ls -la $PROJ/build/examples/llm/llm_inference

echo "=== [3/4] Rebuild engine LLM ==="
singularity exec --nv --env LD_PRELOAD=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_build \
    --onnxDir $ONNX/llm \
    --engineDir $ENG/llm \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096 2>&1 | tail -5

echo "=== [4/4] Batch 685 requests (4 videos) ==="
echo "--- sanity: EGO single request ---"
singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_ego.json \
    --outputFile $ENG/output_lmheadfix_ego.json \
    --warmup 0 \
    --dumpOutput 2>&1 | grep "Response for" || true

echo "--- batch completo ---"
singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_video_batch.json \
    --outputFile $ENG/output_video_batch_lmheadfix.json \
    --warmup 3 \
    --dumpProfile 2>&1 | tail -30

echo "=== LMHEAD FIX PIPELINE DONE ==="
