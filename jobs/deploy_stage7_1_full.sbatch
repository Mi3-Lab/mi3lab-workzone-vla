#!/bin/bash
#SBATCH --job-name=alp_s71_deploy
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/s71_deploy_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/s71_deploy_%j.err

# Pipeline completo de deployment do stage7 (hard negatives):
#   1. quantiza INT4 AWQ (mesma receita v1: 512 amostras calib_workzone)
#   2. exporta ONNX (llm + visual -- visual treinou com lr 0.1x no stage7)
#   3. compila engines TensorRT (dir novo, stage6 preservado)
#   4. re-roda os 4 batches de benchmark no engine novo

set -e
module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG6=$BASE/models/workzone-2b-stage6-engines
ENG=$BASE/models/workzone-2b-stage7-1-engines
ONNX=$BASE/models/workzone-2b-stage7-1-onnx

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton-cache-$SLURM_JOB_ID

echo "=== [1/4] Quantizacao INT4 AWQ (receita v1) ==="
grep -o '"tie_word_embeddings": [a-z]*' $BASE/models/workzone-2b-stage7-1-hf/config.json | head -2
tensorrt-edgellm-quantize llm \
    --model_dir  $BASE/models/workzone-2b-stage7-1-hf \
    --output_dir $BASE/models/workzone-2b-stage7-1-int4awq \
    --quantization int4_awq \
    --dataset $BASE/data/roadwork/calib_workzone.jsonl \
    --num_samples 512

echo "--- garantindo tie_word_embeddings=false no checkpoint quantizado ---"
python3 - <<'EOF'
import json
p = "/data/wesleyferreiramaia/wokzone-alpamayo/models/workzone-2b-stage7-1-int4awq/config.json"
c = json.load(open(p))
ch = []
if c.get("tie_word_embeddings") is True:
    c["tie_word_embeddings"] = False; ch.append("top")
tc = c.get("text_config")
if isinstance(tc, dict) and tc.get("tie_word_embeddings") is True:
    tc["tie_word_embeddings"] = False; ch.append("text_config")
json.dump(c, open(p, "w"), indent=2)
print("fix aplicado:", ch or "ja estava false")
EOF

echo "=== [2/4] Export ONNX (llm + visual) ==="
tensorrt-edgellm-export \
    $BASE/models/workzone-2b-stage7-1-int4awq \
    $ONNX
deactivate

echo "=== [3/4] Build engines ==="
mkdir -p $ENG
singularity exec --nv --env LD_PRELOAD=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_build \
    --onnxDir $ONNX/llm \
    --engineDir $ENG/llm \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096 2>&1 | tail -3
singularity exec --nv --env LD_PRELOAD=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/multimodal/visual_build \
    --onnxDir $ONNX/visual \
    --engineDir $ENG \
    --minImageTokens 128 \
    --maxImageTokens 512 \
    --maxImageTokensPerImage 512 2>&1 | tail -3

echo "=== [4/4] Batches de benchmark no engine stage7 ==="
run_batch () {
    echo "--- $1 ---"
    singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
        $PROJ/build/examples/llm/llm_inference \
        --engineDir $ENG/llm \
        --multimodalEngineDir $ENG/visual \
        --inputFile $ENG6/$1 \
        --outputFile $ENG/$2 \
        --warmup 1 \
        --dumpProfile 2>&1 | grep -E "Processing complete|Average Time per Run|Tokens/Second" | head -5
}
run_batch input_video_batch.json            output_video_batch_s71.json
run_batch input_negcontrol_video_batch.json output_negcontrol_video_batch_s71.json
run_batch input_gate_batch.json             output_gate_batch_s71.json
run_batch input_negcontrol_batch.json       output_negcontrol_batch_s71.json

echo "=== STAGE7 DEPLOY PIPELINE DONE ==="
du -sh $BASE/models/workzone-2b-stage7-1-int4awq $ONNX $ENG