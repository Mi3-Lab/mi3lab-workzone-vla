#!/bin/bash
#SBATCH --job-name=alp_engines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/build_engines_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/build_engines_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ONNX=$BASE/models/workzone-2b-stage6-onnx
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Build engines TensorRT (checkpoint v1, INT4 AWQ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p $ENG

echo "--- llm_build ---"
singularity exec --nv --env LD_PRELOAD=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_build \
    --onnxDir $ONNX/llm \
    --engineDir $ENG/llm \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096 2>&1

echo "--- visual_build ---"
singularity exec --nv --env LD_PRELOAD=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/multimodal/visual_build \
    --onnxDir $ONNX/visual \
    --engineDir $ENG \
    --minImageTokens 128 \
    --maxImageTokens 512 \
    --maxImageTokensPerImage 512 2>&1

echo "=== ENGINES BUILD DONE ==="
find $ENG -type f -exec ls -la {} \;
