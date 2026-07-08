#!/bin/bash
#SBATCH --job-name=alp_trt_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/trt_bench_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/trt_bench_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Inferencia real TensorRT INT4 — medindo latencia ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_ego.json \
    --outputFile $ENG/output_ego.json \
    --warmup 3 \
    --dumpProfile \
    --dumpOutput 2>&1

echo "=== BENCH DONE ==="
cat $ENG/output_ego.json 2>&1
