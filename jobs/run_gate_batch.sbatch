#!/bin/bash
#SBATCH --job-name=alp_gate
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/gate_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/gate_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Pergunta de GATE (existencia de obra) -- 466 frames (positivos + negativos) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_gate_batch.json \
    --outputFile $ENG/output_gate_batch.json \
    --warmup 3 \
    --dumpProfile 2>&1 | tail -15

echo "=== GATE BATCH DONE ==="
