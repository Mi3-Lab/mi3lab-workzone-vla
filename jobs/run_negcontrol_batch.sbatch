#!/bin/bash
#SBATCH --job-name=alp_negctrl
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/negctrl_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/negctrl_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Controle negativo (sem zona de obra) -- engine INT4 fixed, 2245 requisicoes ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_negcontrol_batch.json \
    --outputFile $ENG/output_negcontrol_batch.json \
    --warmup 3 \
    --dumpProfile 2>&1 | tail -40

echo "=== NEGCONTROL BATCH DONE ==="
