#!/bin/bash
#SBATCH --job-name=alp_annval
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/annval_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/annval_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage7-1-engines

echo "=== Validacao vs anotacao humana: 27 snippets, 8202 requisicoes (stage7_1 INT4) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_annval_batch.json \
    --outputFile $ENG/output_annval_batch.json \
    --warmup 1 \
    --dumpProfile 2>&1 | grep -E "Processing complete|Average" | head -6

echo "=== ANNVAL BATCH DONE ==="
