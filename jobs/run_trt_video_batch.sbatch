#!/bin/bash
#SBATCH --job-name=alp_trt_video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/trt_video_batch_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/trt_video_batch_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Inferencia TensorRT INT4 nos 4 videos (685 requisicoes) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$(date +%s.%N)
singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_video_batch.json \
    --outputFile $ENG/output_video_batch.json \
    --warmup 3 \
    --dumpProfile 2>&1
END=$(date +%s.%N)

echo "=== BATCH DONE ==="
echo "Tempo total (wall clock, inclui overhead de I/O): $(echo "$END - $START" | bc) s"
ls -la $ENG/output_video_batch.json
