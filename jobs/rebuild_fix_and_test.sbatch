#!/bin/bash
#SBATCH --job-name=alp_fix_test
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/fix_test_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/fix_test_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM
ENG=$BASE/models/workzone-2b-stage6-engines

echo "=== Recompilando llm_inference com fix de supressao do token .DATA ==="
singularity exec --nv $SIF bash -c "
cd $PROJ/build
make -j16 llm_inference
"
echo "=== BUILD DONE ==="
ls -la $PROJ/build/examples/llm/llm_inference

echo ""
echo "--- Teste 1: EGO (mesmo request que dava .DATAINSIDE...) ---"
singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_ego.json \
    --outputFile $ENG/output_fixtest.json \
    --warmup 0 \
    --dumpOutput 2>&1 | grep "Response for"

echo ""
echo "--- Teste 2: batch completo dos 4 videos (685 requisicoes) ---"
singularity exec --nv --env EDGELLM_PLUGIN_PATH=$PROJ/build/libNvInfer_edgellm_plugin.so $SIF \
    $PROJ/build/examples/llm/llm_inference \
    --engineDir $ENG/llm \
    --multimodalEngineDir $ENG/visual \
    --inputFile $ENG/input_video_batch.json \
    --outputFile $ENG/output_video_batch_fixed.json \
    --warmup 3 \
    --dumpProfile 2>&1 | tail -40

echo "=== FIX TEST DONE ==="
