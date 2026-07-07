#!/bin/bash
#SBATCH --job-name=alp_quant_v2
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:55:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/quant_int4_v2_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/quant_int4_v2_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
source $BASE/.venv-edgellm/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton-cache-$SLURM_JOB_ID

echo "=== Quantizacao INT4 AWQ v2 (calibracao 2000 amostras) — workzone 2B stage6 ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Calibracao: calib_workzone_v2.jsonl (2000 amostras: 1000 combined + 300 egostate + 300 signtext + 400 general)"

tensorrt-edgellm-quantize llm \
    --model_dir  $BASE/models/workzone-2b-stage6-hf \
    --output_dir $BASE/models/workzone-2b-stage6-int4awq-v2 \
    --quantization int4_awq \
    --dataset $BASE/data/roadwork/calib_workzone_v2.jsonl \
    --num_samples 2000

echo "=== DONE ==="
du -sh $BASE/models/workzone-2b-stage6-int4awq-v2
