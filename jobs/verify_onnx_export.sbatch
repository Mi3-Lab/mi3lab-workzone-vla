#!/bin/bash
#SBATCH --job-name=c3e_verify_onnx
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_verify_onnx_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_verify_onnx_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache

cd $CF
~/.local/bin/uv pip install onnxruntime-gpu 2>&1 | tail -15
echo
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/verify_onnx_export.py
