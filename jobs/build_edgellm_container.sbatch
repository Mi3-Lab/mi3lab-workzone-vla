#!/bin/bash
#SBATCH --job-name=alp_build_trt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/build_trt_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/build_trt_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM

echo "=== Build TensorRT-Edge-LLM dentro do container (x86/GPU) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rm -rf $PROJ/build
mkdir -p $PROJ/build

singularity exec --nv $SIF bash -c "
cd $PROJ/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCUDA_CTK_VERSION=12.6 \
    -DENABLE_CUTE_DSL=OFF \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs
" 2>&1

echo "=== CONFIGURE DONE, iniciando build ==="

singularity exec --nv $SIF bash -c "
cd $PROJ/build
make -j16 llm_build llm_inference visual_build NvInfer_edgellm_plugin
" 2>&1

echo "=== BUILD DONE ==="
ls -la $PROJ/build/examples/llm/ $PROJ/build/examples/multimodal/ 2>&1
