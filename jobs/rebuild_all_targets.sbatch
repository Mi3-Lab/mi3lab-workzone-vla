#!/bin/bash
#SBATCH --job-name=alp_rebuild_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/rebuild_all_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/rebuild_all_%j.err

module load singularityce/4.1.0
BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SIF=$BASE/containers/tensorrt.sif
PROJ=$BASE/TensorRT-Edge-LLM

echo "=== Rebuild de TODOS os alvos (sem apagar cache) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv $SIF bash -c "
cd $PROJ/build
make -j16 llm_build llm_inference visual_build NvInfer_edgellm_plugin
"

echo "=== BUILD DONE ==="
find $PROJ/build -maxdepth 3 -type f -executable -exec ls -la {} \;
find $PROJ/build -iname "libNvInfer_edgellm_plugin*" -exec ls -la {} \;
