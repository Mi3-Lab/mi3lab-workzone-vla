#!/usr/bin/env bash
# Build the Cosmos3-Edge reasoner and visual engines on Jetson.
# MoT/VAE are executed separately by ONNX Runtime TensorRT EP.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <TensorRT-Edge-LLM-dir> <reasoner-ONNX-dir> <visual-ONNX-dir> [engine-dir]" >&2
    exit 2
fi

edge_llm_dir=$(realpath "$1")
reasoner_dir=$(realpath "$2")
visual_dir=$(realpath "$3")
engine_dir=${4:-"$PWD/cosmos3edge-complete-engine"}
mkdir -p "$engine_dir"
engine_dir=$(realpath "$engine_dir")

llm_builder="$edge_llm_dir/build/examples/llm/llm_build"
visual_builder="$edge_llm_dir/build/examples/multimodal/visual_build"
plugin="$edge_llm_dir/build/libNvInfer_edgellm_plugin.so"

[[ -x "$llm_builder" ]] || { echo "Missing builder: $llm_builder" >&2; exit 1; }
[[ -x "$visual_builder" ]] || { echo "Missing builder: $visual_builder" >&2; exit 1; }
[[ -f "$plugin" ]] || { echo "Missing plugin: $plugin" >&2; exit 1; }
[[ -f "$reasoner_dir/model.onnx" ]] || { echo "Missing reasoner ONNX" >&2; exit 1; }
[[ -f "$visual_dir/model.onnx" ]] || { echo "Missing visual ONNX" >&2; exit 1; }

export EDGELLM_PLUGIN_PATH="$plugin"
export LD_PRELOAD="$plugin${LD_PRELOAD:+:$LD_PRELOAD}"

"$llm_builder" \
    --onnxDir "$reasoner_dir" \
    --engineDir "$engine_dir" \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096

"$visual_builder" \
    --onnxDir "$visual_dir" \
    --engineDir "$engine_dir" \
    --minImageTokens 299 \
    --maxImageTokens 299 \
    --maxImageTokensPerImage 299

echo "Reasoner + visual Jetson engines written to: $engine_dir"
echo "Reasoner: INT4 AWQ W4A16; vision encoder: FP16; MRoPE: Cosmos3-Edge [24,20,20]."
