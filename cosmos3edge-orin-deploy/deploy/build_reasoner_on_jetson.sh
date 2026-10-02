#!/usr/bin/env bash
# Build the Cosmos3-Edge INT4 reasoner engine on the target Jetson.
# TensorRT engines are hardware/version specific; do not copy the A100 engine.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <TensorRT-Edge-LLM-dir> <ONNX-dir> [engine-dir]" >&2
    exit 2
fi

edge_llm_dir=$(realpath "$1")
onnx_dir=$(realpath "$2")
engine_dir=${3:-"$PWD/cosmos3edge-reasoner-int4-engine"}
mkdir -p "$engine_dir"
engine_dir=$(realpath "$engine_dir")

builder="$edge_llm_dir/build/examples/llm/llm_build"
plugin="$edge_llm_dir/build/libNvInfer_edgellm_plugin.so"

[[ -x "$builder" ]] || { echo "Missing builder: $builder" >&2; exit 1; }
[[ -f "$plugin" ]] || { echo "Missing plugin: $plugin" >&2; exit 1; }
[[ -f "$onnx_dir/llm/model.onnx" ]] || {
    echo "Missing ONNX: $onnx_dir/llm/model.onnx" >&2
    exit 1
}

export EDGELLM_PLUGIN_PATH="$plugin"
export LD_PRELOAD="$plugin${LD_PRELOAD:+:$LD_PRELOAD}"

"$builder" \
    --onnxDir "$onnx_dir/llm" \
    --engineDir "$engine_dir" \
    --maxBatchSize 1 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096

echo "Jetson engine written to: $engine_dir"
