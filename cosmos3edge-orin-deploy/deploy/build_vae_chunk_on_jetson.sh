#!/usr/bin/env bash
# Build the Cosmos3-Edge VAE encoder engines (chunked) on the target Jetson.
# TensorRT engines are hardware/version specific; do not copy the A100 engines.
#
# WHY CHUNKED: the original monolithic encoder ONNX (encoder-av61-fp16.onnx)
# unrolled all 16 iterations of the VAE's causal streaming loop into one graph
# (495 Conv = 16 x ~31, 44366 nodes).  Building an engine from it needs
# ~125.8 GB of host RAM, which does not fit on a 64 GB AGX Orin -- so that
# graph simply cannot be built on the target.  The chunked graphs contain one
# loop iteration each (~533 nodes, 32 Conv) and build in <4 GB.
#
# Window semantics are preserved exactly: the host resets the cache every tick
# and chains chunk0 -> chunk1 -> chunkN x14, which reproduces the monolithic
# output (validated: max abs err 5.9e-03, fp16 noise).  Measured on A100:
# 19.1 + 40.6 + 14 x 41.5 = 640.9 ms/tick, versus 8679 ms for the current
# ONNX Runtime CUDA EP path.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <chunk-onnx-dir> [engine-dir]" >&2
    echo "  <chunk-onnx-dir> holds encoder-chunk{0,1,N}-fp16.onnx" >&2
    exit 2
fi

onnx_dir=$(realpath "$1")
engine_dir=${2:-"$PWD/cosmos3edge-vae-chunk-engines"}
mkdir -p "$engine_dir"
engine_dir=$(realpath "$engine_dir")

python_bin=${PYTHON_BIN:-python3}
trtexec_bin=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$trtexec_bin" ]] || trtexec_bin=$(command -v trtexec || true)
[[ -x "$trtexec_bin" ]] || { echo "trtexec not found; set TRTEXEC=<path>" >&2; exit 1; }

fold_script="$(dirname "$(realpath "$0")")/../runtime/fold_vae_chunk_graph.py"
[[ -f "$fold_script" ]] || { echo "Missing: $fold_script" >&2; exit 1; }

for tag in chunk0 chunk1 chunkN; do
    src="$onnx_dir/encoder-${tag}-fp16.onnx"
    [[ -f "$src" ]] || { echo "Missing ONNX: $src" >&2; exit 1; }
done

echo "trtexec: $trtexec_bin"
"$trtexec_bin" --version 2>/dev/null | head -1 || true
echo

for tag in chunk0 chunk1 chunkN; do
    src="$onnx_dir/encoder-${tag}-fp16.onnx"
    folded="$engine_dir/encoder-${tag}-folded.onnx"
    engine="$engine_dir/${tag}.engine"

    echo "=== $tag ==="
    # The export leaves one `If` node (mid_block/attentions) whose branches
    # have different ranks; TensorRT's parser rejects it.  Because the graph
    # is fully static, ORT constant folding resolves the condition and drops
    # the node entirely -- do NOT try to equalise the branch ranks by hand,
    # that silently changes tensor ranks and breaks a downstream Transpose.
    echo "  folding (drops the If node, folds Pad into Conv)..."
    "$python_bin" "$fold_script" "$src" "$folded"

    echo "  building engine..."
    "$trtexec_bin" \
        --onnx="$folded" \
        --fp16 \
        --builderOptimizationLevel=1 \
        --memPoolSize=workspace:4096 \
        --saveEngine="$engine" \
        --skipInference

    echo "  -> $engine"
    echo
done

echo "Jetson VAE chunk engines written to: $engine_dir"
echo
echo "Point the runtime at them with:"
echo "  export COSMOS3EDGE_VAE_CHUNK_ENGINES=$engine_dir"
