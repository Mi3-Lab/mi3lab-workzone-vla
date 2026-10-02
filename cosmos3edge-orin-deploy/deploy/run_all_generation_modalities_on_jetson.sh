#!/usr/bin/env bash
# Execute forward dynamics, joint policy and I2V from the operational package.

set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <package-root> [work-dir]" >&2
    exit 2
fi
package=$(realpath "$1")
work=${2:-$package/outputs/modalities}
mkdir -p "$work/engines"
work=$(realpath "$work")
python_bin=${PYTHON_BIN:-python3}
export PYTHONPATH=$package/third_party/diffusers-cosmos3/src:$package/runtime${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1
export COSMOS3EDGE_ORT_TENSORRT=1
export COSMOS3EDGE_TRT_CACHE=$work/engines/ort-tensorrt

"$python_bin" "$package/runtime/cosmos3edge_forward_onnx.py" \
    --checkpoint "$package/pipeline-config" \
    --image "$package/tests/fixtures/example_action_fd_umi_first_frame.png" \
    --actions "$package/tests/fixtures/example_action_fd_umi_action_chunks.json" \
    --vae-encoder "$package/onnx/vae-fp16-shared/onnx/encoder-umi17-fp16.onnx" \
    --vae-decoder "$package/onnx/vae-fp16-shared/onnx/decoder-umi17-fp16.onnx" \
    --forward-transformer "$package/onnx/mot-fp16-shared/onnx/forward-umi16-fp16.onnx" \
    --output-video "$work/forward_umi.mp4" --output-json "$work/forward_umi.json"

"$python_bin" "$package/runtime/cosmos3edge_policy_onnx.py" \
    --checkpoint "$package/pipeline-config" \
    --image "$package/tests/fixtures/example_action_fd_umi_first_frame.png" \
    --actions-spec "$package/tests/fixtures/example_action_fd_umi_action_chunks.json" \
    --vae-encoder "$package/onnx/vae-fp16-shared/onnx/encoder-umi17-fp16.onnx" \
    --vae-decoder "$package/onnx/vae-fp16-shared/onnx/decoder-umi17-fp16.onnx" \
    --policy-transformer "$package/onnx/mot-fp16-shared/onnx/policy-umi16-fp16.onnx" \
    --output-video "$work/policy_umi.mp4" --output-json "$work/policy_umi.json"

"$python_bin" "$package/runtime/cosmos3edge_i2v_onnx.py" \
    --checkpoint "$package/pipeline-config" \
    --image "$package/tests/fixtures/example_i2v_input.jpg" \
    --prompt "$package/tests/fixtures/example_i2v_prompt.json" \
    --vae-encoder "$package/onnx/vae-fp16-shared/onnx/encoder-i2v121-fp16.onnx" \
    --vae-decoder "$package/onnx/vae-fp16-shared/onnx/decoder-i2v121-fp16.onnx" \
    --transformer-cond "$package/onnx/mot-fp16-shared/onnx/i2v-121-cond-fp16.onnx" \
    --transformer-uncond "$package/onnx/mot-fp16-shared/onnx/i2v-121-uncond-fp16.onnx" \
    --output-video "$work/i2v_121.mp4" --output-json "$work/i2v_121.json"

echo "Generation modality results: $work"
