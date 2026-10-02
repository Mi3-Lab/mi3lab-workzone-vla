#!/usr/bin/env bash
# End-to-end Boston acceptance run: reasoner + future AV policy, every 100 ms.

set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <package-root> <TensorRT-Edge-LLM-dir> [work-dir]" >&2
    exit 2
fi

package=$(realpath "$1")
edge=$(realpath "$2")
work=${3:-"$package/outputs/boston_10hz"}
mkdir -p "$work/frames" "$work/engines"
work=$(realpath "$work")
python_bin=${PYTHON_BIN:-python3}

reasoner_onnx=$package/onnx/reasoner-int4
visual_onnx=$package/onnx/visual-fp16
engine=$work/engines/reasoner-visual
timeline=$work/timeline.json
requests=$work/reasoner_requests.json
raw=$work/reasoner_raw.json
profile_log=$work/reasoner_profile.log
reasoner_result=$work/reasoner_10hz.json
trajectory_result=$work/trajectory_10hz.json
trajectory_mode=${COSMOS3EDGE_TRAJECTORY_MODE:-policy}

export PYTHONPATH=$package/third_party/diffusers-cosmos3/src:$package/runtime${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1

"$python_bin" "$package/runtime/prepare_cosmos3edge_video_10hz.py" \
    --video "$package/media/boston.mp4" --frame-dir "$work/frames" \
    --manifest "$timeline" --requests "$requests" --hz 10

"$package/deploy/build_reasoner_visual_on_jetson.sh" \
    "$edge" "$reasoner_onnx" "$visual_onnx" "$engine"

plugin=$edge/build/libNvInfer_edgellm_plugin.so
export EDGELLM_PLUGIN_PATH=$plugin
export LD_PRELOAD=$plugin${LD_PRELOAD:+:$LD_PRELOAD}
started_ns=$(date +%s%N)
"$edge/build/examples/llm/llm_inference" \
    --engineDir "$engine" --multimodalEngineDir "$engine" \
    --inputFile "$requests" --outputFile "$raw" --dumpOutput --dumpProfile \
    2>&1 | tee "$profile_log"
finished_ns=$(date +%s%N)
wall_s=$(awk -v start="$started_ns" -v finish="$finished_ns" 'BEGIN {printf "%.9f", (finish-start)/1000000000}')
"$python_bin" "$package/runtime/finalize_cosmos3edge_reasoner_10hz.py" \
    --manifest "$timeline" --raw-output "$raw" --output "$reasoner_result" \
    --profile-log "$profile_log" --wall-s "$wall_s"

export COSMOS3EDGE_ORT_TENSORRT=1
export COSMOS3EDGE_TRT_CACHE=$work/engines/ort-tensorrt
case "$trajectory_mode" in
    policy)
        "$python_bin" "$package/runtime/cosmos3edge_policy_onnx_10hz.py" \
            --checkpoint "$package/pipeline-config" --manifest "$timeline" \
            --vae-encoder "$package/onnx/vae-fp16-shared/onnx/encoder-av61-fp16.onnx" \
            --policy-transformer "$package/onnx/mot-fp16-shared/onnx/policy-av60-fp16.onnx" \
            --output "$trajectory_result" --steps 30 --seed 0
        ;;
    inverse)
        "$python_bin" "$package/runtime/cosmos3edge_inverse_onnx_10hz.py" \
            --checkpoint "$package/pipeline-config" --manifest "$timeline" \
            --vae-encoder "$package/onnx/vae-fp16-shared/onnx/encoder-av61-fp16.onnx" \
            --inverse-transformer "$package/onnx/mot-fp16-shared/onnx/inverse-av61-fp16.onnx" \
            --output "$trajectory_result" --steps 30 --seed 0
        ;;
    *) echo "Invalid COSMOS3EDGE_TRAJECTORY_MODE=$trajectory_mode (use policy or inverse)" >&2; exit 2 ;;
esac

"$python_bin" "$package/runtime/render_cosmos3edge_video_10hz.py" \
    --video "$package/media/boston.mp4" --manifest "$timeline" \
    --reasoner "$reasoner_result" --trajectory "$trajectory_result" \
    --output "$work/boston_cosmos3edge_onnx_10hz.mp4"

echo "Acceptance results: $work"
