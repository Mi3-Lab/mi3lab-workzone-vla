#!/usr/bin/env bash
# Builds every TensorRT engine ON THIS DEVICE.  Engines are tied to the GPU
# they were built on: Orin engines do not run on Thor and vice versa.
#
#   bash thor/04_build_engines.sh            # all
#   bash thor/04_build_engines.sh 2b c3e     # only some: 2b | c3e | yolo
#   FORCE=1 bash thor/04_build_engines.sh    # rebuild even if the engine exists
set -euo pipefail
source "$(dirname "$0")/config.sh"
FORCE="${FORCE:-0}"
which=("$@"); [[ ${#which[@]} -eq 0 ]] && which=(2b c3e yolo)

B="$EDGELLM_DIR/build"
PLUGIN="$B/libNvInfer_edgellm_plugin.so"
LLM_BUILD="$B/examples/llm/llm_build"
VIS_BUILD="$B/examples/multimodal/visual_build"
need_edgellm() {
    for f in "$PLUGIN" "$LLM_BUILD" "$VIS_BUILD"; do [[ -e "$f" ]] || die "falta $f (rode 02_build_edgellm.sh)"; done
}
skip() { [[ "$FORCE" != 1 && -f "$1" ]] && { c_ylw "já existe: $1 (FORCE=1 para refazer)"; return 0; }; return 1; }

build_2b() {
    c_grn "== 2B fine-tunado (INT4 AWQ + visual FP16) =="
    need_edgellm
    # The tie_word_embeddings fix is baked into this ONNX at export time; if a
    # re-export ever loses it, replies start with junk such as ".DATA".
    # 05_smoke_test.py checks for exactly that symptom.
    if ! skip "$REPO/engines/llm/llm.engine"; then
        LD_PRELOAD="$PLUGIN" "$LLM_BUILD" --onnxDir "$REPO/onnx/llm" --engineDir "$REPO/engines/llm" \
            --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 4096
    fi
    if ! skip "$REPO/engines/visual/visual.engine"; then
        LD_PRELOAD="$PLUGIN" "$VIS_BUILD" --onnxDir "$REPO/onnx/visual" --engineDir "$REPO/engines" \
            --minImageTokens 128 --maxImageTokens 512 --maxImageTokensPerImage 512
    fi
}

build_c3e() {
    c_grn "== Cosmos3-Edge (reasoner INT4 AWQ + visual FP16, grade fixa 26x46 = 736x416) =="
    need_edgellm
    local src="$REPO/cosmos3edge-orin-deploy/onnx"
    if ! skip "$C3E_ENGINE_DIR/llm.engine"; then
        LD_PRELOAD="$PLUGIN" EDGELLM_PLUGIN_PATH="$PLUGIN" "$LLM_BUILD" \
            --onnxDir "$src/reasoner-int4" --engineDir "$C3E_ENGINE_DIR" \
            --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 4096
    fi
    if ! skip "$C3E_ENGINE_DIR/visual/visual.engine"; then
        LD_PRELOAD="$PLUGIN" EDGELLM_PLUGIN_PATH="$PLUGIN" "$VIS_BUILD" \
            --onnxDir "$src/visual-fp16" --engineDir "$C3E_ENGINE_DIR" \
            --minImageTokens 299 --maxImageTokens 299 --maxImageTokensPerImage 299
    fi
}

build_yolo() {
    c_grn "== detector YOLO12s (FP16, 960x960, como no Orin) =="
    local pt="$WZ_WEIGHTS/yolo12s_hardneg_1280.pt" out="$WZ_WEIGHTS/yolo12s_hardneg_1280_960.engine"
    [[ -f "$pt" ]] || die "falta $pt (rode 01_sync_from_orin.sh models)"
    skip "$out" && return 0
    "$VENV/bin/python" - "$pt" "$out" <<'PY'
import os, shutil, sys
from ultralytics import YOLO
pt, out = sys.argv[1], sys.argv[2]
# identical to the Orin engine's recorded export args (engine metadata)
path = YOLO(pt).export(format="engine", imgsz=960, half=True, simplify=True,
                       batch=1, dynamic=False, nms=False, device=0)
shutil.move(path, out)
print("engine:", out)
PY
}

for w in "${which[@]}"; do
    case "$w" in 2b) build_2b ;; c3e) build_c3e ;; yolo) build_yolo ;; *) die "desconhecido: $w" ;; esac
done
c_grn "engines prontos. Próximo: bash thor/05_smoke_test.sh"
