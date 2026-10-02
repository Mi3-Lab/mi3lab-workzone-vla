#!/usr/bin/env bash
# Checks the machine and reports what is ready and what is missing.  Changes nothing.
#   bash thor/00_preflight.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"

plat=$(detect_platform)
echo "== plataforma =="
echo "  modelo:      $( [[ -r /proc/device-tree/model ]] && tr -d '\0' < /proc/device-tree/model || echo '?')"
echo "  detectada:   $plat   (force com PLATFORM=... em thor/config.sh)"
[[ -r /etc/nv_tegra_release ]] && echo "  L4T:         $(head -1 /etc/nv_tegra_release)"
echo "  CUDA (nvcc): $(cuda_version || true)"
echo "  TensorRT:    $(dpkg -l 2>/dev/null | awk '/libnvinfer[0-9]* /{print $3; exit}')"
echo "  Python:      $(python3 --version 2>&1)"
echo "  memória:     $(free -g | awk '/Mem:/{print $2" GB"}')"
echo "  disco livre: $(df -h "$STORAGE" | awk 'NR==2{print $4" em "$6}')"
[[ "$plat" == unknown ]] && c_ylw "  plataforma desconhecida: defina PLATFORM em thor/config.sh"

ok=0; miss=0
chk() {  # chk <description> <test...>
    local d="$1"; shift
    if "$@" >/dev/null 2>&1; then c_grn "  [ok]    $d"; ok=$((ok+1)); else c_red "  [falta] $d"; miss=$((miss+1)); fi
}

echo; echo "== repositório e layout =="
chk "repo em ~/jetson-deploy (scripts assumem esse caminho)" test "$(readlink -f "$REPO")" = "$(readlink -f "$REPO_DEFAULT")"
chk "~/eval_cache -> repo/eval_cache" test -d "$EVAL_CACHE_LINK/ood_stream"
chk "patch do Edge-LLM no repo" test -f "$THOR_DIR/edgellm_patch/edgellm_v0.9.0_workzone.patch"

echo; echo "== modelos (fonte ONNX/PT, vindos do Orin) =="
chk "2B ONNX  (repo/onnx/llm, onnx/visual)" test -f "$REPO/onnx/llm/model.onnx" -a -f "$REPO/onnx/visual/model.onnx"
chk "C3E ONNX (reasoner-int4, visual-fp16)" test -f "$REPO/cosmos3edge-orin-deploy/onnx/reasoner-int4/model.onnx" -a -f "$REPO/cosmos3edge-orin-deploy/onnx/visual-fp16/model.onnx"
chk "YOLO .pt  (yolo12s_hardneg_1280.pt)" test -f "$WZ_WEIGHTS/yolo12s_hardneg_1280.pt"
chk "CLIP (cache open_clip, só para o baseline --clip)" test -d "$WZ_WEIGHTS/clip"

echo; echo "== Edge-LLM e engines (construídos NESTA máquina) =="
chk "Edge-LLM $EDGELLM_TAG em $EDGELLM_DIR" test -d "$EDGELLM_DIR/.git"
chk "llm_stream_video (binário do pipeline)" test -x "$EDGELLM_DIR/build/examples/llm/llm_stream_video"
chk "plugin libNvInfer_edgellm_plugin.so" test -f "$EDGELLM_DIR/build/libNvInfer_edgellm_plugin.so"
chk "engines 2B (repo/engines/llm, visual)" test -f "$REPO/engines/llm/llm.engine" -a -f "$REPO/engines/visual/visual.engine"
chk "engines C3E (~/cosmos3edge-engines)" test -f "$C3E_ENGINE_DIR/llm.engine" -a -f "$C3E_ENGINE_DIR/visual/visual.engine"
chk "engine YOLO (yolo12s_hardneg_1280_960.engine)" test -f "$WZ_WEIGHTS/yolo12s_hardneg_1280_960.engine"

echo; echo "== dados =="
chk "anotações 4 estados (Boston/Seattle)" test -f "$WZ_DATA/workzone_annotations_full.json"
chk "vídeos ROADWork (~/workzone/data/videos)" bash -c "ls '$WZ_DATA/videos'/*.mp4 | head -1"
chk "vídeos Califórnia (All_Construction_Data)" test -d "$WZ_DATA/All_Construction_Data/With_SpeedLimit"
chk "imagem de teste (repo/test/sample_frame.jpg)" test -f "$REPO/test/sample_frame.jpg"

echo; echo "== Python =="
chk "venv em ~/workzone/venv" test -x "$VENV/bin/python"
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY=python3
chk "numpy + opencv" "$PY" -c "import numpy, cv2"
chk "torch com CUDA" "$PY" -c "import torch; assert torch.cuda.is_available()"
chk "ultralytics 8.3.250" "$PY" -c "import ultralytics; assert ultralytics.__version__=='8.3.250'"
chk "open_clip (só baseline --clip)" "$PY" -c "import open_clip"

echo; echo "resumo: $ok ok, $miss faltando"
[[ $miss -eq 0 ]] && c_grn "Tudo pronto. Próximo: bash thor/05_smoke_test.sh" || c_ylw "Siga o guia thor/README.md na ordem dos números."
