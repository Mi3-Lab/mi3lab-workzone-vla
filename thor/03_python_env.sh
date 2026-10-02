#!/usr/bin/env bash
# Creates ~/workzone/venv (the path the pipeline expects) with what the
# experiments need.  torch is the platform-specific part:
#   Jetson AGX Thor (JetPack 7, CUDA 13, SBSA): CUDA 13 aarch64 wheels from pytorch.org
#   DRIVE AGX Thor (DriveOS): PyTorch may be unavailable -- VLM experiments still run
#                             without it, detector experiments do not (see README)
#   Override with TORCH_INDEX=<url> if your platform needs a different wheel index.
set -euo pipefail
source "$(dirname "$0")/config.sh"
plat=$(detect_platform)
ensure_layout

if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$VENV"
fi
PIP="$VENV/bin/pip"; PY="$VENV/bin/python"
"$PIP" install -q --upgrade pip

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    case "$plat" in
        jetson-thor|drive-thor) idx="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}" ;;
        jetson-orin)            idx="${TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}" ;;
        *)                      idx="${TORCH_INDEX:-}" ;;
    esac
    c_ylw "instalando torch de: ${idx:-PyPI padrão}"
    if ! "$PIP" install torch torchvision ${idx:+--index-url "$idx"}; then
        c_ylw "não consegui instalar torch automaticamente."
    fi
fi
"$PIP" install -r "$THOR_DIR/requirements-thor.txt"

echo
if "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability(0))"; then
    c_grn "Python completo: todos os experimentos disponíveis."
else
    c_ylw "torch sem CUDA. Funciona: modelos de linguagem (2B, C3E), replay, vídeos."
    c_ylw "Não funciona: detector YOLO, baseline detector+CLIP, estimador conjunto ao vivo, TCN."
    c_ylw "Defina TORCH_INDEX com o índice de wheels da sua plataforma e rode de novo."
fi
c_grn "Próximo: bash thor/04_build_engines.sh"
