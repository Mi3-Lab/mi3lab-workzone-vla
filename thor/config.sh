#!/usr/bin/env bash
# Shared settings for every script in thor/.  Edit the first block, nothing else.
# Sourced, not executed:  source thor/config.sh

# ---------------------------------------------------------------- edit me
# auto | jetson-thor | drive-thor | jetson-orin
PLATFORM="${PLATFORM:-auto}"
# The Orin that holds the models and data, for 01_sync_from_orin.sh (user@host).
ORIN_HOST="${ORIN_HOST:-mi3-jetson@ORIN_IP_HERE}"
# Where the big data should live on this machine.  If it is not $HOME,
# ~/workzone/data is created as a symlink into it.
STORAGE="${STORAGE:-$HOME}"
# ------------------------------------------------------------------------

# Every pipeline script uses these exact paths (os.path.expanduser("~/...")),
# so the kit reproduces the Orin layout under $HOME instead of rewriting them.
REPO_DEFAULT="$HOME/jetson-deploy"
EDGELLM_DIR="$HOME/TensorRT-Edge-LLM"
EDGELLM_URL="https://github.com/NVIDIA/TensorRT-Edge-LLM"
EDGELLM_TAG="v0.9.0"
EDGELLM_COMMIT="1ac0f2b99642045125e1c5ac7b109434ba3b36c7"
C3E_ENGINE_DIR="$HOME/cosmos3edge-engines"
WZ_DATA="$HOME/workzone/data"
WZ_WEIGHTS="$HOME/workzone/weights"
VENV="$HOME/workzone/venv"
EVAL_CACHE_LINK="$HOME/eval_cache"

THOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$THOR_DIR/.." && pwd)"

c_red() { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn() { printf '\033[32m%s\033[0m\n' "$*"; }
c_ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
die()   { c_red "ERRO: $*"; exit 1; }

detect_platform() {
    if [[ "$PLATFORM" != "auto" ]]; then echo "$PLATFORM"; return; fi
    local model=""
    [[ -r /proc/device-tree/model ]] && model=$(tr -d '\0' < /proc/device-tree/model)
    if [[ -r /etc/nv_tegra_release ]]; then
        if grep -qi thor <<<"$model"; then echo jetson-thor; return; fi
        if grep -qi orin <<<"$model"; then echo jetson-orin; return; fi
    fi
    # A Thor board with no JetPack release file is a DRIVE AGX Thor on DriveOS.
    if grep -qi thor <<<"$model"; then echo drive-thor; return; fi
    echo unknown
}

cuda_version() {
    local nvcc
    nvcc=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
    [[ -x "$nvcc" ]] || { echo ""; return; }
    "$nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'
}

edgellm_target() {
    case "$1" in
        jetson-thor) echo jetson-thor ;;
        drive-thor)  echo auto-thor ;;
        jetson-orin) echo jetson-orin ;;
        *)           echo "" ;;
    esac
}

# Recreates the Orin layout the pipeline scripts expect, without moving anything:
# ~/jetson-deploy -> this clone, ~/eval_cache -> <clone>/eval_cache.
ensure_layout() {
    if [[ "$(readlink -f "$REPO")" != "$(readlink -f "$REPO_DEFAULT")" ]]; then
        [[ -e "$REPO_DEFAULT" ]] && die "$REPO_DEFAULT já existe e não é este clone ($REPO)"
        ln -s "$REPO" "$REPO_DEFAULT"; c_grn "criado $REPO_DEFAULT -> $REPO"
    fi
    if [[ ! -e "$EVAL_CACHE_LINK" ]]; then
        ln -s "$REPO/eval_cache" "$EVAL_CACHE_LINK"; c_grn "criado $EVAL_CACHE_LINK -> $REPO/eval_cache"
    fi
    mkdir -p "$HOME/workzone" "$WZ_WEIGHTS"
}
