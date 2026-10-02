#!/usr/bin/env bash
# Overlay the tested Cosmos3-Edge changes on TensorRT Edge-LLM v0.9.0.

set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <package-root> <TensorRT-Edge-LLM-dir>" >&2
    exit 2
fi

package=$(realpath "$1")
edge=$(realpath "$2")
patch_root=$package/third_party/tensorrt_edge_llm_patch
[[ -d "$edge/cpp" ]] || { echo "Invalid TensorRT Edge-LLM checkout: $edge" >&2; exit 1; }
[[ -d "$patch_root/cpp" ]] || { echo "Missing package patch: $patch_root" >&2; exit 1; }

if [[ -d "$edge/.git" ]]; then
    version=$(git -C "$edge" describe --tags --always 2>/dev/null || true)
    case "$version" in
        v0.9.0*) ;;
        *) echo "Patch was validated on TensorRT Edge-LLM v0.9.0; found: ${version:-unknown}" >&2; exit 1 ;;
    esac
fi

# Backups use the .pre-cosmos3edge suffix; apply once to a clean v0.9.0 checkout.
rsync -a --itemize-changes --backup --suffix=.pre-cosmos3edge \
    "$patch_root/cpp/" "$edge/cpp/"
echo "Cosmos3-Edge patch applied to: $edge"
echo "Now rebuild TensorRT Edge-LLM on the Jetson, then run build_reasoner_visual_on_jetson.sh."
