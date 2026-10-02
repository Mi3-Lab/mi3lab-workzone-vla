#!/usr/bin/env bash
# Clones TensorRT Edge-LLM v0.9.0, applies this project's patch (Cosmos3-Edge
# support, the persistent llm_stream_video binary every pipeline script uses,
# and a CuTe DSL link fix), and builds it.
#
#   Jetson AGX Thor / Orin:  bash thor/02_build_edgellm.sh            (on the board)
#   DRIVE AGX Thor:          bash thor/02_build_edgellm.sh --in-docker (inside the
#                            DriveOS SDK Docker image on the x86 host), then copy
#                            the build/ folder to the board -- see thor/README.md
set -euo pipefail
source "$(dirname "$0")/config.sh"

IN_DOCKER=0; [[ "${1:-}" == "--in-docker" ]] && IN_DOCKER=1
plat=$(detect_platform)
[[ $IN_DOCKER -eq 1 ]] && plat=drive-thor
target=$(edgellm_target "$plat")
[[ -n "$target" ]] || die "plataforma '$plat' desconhecida; defina PLATFORM em thor/config.sh"

if [[ "$plat" == drive-thor && $IN_DOCKER -eq 0 ]]; then
    c_ylw "DRIVE Thor: o Edge-LLM é compilado no Docker do DriveOS SDK (x86), não na placa."
    c_ylw "Siga a seção 'DRIVE AGX Thor' de thor/README.md. Depois de copiar build/ para"
    c_ylw "$EDGELLM_DIR/build nesta placa, rode: bash thor/04_build_engines.sh"
    exit 0
fi

ctk=$(cuda_version)
[[ -n "$ctk" ]] || die "nvcc não encontrado (instale o CUDA do JetPack/DriveOS SDK)"
c_grn "plataforma=$plat  EMBEDDED_TARGET=$target  CUDA=$ctk"

# ---- source at the validated commit
if [[ ! -d "$EDGELLM_DIR/.git" ]]; then
    git clone "$EDGELLM_URL" "$EDGELLM_DIR"
fi
cd "$EDGELLM_DIR"
if [[ -f examples/llm/llm_stream_video.cpp ]]; then
    c_ylw "patch já aplicado (llm_stream_video.cpp existe); pulando checkout/patch"
else
    [[ -z "$(git status --porcelain)" ]] || die "$EDGELLM_DIR tem mudanças locais; use uma cópia limpa"
    git fetch --tags -q origin
    git checkout -q "$EDGELLM_COMMIT"
    git submodule update --init --depth 1
    git apply "$THOR_DIR/edgellm_patch/edgellm_v0.9.0_workzone.patch"
    cp "$THOR_DIR/edgellm_patch/llm_stream_video.cpp" examples/llm/
    c_grn "patch aplicado sobre $EDGELLM_TAG ($EDGELLM_COMMIT)"
fi

# ---- configure + build
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET="$target" \
    -DCUDA_CTK_VERSION="$ctk" \
    -DENABLE_CUTE_DSL=ALL
make -j"$(nproc)" llm_build llm_inference llm_stream_video visual_build NvInfer_edgellm_plugin

for f in examples/llm/llm_build examples/llm/llm_stream_video examples/multimodal/visual_build libNvInfer_edgellm_plugin.so; do
    [[ -e "$f" ]] || die "faltou $f no build"
done
c_grn "Edge-LLM pronto em $EDGELLM_DIR/build"
[[ $IN_DOCKER -eq 1 ]] && c_ylw "Agora copie $EDGELLM_DIR/build para a placa DRIVE (mesmo caminho)." || c_grn "Próximo: bash thor/03_python_env.sh"
