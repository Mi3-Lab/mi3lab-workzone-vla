#!/usr/bin/env bash
# Run ON THE THOR.  Verifies and installs a pack made by pack_for_thor.sh into the
# exact paths the pipeline expects.  Replaces step 1 (01_sync_from_orin.sh) when the
# Thor cannot reach the Orin over the network.
#   bash thor/unpack_on_thor.sh /media/<drive>/thor_pack
set -euo pipefail
source "$(dirname "$0")/config.sh"
[[ $# -eq 1 && -d "$1" ]] || die "uso: $0 <pasta-do-pacote>"
PACK="$(cd "$1" && pwd)"
ensure_layout
c_grn "== conferindo integridade (sha256) =="
( cd "$PACK" && sha256sum --quiet -c SHA256SUMS ) || die "arquivo corrompido na cópia; copie o pacote de novo"
mkdir -p "$WZ_DATA"
rsync -ah --info=progress2 "$PACK/repo/"    "$REPO/"
rsync -ah --info=progress2 "$PACK/weights/" "$WZ_WEIGHTS/"
rsync -ah --info=progress2 "$PACK/data/"    "$WZ_DATA/"
c_grn "Instalado. Próximo: bash thor/00_preflight.sh"
