#!/usr/bin/env bash
# Run ON THE ORIN.  Packs everything the Thor needs that is NOT in git and NOT
# downloadable (our INT4 C3E ONNX, the fine-tuned 2B ONNX, our YOLO weights),
# plus optional data, into one folder you can copy to a USB/NVMe drive.
#
#   bash thor/pack_for_thor.sh /media/<drive>/thor_pack            # models + smoke data  (~6.5 GB)
#   bash thor/pack_for_thor.sh /media/<drive>/thor_pack validation  # + 208 benchmark videos (~4.2 GB)
#   bash thor/pack_for_thor.sh /media/<drive>/thor_pack all         # + calibration videos too (~6.3 GB)
#
# On the Thor:  bash thor/unpack_on_thor.sh /media/<drive>/thor_pack
set -euo pipefail
source "$(dirname "$0")/config.sh"
[[ $# -ge 1 ]] || die "uso: $0 <pasta-destino> [validation|all]"
OUT="$1"; EXTRA="${2:-}"
mkdir -p "$OUT"
cp_rel() {  # cp_rel <src-root> <relative path>   -> $OUT/<tag>/<relative path>
    local root="$1" rel="$2" tag="$3"
    mkdir -p "$OUT/$tag/$(dirname "$rel")"
    rsync -ah --info=progress2 "$root/$rel" "$OUT/$tag/$(dirname "$rel")/"
}
c_grn "== modelos =="
cp_rel "$REPO" onnx/llm repo
cp_rel "$REPO" onnx/visual repo
cp_rel "$REPO" cosmos3edge-orin-deploy/onnx/reasoner-int4 repo
cp_rel "$REPO" cosmos3edge-orin-deploy/onnx/visual-fp16 repo
cp_rel "$HOME/workzone/weights" yolo12s_hardneg_1280.pt weights
cp_rel "$HOME/workzone/weights" clip weights

c_grn "== dados do teste de fumaça =="
cp_rel "$WZ_DATA" workzone_annotations_full.json data
cp_rel "$WZ_DATA" All_Construction_Data/With_SpeedLimit/CA99_Day_01.MP4 data
cp_rel "$WZ_DATA" All_Construction_Data/With_SpeedLimit/Speed_Sign_Timestamps.txt data

pick() {  # pick <split> -> copy that split's videos
    python3 -c "import json;print('\n'.join(json.load(open('$REPO/pipeline/eval_split_full.json'))['$1']))" |
    while read -r v; do cp_rel "$WZ_DATA" "videos/$v" data; done
}
[[ "$EXTRA" == validation || "$EXTRA" == all ]] && { c_grn "== vídeos de validação =="; pick validation; }
[[ "$EXTRA" == all ]] && { c_grn "== vídeos de calibração =="; pick calibration; }

( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS )
du -sh "$OUT"
c_grn "Pacote pronto em $OUT. No Thor: bash thor/unpack_on_thor.sh $OUT"
