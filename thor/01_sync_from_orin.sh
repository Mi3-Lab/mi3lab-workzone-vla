#!/usr/bin/env bash
# Copies models and data from the Orin to this machine with rsync (resumable).
# Run on the Thor.  Set ORIN_HOST in thor/config.sh first.
#
#   bash thor/01_sync_from_orin.sh models       # ONNX + YOLO .pt + CLIP          ~6 GB   (needed always)
#   bash thor/01_sync_from_orin.sh smoke        # 1 California + 1 Boston video   ~0.3 GB (smoke test)
#   bash thor/01_sync_from_orin.sh validation   # 208 benchmark videos + labels   ~4.2 GB (in-domain table)
#   bash thor/01_sync_from_orin.sh calibration  # 312 benchmark videos            ~6.3 GB (only to refit on Thor)
#   bash thor/01_sync_from_orin.sh california   # 15 annotated CA videos          ~5 GB   (out-of-domain)
#   bash thor/01_sync_from_orin.sh sf           # 8 San Francisco ROADWork clips  ~0.4 GB
#   bash thor/01_sync_from_orin.sh all
set -euo pipefail
source "$(dirname "$0")/config.sh"
[[ "$ORIN_HOST" == *ORIN_IP_HERE* ]] && die "defina ORIN_HOST em thor/config.sh (ex.: mi3-jetson@192.168.1.50)"
[[ $# -ge 1 ]] || die "uso: $0 models|smoke|validation|calibration|california|sf|all"

RS=(rsync -ah --info=progress2 --partial)
ensure_layout

# heavy data may live elsewhere; ~/workzone/data stays the path the scripts use
mkdir -p "$HOME/workzone" "$WZ_WEIGHTS"
if [[ "$(readlink -f "$STORAGE")" != "$(readlink -f "$HOME")" && ! -e "$WZ_DATA" ]]; then
    mkdir -p "$STORAGE/workzone/data"; ln -s "$STORAGE/workzone/data" "$WZ_DATA"
fi
mkdir -p "$WZ_DATA/videos" "$WZ_DATA/All_Construction_Data"

split_list() {  # split_list <validation|calibration>  -> file with one video name per line
    python3 - "$REPO/pipeline/eval_split_full.json" "$1" <<'PY'
import json, sys
print("\n".join(json.load(open(sys.argv[1]))[sys.argv[2]]))
PY
}

do_models() {
    c_grn "== modelos =="
    mkdir -p "$REPO/onnx" "$REPO/cosmos3edge-orin-deploy/onnx"
    "${RS[@]}" "$ORIN_HOST:jetson-deploy/onnx/" "$REPO/onnx/"
    for d in reasoner-int4 visual-fp16; do
        "${RS[@]}" "$ORIN_HOST:jetson-deploy/cosmos3edge-orin-deploy/onnx/$d/" "$REPO/cosmos3edge-orin-deploy/onnx/$d/"
    done
    "${RS[@]}" "$ORIN_HOST:workzone/weights/yolo12s_hardneg_1280.pt" "$WZ_WEIGHTS/"
    "${RS[@]}" "$ORIN_HOST:workzone/weights/clip/" "$WZ_WEIGHTS/clip/"
}
do_smoke() {
    c_grn "== smoke =="
    mkdir -p "$WZ_DATA/All_Construction_Data/With_SpeedLimit"
    "${RS[@]}" "$ORIN_HOST:workzone/data/All_Construction_Data/With_SpeedLimit/CA99_Day_01.MP4" \
               "$ORIN_HOST:workzone/data/All_Construction_Data/With_SpeedLimit/Speed_Sign_Timestamps.txt" \
               "$WZ_DATA/All_Construction_Data/With_SpeedLimit/"
    "${RS[@]}" "$ORIN_HOST:workzone/data/workzone_annotations_full.json" "$WZ_DATA/"
    local v; v=$(split_list validation | head -1)
    "${RS[@]}" "$ORIN_HOST:workzone/data/videos/$v" "$WZ_DATA/videos/"
}
do_split() {
    c_grn "== $1 =="
    "${RS[@]}" "$ORIN_HOST:workzone/data/workzone_annotations_full.json" "$WZ_DATA/"
    local lst; lst=$(mktemp); split_list "$1" > "$lst"
    "${RS[@]}" --files-from="$lst" "$ORIN_HOST:workzone/data/videos/" "$WZ_DATA/videos/"
    rm -f "$lst"
}
do_california() {
    c_grn "== california =="
    # only With_SpeedLimit carries labels (sign timestamps and the four-state draft)
    "${RS[@]}" "$ORIN_HOST:workzone/data/All_Construction_Data/With_SpeedLimit/" \
               "$WZ_DATA/All_Construction_Data/With_SpeedLimit/"
}
do_sf() {
    c_grn "== sf =="
    local lst; lst=$(mktemp)
    ls "$REPO/eval_cache/sf_stream/"*.npz | xargs -n1 basename | sed 's/\.npz$/_snippet.mp4/' > "$lst"
    "${RS[@]}" --files-from="$lst" "$ORIN_HOST:workzone/data/videos/" "$WZ_DATA/videos/"
    rm -f "$lst"
}

for what in "$@"; do
    case "$what" in
        models) do_models ;; smoke) do_smoke ;; california) do_california ;; sf) do_sf ;;
        validation|calibration) do_split "$what" ;;
        all) do_models; do_smoke; do_split validation; do_split calibration; do_california; do_sf ;;
        *) die "pacote desconhecido: $what" ;;
    esac
done
c_grn "sincronização concluída. Rode: bash thor/00_preflight.sh"
