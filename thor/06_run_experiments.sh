#!/usr/bin/env bash
# Re-runs the paper's experiments on this device.  Everything is latency-honest:
# a faster GPU observes more frames, so the numbers CAN change -- that change is
# the experiment (do temporal constants calibrated on Orin survive new hardware?).
#
# Results go to eval_cache/platforms/<device>/ and never overwrite the Orin caches.
# Only ONE model process may run at a time (they share /dev/shm/cascade_request.json),
# so the sets run one after another.  Long: start it with nohup (see README).
#
#   bash thor/06_run_experiments.sh indomain     # 5 systems x 208 videos  (~8-10 h on Orin)
#   bash thor/06_run_experiments.sh california   # C3E + detector evidence on 14 CA videos (~1 h)
#   bash thor/06_run_experiments.sh sf           # C3E evidence on the 8 SF clips (~10 min)
#   bash thor/06_run_experiments.sh compare      # tables: Orin vs this device
set -uo pipefail
source "$(dirname "$0")/config.sh"
[[ $# -ge 1 ]] || die "uso: $0 indomain|california|sf|compare [...]"

DEV=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null | tr ' /' '__' | cut -c1-40)
OUT="$REPO/eval_cache/platforms/${DEV:-unknown}"
mkdir -p "$OUT"
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY=python3
cd "$REPO/pipeline"
run() {  # run <name> <python> <args...>   -> log in $OUT/<name>.log, skip if done
    local name=$1 py=$2; shift 2
    if [[ -f "$OUT/$name.done" ]]; then c_ylw "já feito: $name"; return; fi
    c_grn "[$(date +%H:%M)] $name"
    if "$py" -u "$@" > "$OUT/$name.log" 2>&1; then touch "$OUT/$name.done"; else c_red "falhou: $name (veja $OUT/$name.log)"; fi
}

indomain() {
    local V="--split-file eval_split_full.json --split validation --limit 10000"
    run 2b_sampled python3 evaluate_cascade.py $V --cache-dir "$OUT/vlm"
    run 2b_greedy  python3 evaluate_cascade.py $V --temperature 0 --cache-dir "$OUT/vlm_greedy"
    run c3e_cal    python3 evaluate_cascade.py $V --temperature 0 \
        --engine-dir "$C3E_ENGINE_DIR" --infer-size 736x416 \
        --csm N=3,k_enter=3,k_exit=2,k_fading=1,use_ego=0 --cache-dir "$OUT/cosmos3edge_cal"
    run detector_clip "$PY" evaluate_yolo_baseline.py --split-file eval_split_full.json \
        --split validation --clip --cache-dir "$OUT/yoloclip"
    run joint_312 "$PY" joint_filter.py --split validation \
        --params "$REPO/eval_cache/joint_params_full.json" --cache-dir "$OUT/joint_full_val"
}
california() { run ood_stream "$PY" dump_ood_stream.py --out "$OUT/ood_stream"; }
sf()         { run sf_stream  python3 dump_sf_stream.py --out "$OUT/sf_stream"; }
compare()    { "$PY" "$THOR_DIR/compare_platforms.py" "$OUT"; }

for what in "$@"; do
    case "$what" in indomain|california|sf|compare) "$what" ;; *) die "desconhecido: $what" ;; esac
done
