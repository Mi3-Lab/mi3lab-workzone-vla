#!/usr/bin/env bash
# Verify every delivered byte against the manifest for the selected archive.

set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <package-root> [full|operational|reference]" >&2
    exit 2
fi
root=$(realpath "$1")
mode=${2:-full}
case "$mode" in
    full) manifest=MANIFEST_SHA256.txt ;;
    operational) manifest=OPERATIONAL_MANIFEST_SHA256.txt ;;
    reference) manifest=REFERENCE_WEIGHTS_MANIFEST_SHA256.txt ;;
    *) echo "Invalid mode: $mode" >&2; exit 2 ;;
esac
[[ -f "$root/$manifest" ]] || { echo "Missing manifest: $root/$manifest" >&2; exit 1; }
(cd "$root" && sha256sum --check "$manifest")

if [[ "$mode" != reference ]]; then
    python_bin=${PYTHON_BIN:-python3}
    "$python_bin" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is required")
if "TensorrtExecutionProvider" not in providers:
    raise SystemExit("TensorrtExecutionProvider is required for the Orin acceptance run")
PY
fi
echo "Package verification PASS: $mode"
