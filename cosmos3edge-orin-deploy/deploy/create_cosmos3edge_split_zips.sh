#!/usr/bin/env bash
# Create fast Zip64 STORE archives: operational ONNX runtime and references.

set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <package-root> [output-dir]" >&2
    exit 2
fi
package=$(realpath "$1")
parent=$(dirname "$package")
name=$(basename "$package")
output=${2:-$parent}
mkdir -p "$output"
output=$(realpath "$output")
operational=$output/COSMOS3EDGE_ORIN_OPERATIONAL_AV_FUTURE_SE3_FINAL.zip
reference=$output/COSMOS3EDGE_REFERENCE_WEIGHTS_BF16_INT4.zip
for archive in "$operational" "$reference"; do
    [[ ! -e "$archive" ]] || { echo "Refusing to overwrite: $archive" >&2; exit 1; }
done
for required in OPERATIONAL_MANIFEST_SHA256.txt REFERENCE_WEIGHTS_MANIFEST_SHA256.txt; do
    [[ -f "$package/$required" ]] || { echo "Missing manifest: $required" >&2; exit 1; }
done

cd "$parent"
find "$name" -type f \
    ! -path "$name/weights/*" \
    ! -name MANIFEST_SHA256.txt ! -name PACKAGE_MANIFEST.json \
    ! -name REFERENCE_WEIGHTS_MANIFEST_SHA256.txt \
    ! -name REFERENCE_WEIGHTS_PACKAGE_MANIFEST.json \
    -print | LC_ALL=C sort | zip -0 -q "$operational" -@

{
    find "$name/weights" -type f -print
    printf '%s\n' \
        "$name/REFERENCE_WEIGHTS_MANIFEST_SHA256.txt" \
        "$name/REFERENCE_WEIGHTS_PACKAGE_MANIFEST.json"
} | LC_ALL=C sort | zip -0 -q "$reference" -@

unzip -tq "$operational"
unzip -tq "$reference"
sha256sum "$operational" > "$operational.sha256"
sha256sum "$reference" > "$reference.sha256"
du -h "$operational" "$reference"
