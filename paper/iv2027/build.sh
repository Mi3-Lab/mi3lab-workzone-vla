#!/usr/bin/env bash
# Builds the paper exactly as IEEE does: pdflatex + bibtex, Times (URW Nimbus).
# TeX Live is installed user-local in ~/texlive (no sudo needed).
set -e
export PATH=$HOME/texlive/bin/aarch64-linux:$PATH
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex | grep -E "^!|Output written"
grep -E "Reference.*undefined|Citation.*undefined" main.log && exit 1 || true
pdfinfo main.pdf | grep -E "Pages|Page size"
