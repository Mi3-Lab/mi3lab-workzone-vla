# Journal version

`main.tex` is the WACV 2027 draft (`../wacv2027/main.tex`) with targeted changes:
journal class, not anonymized, the joint estimator and perception-vs-action results
added, stale claims fixed, and a handful of sentence-level readability edits.
Source style: one paragraph per line (`python3 ../wacv2027/reflow.py main.tex`).
`main_rewrite.tex` is an earlier full rewrite, kept only for reference.

## Build
    tectonic main.tex
Tectonic is in `~/.local/bin`.  Under XeTeX the preamble swaps IEEEtran's
Times for TeX Gyre Termes (same design, has bold and small caps).  A pdflatex
build (Overleaf, the journal's system) skips that and uses Times.

Figures are symlinks to `../wacv2027/`.  Replace them with copies before
uploading anywhere.

## Before submission
- [ ] Choose the journal.  The class is IEEEtran (T-ITS / Access style);
      another publisher's class only touches the preamble and author block.
- [ ] Author list, affiliations, funding, code URL (`TODO` in the title block).
- The baseline [priorsystem2026] shares authors with this paper.  It is cited
  neutrally, like any other reference; do not call it "our earlier work".

## Where every number comes from
`../wacv2027/JOURNAL_RESULTS.md` lists the script and cache behind each table.
All numbers are re-derivable from `~/eval_cache/` without re-running inference,
except the out-of-domain runs, whose logs are in `~/eval_cache/ood_california/`.
