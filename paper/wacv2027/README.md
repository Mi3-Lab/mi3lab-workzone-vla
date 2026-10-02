# WACV 2027 submission — pure-VLM work-zone traversal state estimation

Target: **WACV 2027, Round 2** (paper registration Aug 21, submission
**Aug 28, 2026**, supplemental Aug 30; decisions Oct 9; camera-ready Nov 2).
Track: **Applications** (evaluated on systems-level innovation, novelty of
domain, comparative assessment). Format: 8 pages + references, anonymized,
OpenReview.

## Files
- `main.tex` — full paper (anonymized, `\usepackage[review]{wacv}`)
- `references.bib` — bibliography
- `fig_cascade.tex` — standalone TikZ source for Figure 1; compile with
  `pdflatex fig_cascade.tex` to produce `fig_cascade.pdf` before building
  `main.tex`

## Building
No LaTeX on the Jetson. Use Overleaf:
1. Download the official **WACV 2027 Author Kit** (linked from the WACV
   author guidelines) and create a project from it.
2. Replace the kit's `main.tex` with ours; add `references.bib` and
   `fig_cascade.tex` (compile the figure once, or use the `standalone`
   package).
3. The kit provides `wacv.sty` and `ieee_fullname.bst` which `main.tex`
   expects.

## Anonymization notes (IMPORTANT before submission)
- The baseline system paper (arXiv:2606.08860) **shares authors with this
  submission**. It is cited in third person as `priorsystem2026` with an
  anonymized author field in the .bib. Before submitting, double-check
  WACV's dual-submission/self-citation policy wording and keep all
  references to it in third person ("the system of [X]", "its authors").
- Do not mention lab name, GitHub org, or grant numbers in the submission.
- The benchmark (520 annotated videos) is described as "the same benchmark
  used by [the baseline]" — on acceptance, de-anonymize and name it.

## Provenance of every number in the paper (traceability)
All metrics are from real runs on the Jetson AGX Orin (this repo):
- VLM validation (208 videos, 192,135 frames): `~/eval_full_vlm_validation.log`
  ("METRICAS NO FORMATO DO PAPER" section)
- VLM calibration (312 videos): `~/eval_full_vlm_calibration.log`
- Per-video frame arrays cached: `~/eval_cache/vlm/*.npz` (re-derive any
  metric without re-running inference via `pipeline/paper_metrics.py`)
- Latency measurements: session logs (gate ~90ms / sign ~200ms / ego 160/82ms
  / desc ~540ms @480px; decode 10.4ms/tok; vision 44.6ms@299tok, ~20ms@135tok)
- Baseline numbers: as published in arXiv:2606.08860 Sec. V (same dataset,
  same metric definitions — implemented in `pipeline/paper_metrics.py`)
- Qualifier-filter ablation (39.8%→89.1%): calibration probe video
  `boston_3d896c9c53dd4852b3f4d9bff20602df_000000_14940`
- Prompt-bias probes (binary sign gate misses / distance prompts fail /
  transcription faithful): session probe logs, frames from `boston_2d8e13...`
  and `boston_3d896...`

## TODO before submission
- [x] Local WACV-style compile checked: main text ends on page 8 and
      references begin on page 9 (`main.pdf`, 2026-07-27). Reconfirm with
      the official author kit before submission.
- [x] Supplemental written: `supp.tex`/`supp.pdf` (prompts, parsing rules,
      corroboration lexicon, Bayes matrices, deploy config, protocol)
- [ ] Re-verify WACV 2027 R2 exact dates on the CFP page closer to deadline
- [ ] Internal pass for any de-anonymizing detail

## Leakage claim — RESOLVED (2026-07-19)
The paper claims only what is verifiable from this machine: (a) the 4-state
interval annotations are evaluation-only (never fine-tuning labels), (b)
hard negatives come from BDD100K, and (c) — now stated explicitly in the
Evaluation Protocol section — both the SFT scene imagery and the benchmark
videos derive from the ROADWork corpus; possible zone-level overlap is
disclosed rather than denied, and the comparison stays balanced because the
baseline's detector is likewise ROADWork-trained. A zero-frame-overlap
claim would require the A100-side training manifests (out of scope for this
work cycle); the disclosure wording does not depend on them.
