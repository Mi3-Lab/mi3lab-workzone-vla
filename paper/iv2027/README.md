# IEEE IV 2027 submission — ROADWork only

Built from the authors' text in `../journal/main.tex`, cut to the ROADWork results
(human four-state labels). The California-focused draft is archived in
`../archive/iv2027_california/`.

- Deadline: 15 Nov 2026, 23:59 AWST (PaperCept, its.papercept.net). Double-blind.
- 6 pages incl. references (current: 6). IEEEtran `conference`. Abstract: 192 words (form limit 200).
- Build: `./build.sh` (pdflatex + bibtex + Times, as IEEE builds it; TeX Live user-local in `~/texlive`). Do not use tectonic for page counts: its Times clone paginates differently. One paragraph per source line.

## New relative to the journal text
- Table "Across cities" (leave-one-city-out of the joint estimator): `pipeline/roadwork_analysis.py`
- Table "Response bias" (2B base / 2B fine-tuned / C3E on 100 labeled calibration videos): same script
- Main table reduced to 5 columns (dropped +FE and TCN; TCN quoted in text)

## Cut to fit 6 pages (all still in ../journal/main.tex)
California OOD benchmark and perceive-vs-act section; tables: inherited-vs-calibrated C3E,
paired CIs, joint ablation (numbers kept in text); qualitative and teaser figures;
debounce operating-point paragraph; detector+text fast entry; appendix.

## Response to the review of 2026-10-07 (point by point)

| point | change | evidence |
|---|---|---|
| W1 framing | Title, abstract, intro and conclusion reframed around recalibrating every system | — |
| W2 baseline not recalibrated | Detector state machine recalibrated on calibration (3-stage search, 4,800 configs each, optimum interior or at score ceiling): 0.435 → 0.530 val (0.530 cal). New "Det. cal." column in Table I; new paragraph "The baseline too" | `pipeline/calibrate_detector_sm.py`, `eval_cache/detector_sm_search*.log` |
| W3 hand-set τ/buckets | Stated as set on calibration; CV sensitivity reported | inline CV in session; `pipeline/cv_joint_calibration.py` |
| W4 validation used for selection | **Valid.** The 33.2 FA/h sentence was removed; protocol is "fit on all calibration" with no selection; 5-fold CV on calibration reported | `pipeline/cv_joint_calibration.py` |
| W5 drive-disjoint split | Not done; acknowledged in limitations | — |
| W6 missing CIs | Paired CIs added for joint vs recalibrated detector, vs C3E; "within alarm budget" claim removed | `pipeline/review_w2_caches.py` |
| W7 36 pure-negative videos | Now validation-only (13 videos, 67.4%) | — |
| W8 joint-system cost | 39.0 W median (46.9 peak) vs 33.5 W detector alone, 11.1 W idle; detector cycle unchanged | `eval_cache/power_joint/*.log` |
| W9 gate calibration tension | Sec. III-A qualified, pointing to Table III | — |
| W10 reproducibility | Base checkpoint named; TCN specified; release "upon acceptance". **Fine-tuning hyper-parameters still missing (on the A100 side)** | — |
| minor | Abstract antecedent; FA/h and timing defined; unreported transition matching removed; greedy 2B is the table column; 82x → absolute first; distance per cycle at highway speed | — |

Extra check (not asked): the C3E threshold search optimum (N=3) was on the grid edge; N=1,2,4 added, optimum unchanged and now interior.
