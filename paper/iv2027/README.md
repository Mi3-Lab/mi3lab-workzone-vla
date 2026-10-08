# IEEE IV 2027 submission — ROADWork only

Built from the authors' text in `../journal/main.tex`, cut to the ROADWork results
(human four-state labels). The California-focused draft is archived in
`../archive/iv2027_california/`.

- Deadline: 15 Nov 2026, 23:59 AWST (PaperCept, its.papercept.net). Double-blind.
- 7 pages incl. references (one paid page; decided 2026-10-08). IEEEtran `conference`. Abstract: 185 words (form limit 200).
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

## Response to the two reviews of 2026-10-07 (Weak Reject x2)

| point | change | evidence |
|---|---|---|
| Fig. 1 "??", 39 approaches, 5%/95% | Teaser removed (cited a non-existent table and the withdrawn California result) | — |
| W1/C4 baseline without CLIP | Full detector+CLIP pipeline recorded on all 520 videos and recalibrated by replay (same grids): 0.470 -> 0.551, best system. Replay of the hand config reproduces the live run (0.473 vs 0.470) | `pipeline/dump_detclip.py`, `pipeline/detclip_recal.py`, `eval_cache/detclip_recal.log` |
| W2/C2 drives in both splits | 5-fold drive-disjoint replay of every replay-fitted system (new Table III, Sec. VII-A); all conclusions hold | `pipeline/drive_disjoint.py`, `eval_cache/dd_*` |
| C2 video-level bootstrap | All intervals now drive-level cluster bootstrap | `pipeline/review3_analysis.py --cluster` |
| C5 different operating points | Shared causal persistence filter, F1 at <=20 FA/h (Tables I, III) | `review3_analysis.py --equalfa` |
| W5/C8 anticipation | "Lead >=2 s" row; matched-nuisance anticipation in VII-A | `review3_analysis.py --lead` |
| W3/C6 HMM naming, VLM value | Filter called an HMM with ML counts, cited [Rabiner, Thrun]; channel ablation and efficiency stated | — |
| C3 debounce chosen on validation, mixed 2B column | 2B column now greedy without post-filter, all cells verified; debounce only as a sentence | `eval_cache/vlm_greedy` |
| V-C ">1.5 points" contradiction | Ablation Table II with all 7 toggles | ablation caches `eval_cache/vlm_*` |
| interior grid, unequal budgets | Grid extension stated; budgets stated (favour the baselines) | — |
| power/latency inconsistencies | Two sessions reported as increments over idle; 44 ms (recorded) vs 24 ms (later run) explained | `eval_cache/power_joint/` |
| AUC on binary output | Renamed balanced accuracy | — |
| Cosmos-Reason2 citation, Viola-Jones | Model card cited; cascades paragraph rewritten with FrugalGPT | `references.bib` |
| world model = reasoner only, Orin not automotive | Stated in III-A | — |
| title too universal | "The Role of Model-Specific Temporal Calibration" | — |
| C3E EGO-off behaviour | Described (INSIDE after 13 cycles on the transition prior), checked against `cascade_state_machine.py` | — |

Not done: fine-tuning hyper-parameters (A100 side); frame-level overlap check between fine-tuning images and benchmark videos; yes/no logits.

## Response to the third round (2026-10-08)

| point | change | evidence |
|---|---|---|
| "[?]" citations | Not in our build (0 undefined); the reviewed PDF was compiled with an old `references.bib`. Upload both files from `iv2027_overleaf.zip` | `./build.sh` |
| MC2 GPU concurrency | Live concurrent run (camera thread + detector + C3E process sharing the GPU), 20 val videos, twice: detector 24 -> 54 ms, C3E 1.44 s; detector-only filter live 0.425 vs replay 0.423; joint live 0.44 vs replay 0.50, paired -0.07 [-0.16, +0.01] | `pipeline/concurrent_joint.py`, `eval_cache/concurrent*` |
| bug found while doing MC2 | The "detector-only" counted filter kept the age of the world-model answer as an observation. Fixed (age masked): 0.524 at 22.5 FA/h (was 0.536 at 18.5); conclusions unchanged | `eval_cache/det_counted_noage`, `dd_det_counted_noage` |
| R2 unequal budgets | Detector+CLIP recalibrated with the VLM budget (one grid, same 100 videos): 0.522 val, +0.052 over hand-tuned, level with C3E | `eval_cache/detclip_cal100_val` |
| MC4 HMM sensitivity | Cell occupancy (68/108 cells seen, 20 cells = 97%), 3/6/10 detector buckets 0.538/0.546/0.545, counting rule stated | inline |
| MC5 where the VLM helps | 10 onsets preceded by a transcribed sign: joint and C3E warn >=2 s on 10/10, recalibrated det+CLIP 8/10 | inline |
| contributions too many | Reorganised into three (evaluation, findings, systems) | — |
| no recalibrated 2B in Table I | Added to caption (0.459) | — |
| double-blind | Award note removed from the prior-system bib entry | `references.bib` |

Not done (need GPU-hours or data not on this machine): counted filter on 2B evidence (~9 h), VLM search on all 312 videos (~3 h dump), fine-tuning/benchmark overlap check, time-based cascade windows, full F1-vs-FA curve figure (space).
