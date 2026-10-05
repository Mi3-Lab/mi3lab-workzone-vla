# IEEE IV 2027 submission — ROADWork only

Built from the authors' text in `../journal/main.tex`, cut to the ROADWork results
(human four-state labels). The California-focused draft is archived in
`../archive/iv2027_california/`.

- Deadline: 15 Nov 2026, 23:59 AWST (PaperCept, its.papercept.net). Double-blind.
- 6 pages incl. references (current: 6). IEEEtran `conference`. Abstract: 192 words (form limit 200).
- Build: `tectonic main.tex`. One paragraph per source line.

## New relative to the journal text
- Table "Across cities" (leave-one-city-out of the joint estimator): `pipeline/roadwork_analysis.py`
- Table "Response bias" (2B base / 2B fine-tuned / C3E on 100 labeled calibration videos): same script
- Main table reduced to 5 columns (dropped +FE and TCN; TCN quoted in text)

## Cut to fit 6 pages (all still in ../journal/main.tex)
California OOD benchmark and perceive-vs-act section; tables: inherited-vs-calibrated C3E,
paired CIs, joint ablation (numbers kept in text); qualitative and teaser figures;
debounce operating-point paragraph; detector+text fast entry; appendix.
