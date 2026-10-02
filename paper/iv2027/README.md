# New paper: response bias under domain shift (IEEE IV 2027 target)

Separate from `../journal/` (the system paper). Build: `tectonic main.tex`.

Venue: IEEE Intelligent Vehicles Symposium 2027, Perth.
- Deadline: **15 Nov 2026, 23:59 AWST = 15 Nov, 07:59 PST** (submission via PaperCept, its.papercept.net).
- Double-blind: no names, affiliations, lab names; the baseline [priorsystem2026] shares authors and must stay cited in the third person. Videos (optional) must also be anonymized: no faces, voices or lab identifiers.
- 6 pages including references; up to 8 with page charges. Current draft: 5.
- IEEE conference template (IEEEtran `conference`); remove the IEEE copyright notice for submission.
- The PaperCept form takes an abstract under 200 words; the paper's abstract is longer and needs a short version.
- Notification 15 Jan 2027; camera-ready 1 Feb 2027.

Source style: one paragraph per line.

## Status: BLOCKED on label verification

Every out-of-domain number comes from `annotation/california_draft.json`, a
first pass made from contact sheets. Verify at full frame rate:

    cd ~/jetson-deploy/annotation
    python3 label_tool.py --list
    python3 label_tool.py <video> --annotator "Name"

That writes `california_labels.json`; the scripts below pick it up
automatically. Then regenerate and update Tables 2-4 and the numbers in the
abstract and Sec. 5, and remove the red `\draftnote` blocks.

## Where every number comes from

| paper | script | notes |
|---|---|---|
| Table 1 (in domain) | `pipeline/compare_all_systems.py`, `ablate_joint.py`, `paired_ci_fast.py` | 208 validation videos, verified labels |
| window hits 95/72, clean windows 72/18, yes-rate 40.2/5.6 | `annotation/draft_eval.py` + inline window count | DRAFT labels |
| Table 2 (AUC) | `annotation/draft_eval.py` | DRAFT labels |
| Table 3 (four-state OOD) | `pipeline/california_4state.py` | DRAFT labels |
| Table 4 (adaptation) | `pipeline/california_adapt.py --lams 0.25,0.5,1.0` | DRAFT labels |
| Fig. 1 | `eval_cache/ood_california/far_yes_spotcheck.jpg` | 16 random frames, gate "yes", >20 s from a sign |
| evidence record | `pipeline/dump_ood_stream.py` -> `eval_cache/ood_stream/` | replay checked against 5 GPU runs |

## Open work before submission

- Label verification (above).
- A second world model or VLM, to show the bias is not specific to Cosmos3-Edge.
- A method: the adaptation section ends with two label-free signals
  (confident-negative stretches, cross-sensor agreement); neither is tested yet.

## New since the draft: Qwen-Drive-1.0-4B as a third sensor (2026-10-02, NOT yet in main.tex)

Qwen-Drive-1.0-4B (Qwen team, arXiv 2609.00111, Apache 2.0): Qwen3.5-4B VLM, unchanged, plus BEV and planning heads we do not use. Its training mix **includes ROADWork**, so Boston/Seattle/SF numbers would be contaminated; California is its clean test. Run with PyTorch BF16 on the Orin (not TensorRT), 736x416 = 299 image tokens, same prompts and budgets as C3E, greedy. Script: `pipeline/dump_qwendrive_stream.py`; table: `pipeline/compare_sensors.py`. DRAFT labels.

| | detector | C3E (zero-shot world model) | Qwen-Drive (driving VLM, saw ROADWork) |
|---|---|---|---|
| "yes" with no work in view | 5.6% | 40.2% | **1.9%** |
| "yes" on ego-relevant work | 35.6% | 73.9% | 37.5% |
| perception AUC, all | 0.72 | 0.72 | 0.75 |
| — day / sunset | 0.80 / 0.81 | 0.67 / 0.59 | 0.77 / 0.84 |
| — night / rain / fog | 0.52 / 0.62 / 0.60 | 0.70 / 0.77 / 0.76 | 0.57 / 0.73 / 0.72 |
| ego-relevance AUC | 0.43 | 0.47 | 0.41 |
| window hit at 39 signs | 56%* | 95% | 51% |
| window hit on work-free windows | 18%* | 72% | 5% |

\* detector at 1 Hz samples; the dense GPU run gave 72% at signs.

Paired video-level bootstrap on perception AUC (14 videos, 2000 reps): no pair differs significantly (Qwen-Drive − C3E +0.028 [−0.039, +0.100]; Qwen-Drive − detector +0.027 [−0.022, +0.077]).

Reading: three sensors with statistically indistinguishable discrimination and very different response bias. The window-hit proxy ranks C3E far above Qwen-Drive (95% vs 51%); bias-free AUC ties them. A driving-specialized model that saw ROADWork still fails ego-relevance. This strengthens every claim of the paper; SIGN/DESC for Qwen-Drive (needed for the four-state replay) are being recorded to `eval_cache/qwendrive_ood_stream`.
