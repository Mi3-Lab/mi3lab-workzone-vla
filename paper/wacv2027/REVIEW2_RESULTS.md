# Second-review experimental results (Jetson-only, no A100, no retrain)

## STATUS: all runnable experiments done + paper updated (2026-07-19). Body fits 8pp.
## Addressed in paper: W1(partial/per-city), W2, W4, W5, W7, W8, W9(text), W10(supp).
## Scoped as future work (need annotation/new baselines): W3, W6, full W9 decomposition.


All from cached per-frame predictions unless noted. Scripts:
`pipeline/review2_analysis.py`, `pipeline/bench_latency_power.py`.

## W1/Q12 — per-city (generalization)
| system | city | frame acc | F1 | appr IoU | ins IoU | ins P | ins R |
|---|---|---|---|---|---|---|---|
| VLM raw | Boston (154) | 61.2% | 0.444 | 0.124 | 0.515 | 51.6% | 95.7% |
| VLM raw | Seattle (54) | 66.3% | 0.520 | 0.224 | 0.587 | 66.2% | 100% |
| Det+CLIP | Boston | 63.3% | 0.469 | 0.113 | 0.530 | 82.0% | 95.7% |
| Det+CLIP | Seattle | 61.7% | 0.462 | 0.152 | 0.576 | 74.2% | 95.5% |
VLM works in BOTH cities (no collapse); weaker precision on Boston. Not concentrated in one memorized city.

## W2/Q2 — false alarms / hour
| config | FA/h | mean dur | median |
|---|---|---|---|
| VLM raw (sampled) | 60.1 | 2.9s | 1.6s |
| VLM greedy, no debounce | 56.2 | 3.2s | 1.8s |
| **VLM greedy + 1s debounce (REC)** | **37.7** | 4.5s | 3.1s |
| VLM greedy + 2s debounce | 27.5 | 5.6s | 3.8s |
| Detector+CLIP (paired) | 30.9 | 5.8s | 4.8s |
HONEST: recommended config (37.7/h) still ABOVE baseline (30.9/h). 2s debounce (27.5) finally beats it but costs recall.

## W4/Q3 — SIGN channel contribution (UNDERCUTS current strong claim)
- entries present in full but absent w/o SIGN: 4/177 videos (2.3%)
- SIGN acceleration median: -0.13s (≈0!), mean -0.44s  (>0 = SIGN earlier)
- SIGN made entry >=0.5s earlier in only 43/173 videos
- CAUSE: running SIGN every OUTSIDE cycle slows patrol (238ms vs 83ms gate-only),
  so latency-honest protocol partly cancels the earlier-detection benefit.
- CONCLUSION: SIGN helps in a MINORITY of advance-signage cases; strong causal
  attribution in current draft is NOT supported. Must soften.

## W8/Q4 — paired bootstrap 95% CIs on DIFFERENCES (VLM - baseline)
- APPROACHING IoU: +0.030, 95% CI [+0.003, +0.057], p=0.030 → EXCLUDES ZERO (significant)
- INSIDE-entry signed offset: +0.77s, CI [+0.08, +1.43], p=0.029 → EXCLUDES ZERO
  but POSITIVE = VLM enters LATER (baseline MORE anticipatory). So "alerts earlier"
  is WRONG in signed terms. VLM has smaller ABSOLUTE error (2.73 vs 3.47 median |Δt|),
  baseline is more anticipatory. Must fix "alerts earlier" -> "more accurate |Δt|".

## W5/Q6 — latency distribution (p50/p95/p99, ms, greedy T=0, 480px, 60 reps)
| channel | budget | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| gate | 3 | 83.4 | 89.9 | 93.7 | 94.0 |
| ego | 8 | 135.5 | 148.8 | 153.1 | 158.5 |
| sign | 15 | 155.1 | 169.4 | 171.1 | 173.5 |
| desc | 40 | 468.4 | 508.4 | 514.2 | 515.9 |
| workers | 25 | 313.9 | 341.3 | 344.1 | 344.7 |

Composite cascade paths:
- OUTSIDE patrol (gate+sign): p50 238ms (4.2Hz), p95 259ms (3.9Hz)
- OUTSIDE + corroboration (gate+sign+desc): p50 707ms (1.4Hz), p95 768ms (1.3Hz)
- active (gate+ego): p50 219ms (4.6Hz), p95 239ms (4.2Hz)
- state-only gate: p50 83ms (12.0Hz), p95 90ms (11.1Hz)

## W7/Q8 — power / energy (MAXN mode, tegrastats, VDD_GPU_SOC+VDD_CPU_CV)
- under inference load: p50 36.5W, mean 36.9W, p95 41.3W, peak 42.5W
- idle: p50 7.9W
- energy/decision: gate cycle 83ms×36.5W ≈ 3.0J; patrol cycle (gate+sign 238ms) ≈ 8.7J;
  full corroboration cycle (707ms) ≈ 25.8J; active cycle (gate+ego 219ms) ≈ 8.0J
- visual embedding is NOT reused across channels: each channel is a separate
  prefill+decode request (re-encodes the frame). Sequential, not parallel.
  This is a real inefficiency and an obvious future optimization.

## Scoped as future work (need annotation or new baselines; not Jetson-cache-doable)
- W3 hard ego-negative category benchmark (needs manual per-category labels)
- W6 OCR+lexicon / detector+OCR baselines (needs building new systems)
- W9 full offline-vs-latency-honest protocol decomposition for macro-F1

## Item 1 (2nd-review W2) — detector+CLIP + VLM-sign fast-entry HYBRID (done 2026-07-20)
Script: pipeline/evaluate_yolo_fastentry.py; cache ~/eval_cache/yoloclip_fastentry (208).
| metric | VLM raw | Det+CLIP (paired) | Hybrid (det + text fast-entry) |
|---|---|---|---|
| frame acc | 62.5% | 62.9% | 62.9% |
| macro F1 | 0.464 | 0.470 | 0.473 |
| APPROACHING IoU (agg) | 0.152 | 0.124 | 0.140 |
| INSIDE IoU | 0.535 | 0.544 | 0.555 |
| INSIDE ev precision | 54.1% | 79.9% | 81.9% |
| INSIDE entry median |dt| | 2.73s | 3.47s | 3.05s |
Paired per-video APPROACHING IoU: hybrid-base +0.005 (p=0.61, NS); VLM-hybrid +0.025 (p=0.055); VLM-base +0.030 (p=0.032).
CONCLUSION: sign fast-entry is a transferable signal that improves even the detector (timing 3.47->3.05s, keeps 81.9% precision)
but does NOT fully explain the VLM edge (VLM still 0.152 appr IoU, enters earlier; hybrid gain over detector not significant).
Practical takeaway added to paper: detector + cheap text fast-entry may be the best practical mix. Integrated in Results (Sec 5.1).

## External OOD test (NOT in paper — user decision 2026-07-20; kept for records)
Self-recorded California footage (All_Construction_Data): CA50/CA99/I5/Merced/Fresno/Sacramento/Shasta,
day/evening/night/rain/fog, different camera (COOAU fisheye + dashboard overlay), zero ROADWork leakage.
Ground truth = annotated construction-sign timestamps (txt). Approach-window detection, same 25 signs:
  VLM cascade (GATE or SIGN keyword): 4% overall (day 10%, night/evening/fog/rain 0%)
  YOLO detector (any work-zone class): 72% overall (day 100%, evening 100%, night 50%, rain 50%, fog 40%)
CONCLUSION: cause isolated -> detector generalizes to new camera/geo/conditions; fine-tuned VLM is
markedly more domain-brittle. Both ROADWork-trained. Scripts: eval_external_ca.py, eval_external_yolo.py.
Decision: hold out of the paper for now (VLM-negative). R3 stays as disclosed limitation; no generalization claim.

## Cosmos3-Edge INT4 no Orin — PRIMEIRO teste em hardware real (2026-07-26)
Pacote `cosmos3edge-orin-deploy` (docs diziam "nunca executado em Jetson real").
Engines construídos no Orin: reasoner INT4 AWQ (1.27GB) + visual FP16 (983MB).
Edge-LLM v0.9.0 + patch Cosmos3 (6 arquivos); regressao do nosso 2B: 8/8 identicos.

### Latencia (Orin, INT4, greedy, 736x416)
| canal | Cosmos3-Edge | nosso 2B |
|---|---|---|
| GATE | 162 ms | ~90 ms |
| EGO | 182 ms | ~160 ms |
| SIGN | 556 ms | ~200 ms |
| DESC | 488 ms | ~540 ms |
(vs 90 **segundos** no meu teste BF16/HF anterior -> ~550x mais rapido)

### Generalizacao OOD — footage California (MESMOS 25 sinais anotados)
| sistema | geral | day | evening | night | rain | fog |
|---|---|---|---|---|---|---|
| nosso 2B fine-tunado | 4% | 10% | 0% | 0% | 0% | 0% |
| detector YOLO+CLIP | 72% | 100% | 100% | 50% | 50% | 40% |
| **Cosmos3-Edge base (zero-shot)** | **92%** | 100% | 100% | 100% | 75% | 80% |

### Calibracao (benchmark ROADWork, validation)
- positivos (dentro da obra): 6/6 detectados (100% recall)
- negativos puros: 3/6 FALSOS POSITIVOS -> affirmation bias forte
- EGO-state: erra (disse OUTSIDE num frame claramente INSIDE)

CONCLUSAO: o modelo BASE Cosmos3-Edge tem percepcao/generalizacao muito superior
(92% OOD vs 4% do nosso fine-tunado, e supera ate o detector 72%), com latencia
viavel em INT4. MAS precisa exatamente dos estagios que nosso curriculum faz:
hard-negative (calibracao) e ego-state (traversal state). Caminho ideal =
Cosmos3-Edge + nosso curriculum.
