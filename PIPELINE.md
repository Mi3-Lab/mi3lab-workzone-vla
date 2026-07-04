# Pipeline de Treinamento — Mi3 Lab Work Zone VLA

**UC Merced | Edge-Deployed Multimodal Safety Reasoning for Autonomous Vehicles**

---

## Arquitetura

O Alpamayo 1.5 (NVIDIA) é um VLA cujo backbone é um Qwen3-VL
(Cosmos-Reason2) com o vocabulário expandido por tokens discretos de
trajetória:

```
Imagens (4 câmeras × 4 frames) ─┐
Histórico de trajetória (48 tok)─┤→ [vlm: Cosmos-Reason2] → CoT (texto) + trajetória futura (128 tok)
Rota                            ─┘                                    │
                                              DiscreteTrajectoryTokenizer.decode → waypoints (x,y,z)
```

O nosso student 2B (`kd_model.build_student_model`) usa a **mesma
arquitetura** com o backbone Cosmos-Reason2-2B: mesmo tokenizer de
trajetória (vocab 4000), mesmos 48/128 tokens de história/futuro, mesmo
vocabulário final (155.697 — idêntico ao teacher, o que permite KD de
logits direto).

---

## Histórico — como chegamos ao 2B atual

### Stages 1–3 no 10B (superados, ver `archive/`)

- **Stage 1 v1** (10B, labels sintéticos) — falhou: dizia "dangerous work
  zone" pra tudo. Lições: labels reais, negativos, encoder visual em 0.1x.
- **Stage 1 v2/v3** (10B, labels humanos reais do ROADWork) — convergiu
  (eval_loss 0.260), mas o 10B é grande demais pro Jetson.
- **Stage 3 KD v1** (10B fine-tuned → 2B via LingoQA) — abandonado em favor
  de treinar o 2B diretamente nos dados (mais simples, mesmo resultado).

**Decisão estrutural**: treinar o **2B diretamente** (student_2b_clean_init
= Cosmos-Reason2-2B + arquitetura Alpamayo, embeddings redimensionados) e
usar o 10B só como teacher/referência.

### Stages 4–6 no 2B (lineage ativa) ✅

| Stage | Checkpoint | O que ensinou |
|-------|-----------|----------------|
| 4 v4 | `sft_stage4_v4/checkpoint-8622` | VQA work zone (ROADWork pathways+scene, 30k pares, labels reais, ~10% negativos) |
| 5 | `sft_stage5_egostate/checkpoint-1700` | Estado ego-cêntrico: OUTSIDE/APPROACHING/INSIDE (posição do ego vs início da zona, não conteúdo da cena) |
| 5.1 | `sft_stage5_1_signs/checkpoint-2472` | Leitura de placas TTC ("ROAD WORK AHEAD", "UTILITY WORK"…) |
| 6 | `sft_stage6_general/checkpoint-2194` | Retenção de direção geral (mix anti-esquecimento) ← **ATUAL** |

### Inferência em vídeo (v16) ✅

`inference/video_inference_v16.py` — 4 perguntas VQA por ciclo (estado ego,
fluxo de tráfego, ativo/passivo, leitura de placa) viram emissões de um
filtro Bayesiano (HMM 4 estados); a exibição usa `DisplayPolicy` com
histerese (COMMIT=0.60), refutação (REFUTE=0.15) e transições só entre
estados fisicamente adjacentes. Validado em Boston/Denver/Chicago/Seattle.

---

## Fase VLA (atual) — dar "ação" ao 2B

A porta de trajetória do 2B existe mas nunca foi treinada (loss ~25 em
traj_future no smoke test = chute aleatório). Plano:

### 1. Trajetória — SFT + KD (em andamento)

- **Config**: `configs/sft_kd_general_driving.yaml`
- **Dados**: PhysicalAI-AV, 39 chunks (19 oficiais + 20 reasoning-dense),
  3.744 clipes válidos, keyframe fixo t=5.1s (`use_default_keyframe`).
  **Sem** `reasoning_metadata` — o filtro de eventos derruba 3.744→103
  clipes (bug descoberto na Fase A: chunk 214 → 0 clipes).
- **Loss**: `L = 0.5·L_SFT + 0.5·τ²·KL(teacher‖student)`, τ=0.7,
  **KL restrita aos tokens supervisionados** (`labels_mask`) — sem a
  máscara a KL soma sobre os 3.204 tokens da sequência (imagem+prompt) e
  infla ~480x (11.855 vs 5.81 medido), afogando o sinal SFT.
- **Teacher**: Alpamayo-1.5-10B-A1-format **original** (sem fine-tune) —
  17.6 GB VRAM congelado; student treina com visual em 0.1x, expert 0.0x.
- **Validação**: smoke test job 171565 — loss_sft 25.02, loss_kd 5.81,
  pico 28.4 GB (A100 80GB ok).

### 2. CoT de direção (próximo)

Preprocessor `nav_cot` com `cot` em `components_order`/`label_components`;
treina nos ~124 clipes com reasoning real (`ood_reasoning.parquet`,
filtrados pelas margens de segurança 1.6s/6.4s). Dá o "porquê" junto com o
"pra onde".

### 3. Avaliação

minADE/FDE no val set (chunks 2868, 3126) vs teacher (referência medida:
0.35–1.97 m, job 170613).

### 4. Re-especialização work zone

Fine-tune leve devolvendo o conhecimento VQA do stage 6 ao modelo com
trajetória (mix pequeno de LingoQA work zone + trajetória pra não esquecer
nenhum dos dois).

### 5. Deploy Jetson

TensorRT W4A16 (~2 GB LLM + ~1.5 GB ViT + KV cache). Meta <100 ms/frame.
(Pruning Minitron do 10B via Megatron-Bridge foi avaliado e **arquivado**
por custo de engenharia — ver discussão em jul/2026.)

---

## Bugs estruturais conhecidos (e como evitá-los)

| Bug | Sintoma | Causa/estado |
|-----|---------|--------------|
| `reasoning_metadata` filtra quase tudo | `PAIDataset` com 0–103 clipes de 3.744 | `filter_clips_by_event_t0s` descarta clipes sem entrada no parquet de reasoning (só 124 têm). Omitir o parâmetro em treino de trajetória pura. |
| `collate_fn_from_model_config` não é currificável | `TypeError: missing 1 required positional argument: 'data'` | `data` é posicional; nos scripts standalone chamar direto com `(samples, model_config=..., chat_template_version=...)`. No Hydra funciona por causa do `_partial_: true`. |
| KL sem máscara | loss_kd ~1000x maior que loss_sft | Corrigido em `trainer_kd.py` (usa `labels_mask`). |
| Logprob scoring congela | scores idênticos pra qualquer imagem | Checkpoints stage 5+ — usar `generate()` real, nunca scoring de logprobs. |

---

## Cronograma

| Fase | Status |
|------|--------|
| Stages 4–6 (VQA work zone no 2B) | ✅ jun/2026 |
| Inferência v16 (filtro Bayesiano) | ✅ jul/2026 |
| Trajetória SFT+KD | 🔄 jul/2026 |
| CoT de direção | 📋 |
| Avaliação minADE/FDE | 📋 |
| Re-especialização work zone | 📋 |
| TensorRT + Jetson | 📋 |
