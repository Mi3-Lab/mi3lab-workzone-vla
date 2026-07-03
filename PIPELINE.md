# Pipeline de Treinamento — Alpamayo Work Zone Safety

**UC Merced | Edge-Deployed Multimodal Safety Reasoning for Autonomous Vehicles**

---

## Arquitetura do Alpamayo 1.5-10B

O Alpamayo é um VLA (Vision-Language-Action model) com dois componentes independentes:

```
Imagem → [vlm.*]  Cosmos-Reason2-8B    → texto de reasoning
          [expert.*] Trajectory diffusion → 20 waypoints (x,y) pixel space
```

- `vlm.*` — backbone de linguagem+visão; treinado no Stage 1
- `expert.*` — diffusion head que prediz trajetórias; treinado no Stage 2
- Os dois módulos compartilham os embeddings visuais do VLM

---

## Stage 1 — SFT VLM (Fine-tuning do Backbone de Linguagem)

### Objetivo

Ensinar o Cosmos-Reason2-8B a reconhecer e descrever elementos de zonas de obra: cones, barreiras, trabalhadores, arrow boards, TTC signs, travel alterations. Sem esse conhecimento visual, o expert de trajetória do Stage 2 toma decisões com base numa percepção errada da cena.

### O que fica congelado e o que treina

| Módulo | LR multiplier | Justificativa |
|--------|-------------|---------------|
| `vlm.model.visual.*` (SigLIP2 encoder) | 0.1x | Parcialmente descongelado — aprende features visuais de work zones |
| `vlm.model.language_model.*` | 1.0x (5e-6) | Treina normalmente |
| `expert.*` (trajectory diffusion) | 0.0 | Congelado — treinado só no Stage 2 |

### Lição aprendida: v1 falhou

A primeira tentativa (v1) produziu um modelo que dizia "dangerous work zone — reduce speed immediately" para **todos** os frames, inclusive asfalto vazio. Motivos:

1. **Labels sintéticos** — Q&A gerado automaticamente dos bboxes COCO com 10 templates rígidos. Sem negativos. Modelo decorou o template, não aprendeu a ver a cena.
2. **Visual encoder congelado** (0.0x LR) — o modelo nunca aprendeu a olhar para a imagem.
3. **Sem exemplos negativos** — 100% das amostras descreviam perigo. Modelo aprendeu a sempre dizer "perigo".

### Solução: v2 com labels reais

O ROADWork paper tem dois campos de labels reais que não estávamos usando:

- `pathways/annotations/trajectories_*.json` → campo `description`: texto humano real descrevendo a cena
- `scene/annotations/instances_*.json` → campo `scene_description`: idem, mais `travel_alteration` (Partially Blocked, Fully Blocked, Lane Shift, None)

### Dataset Stage 1

**v2** — só `pathways/`:
- 3.117 imagens de treino → 9.124 pares Q&A
- 2.313 imagens de val → 6.721 pares Q&A
- ~8% negativos ("No Description" = cena sem work zone)

**v3** — `pathways/` + `scene/` combinados:
- 8.435 imagens de treino → 30.395 pares Q&A (3.3× mais que v2)
- 4.411 imagens de val → 15.113 pares Q&A
- ~9.6% negativos
- W&B habilitado (projeto `wokzone-alpamayo`, equipe `usp`)

### Tipos de Q&A gerados por imagem

| Tipo | Pergunta | Fonte do label |
|------|----------|---------------|
| Descrição da cena | "Describe the work zone elements visible in this scene." | `description` / `scene_description` |
| Inventário de objetos | "What objects are present and where?" | COCO detections |
| Ativo vs. passivo | "Are workers present? What does this mean for vehicle speed?" | COCO categories (Worker, Police Officer) |
| Travel alteration (só scene/) | "How is traffic flow affected?" | `travel_alteration` tag |
| Negativo | "Is there active road work?" | Amostras sem descrição / travel_alteration=None |

### Configuração técnica

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | `Alpamayo-1.5-10B-A1-format` |
| VLM backbone | `Cosmos-Reason2-8B` |
| Hardware | 2× H200 NVL (141 GB VRAM cada) |
| Batch efetivo | 8 (2 GPUs × 1 sample × GA=4) |
| Learning rate | 5e-6, cosine decay |
| Warmup | 100 steps |
| Épocas | 4 |
| eval_steps | 500 |
| Melhor checkpoint | `load_best_model_at_end=True` por `eval_loss` |

### Resultados

**v2 — convergido (job 166121, gnode028):**

| Step | Epoch | eval_loss |
|------|-------|-----------|
| 500  | 0.44  | 0.472 |
| 1000 | 0.88  | 0.328 |
| 1500 | 1.31  | 0.289 |
| 2000 | 1.75  | 0.271 |
| 2500 | 2.19  | 0.264 |
| 3000 | 2.63  | 0.262 |
| 3500 | 3.07  | 0.261 |
| 4000 | 3.51  | 0.260 |
| 4500 | 3.94  | **0.260** ← best |

- Sem overfitting: train_loss ≈ eval_loss durante todo o treino
- Best checkpoint: `checkpoints/sft_stage1_roadwork_v2/checkpoint-4500`

**v3 — em andamento (job 166310, gnode028):**
- Dataset: `lingoqa_combined/` (pathways + scene)
- W&B: projeto `wokzone-alpamayo`, equipe `usp`

### Inferência em vídeo (Stage 1)

O script `video_inference.py` roda o VLM frame a frame em qualquer vídeo. Pergunta usada na inferência: *"Describe the work zone elements visible in this scene."* — a mesma do treino.

```bash
sbatch video_inference.sbatch <video.mp4> <output.mp4> <checkpoint_path>
```

Nota: este script testa APENAS o VLM (linguagem). O expert de trajetória não é chamado — isso é Stage 2.

---

## Stage 2 — SFT Trajectory Expert

### Objetivo

Descongelar o `expert.*` (diffusion head) e treiná-lo para gerar trajetórias seguras em work zones. O VLM (Stage 1) fornece os embeddings visuais corretos; o expert aprende a reagir a eles.

### Dataset

- `pathways/annotations/trajectories_train_equidistant.json` — 3.117 amostras com 20 waypoints (x,y) cada
- `pathways_dense/annotations/trajectories_dense__train_equidistant.json` — 18.100 amostras (sem texto, só trajetória)

### Configuração planejada

```yaml
trainer:
  lr_multiplier:
    vlm.model.visual: 0.0    # congelado
    vlm.model.language_model: 0.01  # quase congelado
    expert: 1.0              # treina completamente
  learning_rate: 2e-6
  num_train_epochs: 4
```

### Métrica

- ADE (Average Displacement Error) e FDE (Final Displacement Error) vs. trajetórias ground truth do motorista humano

---

## Stage 3 — Knowledge Distillation (10B → 2B)

### Arquitetura do student

**`nvidia/Cosmos-Reason2-2B`** — mesma família do teacher, mesmo visual encoder (SigLIP2), sem necessidade de adaptador. Trajectory expert transferido/adaptado do teacher com LoRA se houver mismatch de dimensão.

### Pipeline

1. **Logit distillation** — `L = L_SFT + λ·KL(teacher || student)`, temperatura τ=0.7
2. **Feature distillation** (Drive-KD) — hidden states por capacidade (percepção / raciocínio / planejamento)
3. **On-policy refinement** — student gera, teacher avalia
4. **Expert adaptation** — transferir pesos do diffusion head

### Referências

- Drive-KD (arxiv:2601.21288) — metodologia principal
- LLAVADI (arxiv:2407.19409) — temperatura ótima τ=0.7 para VLMs
- SPEED-Q (arxiv:2511.08914) — quantização staged integrada à distilação

---

## Stage 4 — TensorRT W4A16

```bash
trtllm-build \
  --model_dir ./models/alpamayo_student_2b \
  --quantization W4A16 \
  --max_batch_size 1 \
  --output_dir ./models/alpamayo_student_2b_trt
```

Memória esperada no Jetson: LLM ~2 GB + ViT ~1.5 GB + KV cache ~5 GB = **~8.5 GB total**.

---

## Stage 5 — Jetson AGX Orin

Latência estimada por frame: SigLIP2 INT8 (~8ms) + LLM prefill (~20ms) + decode 80 tokens (~30ms) + diffusion expert 4 steps (~25ms) = **~83ms** (target: <100ms).

---

## Cronograma

| Stage | Status | ETA |
|-------|--------|-----|
| Stage 1 v2 (pathways) | ✅ Completo | Jun 2026 |
| Stage 1 v3 (pathways+scene) | 🔄 Job 166310 | Jun/Jul 2026 |
| Stage 2 (trajectory expert) | 📋 Planejado | Jul 2026 |
| Stage 3 (KD 10B→2B) | 📋 Planejado | Set/Out 2026 |
| Stage 4 (TensorRT) | 📋 Planejado | Out/Nov 2026 |
| Stage 5 (Jetson) | 📋 Planejado | Nov/Dez 2026 |
