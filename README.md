# Edge-Deployed Multimodal Safety Reasoning for Autonomous Vehicles in Work Zones

**UC Merced — Mi3 Lab**

Construção de um VLA (Vision-Language-Action) compacto de 2B parâmetros para
reconhecimento e navegação segura em zonas de obra viária, destilado do
Alpamayo 1.5-10B (NVIDIA) e treinado no ROADWork + PhysicalAI-AV, com destino
a deploy no NVIDIA Jetson.

---

## Estado atual (jul/2026)

O modelo ativo é um **Cosmos-Reason2-2B** com a arquitetura Alpamayo
(tokenizer discreto de trajetória, vocab 4000, 128 tokens de trajetória
futura), treinado em 4 estágios de VQA de work zone:

```
student_2b_clean_init            (2B limpo + arquitetura Alpamayo)
  └─ sft_stage4_v4/checkpoint-8622        VQA work zone (ROADWork real labels)
       └─ sft_stage5_egostate/checkpoint-1700    estado ego-cêntrico (OUTSIDE/APPROACHING/INSIDE)
            └─ sft_stage5_1_signs/checkpoint-2472     leitura de placas TTC
                 └─ sft_stage6_general/checkpoint-2194    retenção de direção geral   ← ATUAL
```

O que ele **já faz** (V+L): descreve cenas de work zone, classifica o estado
ego-cêntrico, lê placas, responde VQA geral. A inferência em vídeo
(`inference/video_inference_v16.py`) combina 4 sensores VQA num filtro
Bayesiano (HMM) com política de exibição com histerese.

O que **falta** (A de "action"): a porta de trajetória existe na arquitetura
mas nunca foi treinada — é a fase atual (ver PIPELINE.md, "Fase VLA").

---

## Fase atual: VLA — trajetória + KD

1. **Trajetória (SFT+KD)** — PhysicalAI-AV (39 chunks, 3.744 clipes), ground
   truth real de trajetória + distilação de logits do teacher Alpamayo-10B
   puro (λ=0.5, τ=0.7, KL restrita aos tokens supervisionados).
   Config: `configs/sft_kd_general_driving.yaml`. Pipeline validado por
   smoke test (job 171565): vocab teacher/student idênticos (155697),
   pico 28.4 GB VRAM em A100 80GB.
2. **CoT de direção** — ~124 clipes com reasoning anotado
   (`ood_reasoning.parquet`), preprocessor com `cot` no `components_order`.
3. **Avaliação** — minADE/FDE vs teacher (referência: teacher 0.35–1.97 m).
4. **Re-especialização work zone** — fine-tune leve devolvendo o
   conhecimento do stage 6 ao modelo que agora dirige.
5. **Jetson** — quantização W4A16 / TensorRT (só após 1–4).

---

## Datasets

| Dataset | Conteúdo | Uso |
|---------|----------|-----|
| ROADWork `pathways/` + `scene/` (→ LingoQA parquet) | 30k pares Q&A com labels humanos reais | Stages 4–6 (VQA) ✅ |
| PhysicalAI-AV (39 chunks, 213 GB) | 3.744 clipes, 4 câmeras, trajetória GT 6.4 s | Fase VLA (atual) |
| ROADWork `videos/` | vídeos crus (Boston, Denver, Chicago, Seattle) | avaliação qualitativa |

## Modelos

| Modelo | Papel |
|--------|-------|
| `models/Alpamayo-1.5-10B-A1-format` | teacher KD (original, sem fine-tune) |
| `models/Cosmos-Reason2-8B` | backbone VLM do teacher |
| `models/Cosmos-Reason2-2B` | backbone do student |
| `checkpoints/sft_stage6_general` | 2B atual (VQA work zone + geral) |
| `checkpoints/kd_general_driving` | 2B + trajetória (em treino) |

---

## Estrutura do repositório

```
mi3lab-workzone-vla/        ← este repo (github.com/Mi3-Lab/mi3lab-workzone-vla)
├── configs/                configs Hydra (symlinkados no vendor)
├── training/               train_hf.py, train_kd.py, trainer_kd.py, models/kd_model.py
├── inference/              video_inference_v16.py (atual) + archive/
├── data_prep/              conversores ROADWork→LingoQA, downloaders
├── diagnostics/            smoke tests (fasea, kd_general)
├── jobs/                   *.sbatch
├── patches/                patches do vendor + instruções de symlink
├── archive/                lineage morta (configs/jobs/scripts superados)
└── workzone_state.py       máquina de estados ego-cêntrica
```

Dependências vendored (clonadas ao lado, **fora** deste repo):
`../alpamayo-recipes` e `../alpamayo1.5` (NVlabs) — ver `patches/README.md`
para reconstruir o ambiente.

## Cluster

- Treino: partição `gpu` (A100 80GB ×2/nó) ou `cenvalarc.gpu` (L40S/H200)
- Testes rápidos: partição `test` (A100/L40S/H100, ≤1 h)
- VENV: `alpamayo-recipes/recipes/alpamayo1_5_sft/.venv`
- **Todos os dados/caches em `/data/`** — nunca `~/`
