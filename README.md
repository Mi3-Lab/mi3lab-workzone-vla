# Edge-Deployed Multimodal Safety Reasoning for Autonomous Vehicles in Work Zones

**UC Merced — Research Pipeline**

Fine-tuning do Alpamayo 1.5-10B (VLA = VLM + trajectory expert) no ROADWork dataset para reconhecimento e navegação segura em zonas de obra viária, com destino a deploy no NVIDIA Jetson AGX Orin.

---

## Pipeline

```
[ROADWork Dataset — pathways/ + scene/]
              │
              ▼
  Stage 1 ✅/🔄  SFT VLM
  Ensina o backbone a descrever cenas de work zone
  (visual encoder parcialmente descongelado — 0.1x LR)
              │
              ▼
  Stage 2 📋  SFT Trajectory Expert
  Fine-tuna o diffusion head nos waypoints do pathways/
              │
              ▼
  Stage 3 📋  Knowledge Distillation (10B → 2B)
  Student: Cosmos-Reason2-2B + expert adaptado
              │
              ▼
  Stage 4 📋  TensorRT W4A16
  Quantização para ~2GB VRAM
              │
              ▼
  Stage 5 📋  Jetson AGX Orin
  Inferência <100ms por frame
```

---

## Dataset: ROADWork (ICCV 2025)

Dataset de zonas de obra viária colhido em 5 cidades americanas (Boston, Pittsburgh, Los Angeles, San Francisco, Chicago/Columbus).

| Split | Imagens | Labels reais | Uso |
|-------|---------|-------------|-----|
| `pathways/` | 5.430 | ✅ Descrições humanas + 20 waypoints (x,y) | Stage 1 + Stage 2 |
| `scene/` | 7.416 | ✅ `scene_description` + `travel_alteration` + COCO bboxes | Stage 1 |
| `pathways_dense/` | 31.562 | Só trajetória (sem texto) | Stage 2 |
| `videos/` | ~77 GB | Raw | Inferência/avaliação |

### Datasets convertidos para LingoQA

| Dataset | Train pares | Val pares | Imagens train |
|---------|------------|----------|---------------|
| `lingoqa_pathways/` | 9.124 | 6.721 | 3.117 |
| `lingoqa_scene/` | 21.271 | 8.392 | 5.318 |
| `lingoqa_combined/` | 30.395 | 15.113 | 8.435 |

---

## Modelos

| Modelo | Tamanho | Papel | Status |
|--------|---------|-------|--------|
| `Alpamayo-1.5-10B-A1-format` | 37 GB | VLA completo (VLM + expert) | ✅ `/data/.../models/` |
| `Cosmos-Reason2-8B` | 26 GB | Backbone VLM do Alpamayo | ✅ `/data/.../models/` |
| `Cosmos-Reason2-2B` | ~8 GB | Student para KD | 📋 Pendente download |

### Checkpoints de fine-tuning

| Checkpoint | eval_loss | Status |
|------------|----------|--------|
| `checkpoints/sft_stage1_roadwork_v2/checkpoint-4500` | 0.260 | ✅ Completo |
| `checkpoints/sft_stage1_roadwork_v3/` | em andamento | 🔄 Treinando (job 166310) |

---

## Estrutura do Repositório

```
wokzone-alpamayo/
├── alpamayo1.5/                    # NVlabs/alpamayo1.5 (clonado)
├── alpamayo-recipes/               # NVlabs/alpamayo-recipes (clonado)
│   └── recipes/alpamayo1_5_sft/
│       ├── configs/
│       │   ├── sft_stage1_roadwork_v2.yaml   # pathways only
│       │   └── sft_stage1_roadwork_v3.yaml   # pathways + scene
│       ├── eval_loss.py            # avaliação de perplexidade no val set
│       ├── video_inference.py      # inferência anotada em vídeo
│       └── inference_val.py        # inferência frame a frame
├── data/roadwork/
│   ├── pathways/                   # imagens + annotations/trajectories_*.json
│   ├── scene/                      # imagens + annotations/instances_*.json
│   ├── pathways_dense/             # imagens densas + trajectories_dense*.json
│   ├── lingoqa_pathways/           # parquets convertidos (pathways/)
│   ├── lingoqa_scene/              # parquets convertidos (scene/)
│   ├── lingoqa_combined/           # pathways + scene mesclados
│   ├── convert_pathways_to_lingoqa.py
│   ├── convert_scene_to_lingoqa.py
│   └── merge_lingoqa.py
├── models/                         # pesos dos modelos
├── checkpoints/                    # checkpoints de fine-tuning
├── logs/                           # outputs SLURM + vídeos anotados
├── sft_v2.sbatch                   # job v2 (pathways, 2×H200, 3d)
├── sft_v3.sbatch                   # job v3 (combined, 2×H200, 3d, W&B)
└── video_inference.sbatch          # inferência em vídeo
```

---

## Progresso

### ✅ Concluído

- **Dataset convertido** — `pathways/` e `scene/` convertidos para LingoQA parquet usando labels reais (não sintéticos)
- **Stage 1 v2** — 4 épocas no `pathways/`, eval_loss 0.260, convergiu
  - Visual encoder parcialmente descongelado (0.1x LR)
  - Sem overfitting (train_loss ≈ eval_loss durante todo treino)
  - Vídeo de inferência: `logs/alpamayo_video_v2_boston.mp4`
- **Commit no alpamayo-recipes** — configs, scripts e `.gitignore` corrigido

### 🔄 Em andamento

- **Stage 1 v3** — job 166310, gnode028 (2×H200), dataset combinado (30.395 pares), W&B ativo

### 📋 Planejado

- **Stage 2** — SFT do trajectory expert nos waypoints do `pathways/`
- **Stage 3** — Knowledge distillation 10B → 2B (Cosmos-Reason2-2B como student)
- **Stage 4** — TensorRT W4A16
- **Stage 5** — Deploy Jetson AGX Orin

---

## Cluster

- **Partição**: `cenvalarc.gpu`
- **H200 NVL**: gnode026–029 (2 GPUs por nó, 141 GB VRAM cada)
- **L40S**: gnode017–024
- **VENV SFT**: `alpamayo-recipes/recipes/alpamayo1_5_sft/.venv`
- **Todos os dados em `/data/`** — nunca `~/` ou `~/.cache`
