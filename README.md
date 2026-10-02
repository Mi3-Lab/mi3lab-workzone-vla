# Work Zone VLA 2B — pacote de deploy para Jetson (Orin / Thor)

> **Rodar no Thor:** guia completo em [`thor/README.md`](thor/README.md).
> **Papers:** `paper/iv2027/` (IEEE IV 2027, alvo atual), `paper/journal/` (versão de revista), `paper/wacv2027/` (histórico).

Modelo: `Cosmos-Reason2-2B` (arquitetura Qwen3-VL) fine-tuned em detecção de
zona de obras (ROADWork), lineage `student_2b_clean_init → stage4_v4 →
stage5_egostate → stage5_1_signs → stage6_general →
stage7_1/checkpoint-1394` (hard negatives + gate). Quantizado INT4 AWQ
(W4A16, group_size 128) e exportado via
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) da NVIDIA
(suporte oficial ao Cosmos-Reason2-2B / Qwen3-VL).

**Status: validado com o engine TensorRT real (não simulação) em positivos
E negativos** — 685 requisições nos 4 vídeos de obra (0% degeneração, 4/4
zonas detectadas) + 3.111 requisições de controle negativo (249 imagens
BDD100k + vídeo contínuo de rodovia sem obra: **0% de falso positivo**).
Ver seção [Validação de qualidade](#validação-de-qualidade) abaixo.

## O que tem aqui

```
onnx/
├── llm/      model.onnx + embedding.safetensors + tokenizer (backbone de linguagem, INT4)
└── visual/   model.onnx (encoder de visão Qwen3-VL, FP16 — não quantizado)
pipeline/
└── cascade_state_machine.py   (máquina de estados evidence-first — usar em vez
                                do filtro Bayesiano puro; ver seção Pipeline)
```

Tamanho total: ~4.5 GB. O grosso da economia de VRAM em runtime vem da
compilação em INT4 no engine, não do tamanho do ONNX em si (pesos INT4 são
empacotados no `.engine` gerado no passo 3, não neste ONNX).

## ⚠️ Se for re-exportar este ONNX a partir de um checkpoint novo

Este pacote já tem o fix aplicado, mas se você re-treinar/re-quantizar e
re-exportar do zero, **verifique `config.json` antes de rodar
`tensorrt-edgellm-export`**:

```bash
grep tie_word_embeddings <checkpoint_dir>/config.json
```

Tem que ser `"tie_word_embeddings": false`. Este checkpoint teve o SFT
treinando `lm_head.weight` separado da embedding de entrada, mas o
`config.json` exportado herdava `tie_word_embeddings: true` do modelo base.
O `transformers` (HuggingFace) detecta a divergência entre os dois tensores
e ignora o tie automaticamente (com um warning) — por isso toda validação
em Python/BF16 sempre pareceu perfeita. O loader C++ do Edge-LLM
(`checkpoint/loader.py:189`) **não tem essa proteção**: confia cegamente no
flag e sobrescreve `lm_head.weight` com a embedding de entrada sempre que
`tie_word_embeddings=true`, mesmo quando os dois tensores existem com
valores diferentes no checkpoint.

**Sintoma se isso acontecer de novo**: a primeira palavra gerada quase
sempre um token de baixa frequência tipo `.DATA`, `.INSTANCE`, ou lixo
similar, seguido por texto ok ou por um loop de repetição — porque o logit
do token especial `<|answer_start|>` (o mais afetado pela troca de matriz)
morre e o sampling reforça a cauda ruidosa da distribuição. Root-caused e
corrigido nesta sessão: ver `mi3lab-workzone-vla/diagnostics/` para os
scripts de diagnóstico (`diag_fp16_overflow.py`,
`compare_bf16_broken_frames.py`) e comparação de pesos ONNX vs checkpoint.
Reportar/checar antes de reabrir: possível issue upstream em
[NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues).

## Compilar o engine NO PRÓPRIO JETSON

Engines TensorRT são travados na arquitetura exata de GPU usada na
compilação (compute capability). Este ONNX foi gerado num A100 (x86) — o
**engine final tem que ser compilado no Jetson de destino**, não pode ser
copiado de outra GPU (nem entre Orin e Thor, que já são compute
capabilities diferentes entre si).

### 1. Instalar/compilar o TensorRT Edge-LLM no Jetson

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM
cd TensorRT-Edge-LLM

# IMPORTANTE: clone raso não traz os submódulos (nlohmann/json, NVTX,
# googletest ficam como diretórios vazios). Inicializar manualmente:
git submodule update --init --depth 1

mkdir -p build && cd build

# Jetson AGX Orin (JetPack 6.x/7.x, verificar `apt show nvidia-jetpack`)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=12.6 \
    -DENABLE_CUTE_DSL=ALL

# OU Jetson AGX Thor (JetPack 7.2, CUDA 13.2):
# cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DTRT_PACKAGE_DIR=/usr \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
#     -DEMBEDDED_TARGET=jetson-thor \
#     -DCUDA_CTK_VERSION=13.2 \
#     -DENABLE_CUTE_DSL=ALL

# Alvos necessários: llm_build, llm_inference, visual_build E o plugin
# customizado (NvInfer_edgellm_plugin) — o build default pode não incluir
# o plugin, e sem ele o parse do ONNX falha com "Plugin not found".
make -j$(nproc) llm_build llm_inference visual_build NvInfer_edgellm_plugin
```

Binários resultantes: `build/examples/llm/llm_build`,
`build/examples/llm/llm_inference`, `build/examples/multimodal/visual_build`,
`build/libNvInfer_edgellm_plugin.so`.

### 2. Copiar este diretório `onnx/` para o Jetson

```bash
scp -r onnx/ <user>@<jetson-ip>:~/tensorrt-edgellm-workspace/workzone-2b/
```

### 3. Compilar os engines (no Jetson)

O plugin customizado precisa estar carregado durante o build — `llm_build`
e `visual_build` usam `LD_PRELOAD` (mecanismo diferente do runtime de
inferência, que usa uma variável de ambiente própria — ver passo 4):

```bash
cd TensorRT-Edge-LLM
PLUGIN=$PWD/build/libNvInfer_edgellm_plugin.so
WS=~/tensorrt-edgellm-workspace/workzone-2b

LD_PRELOAD=$PLUGIN ./build/examples/llm/llm_build \
  --onnxDir $WS/onnx/llm \
  --engineDir $WS/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096

LD_PRELOAD=$PLUGIN ./build/examples/multimodal/visual_build \
  --onnxDir $WS/onnx/visual \
  --engineDir $WS/engines \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

Tempo esperado: poucos minutos por engine (referência A100: ~1-2 min cada;
Jetson deve ser mais lento, mas é operação única, feita uma vez por device).

### 4. Rodar inferência

Criar `input.json`:

Este pacote já inclui um frame de exemplo real (`test/sample_frame.jpg`,
Boston, dentro de zona de obra) — use-o direto pra validar que o engine
compilou certo, sem precisar arrumar uma imagem antes:

```json
{
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 60,
    "requests": [{
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "image": "test/sample_frame.jpg"},
                {"type": "text", "text": "Describe the work zone elements visible in this scene."}
            ]
        }]
    }]
}
```

Resposta esperada (validada nesta sessão): algo como *"Barricade on right
side of road. Tubular markers on right side of road..."* — se sair
`.DATA` ou lixo no início, o `tie_word_embeddings` voltou a `true` em
algum re-export (ver seção de aviso acima).

`llm_inference` usa uma variável de ambiente diferente do build
(`EDGELLM_PLUGIN_PATH`, não `LD_PRELOAD`) para localizar o plugin:

```bash
EDGELLM_PLUGIN_PATH=$PLUGIN ./build/examples/llm/llm_inference \
  --engineDir $WS/engines/llm \
  --multimodalEngineDir $WS/engines/visual \
  --inputFile input.json \
  --outputFile output.json \
  --dumpOutput
```

## Parâmetros de sampling — usar os valores validados

**Não usar greedy puro** (`temperature=0`/`top_k=1`). Toda a validação de
685 requisições reais (ver abaixo) rodou com `temperature=0.4, top_p=0.9,
top_k=40` — são os parâmetros testados e aprovados, não apenas uma
recomendação teórica. O runtime do Edge-LLM não expõe `repetition_penalty`
(esse parâmetro só existe no runtime de TTS/Qwen3-Omni), então sampling
leve é a mitigação disponível para os casos raros de degeneração sob
decodificação puramente gulosa observados em teste anterior (cena com
muita placa/texto).

## Pipeline recomendado — cascade evidence-first (NÃO usar só o filtro Bayesiano)

A validação com controles negativos (2026-07-08) mostrou que o design
antigo (filtro Bayesiano alimentado direto pelas 5 perguntas) tinha um
defeito estrutural: a pergunta EGO pressupõe que a obra existe (leading
question) e um único canal conseguia "sequestrar" o estado sem corroboração
— numa rodovia sem obra, o sistema ficava travado em APPROACHING.

`pipeline/cascade_state_machine.py` implementa o design corrigido:

1. **Gate de existência** (pergunta neutra binária: *"Are there any road
   work indicators in this scene (cones, barriers, temporary signs,
   workers, work vehicles)? Answer Yes or No."*) roda primeiro em todo
   frame. Se No, estado = OUTSIDE e as demais perguntas nem rodam
   (ciclo ~3x mais barato fora de obra).
2. **Entrada exige evidência sustentada + corroboração**: gate Yes em ≥3
   dos últimos 5 frames E um segundo canal citando objeto de obra
   (DESC/SIGN) em pelo menos 1 deles.
3. **Saída com histerese**: gate No em ≥4 dos últimos 5 frames.
4. O filtro Bayesiano original continua rodando DENTRO do modo ativo
   (APPROACHING/INSIDE/EXITING), onde é comprovadamente bom.

O modelo (stage7_1) foi treinado especificamente para suportar esse
pipeline: a pergunta do gate e as respostas negativas canônicas fazem
parte do treino (10k hard negatives BDD100k + 2.5k gate-positivos + 2k
positivos de DESC/TRAFFIC + 8k replay).

## Validação de qualidade

Validação real via engine TensorRT (não simulação Python), positivos e
negativos, fim-a-fim no `llm_inference` real.

**Positivos** (4 vídeos de obra — Boston/Denver/Chicago/Seattle, 685
requisições frame×pergunta): 4/4 zonas detectadas pela cascade
(APPROACHING→INSIDE em todos), DESC descrevendo os elementos, zero
degeneração de texto.

**Negativos** (249 imagens BDD100k + vídeo contínuo de 47s de rodovia sem
obra, nunca vistos em nenhum treino):

| Métrica (negativos) | stage6 (antes) | **stage7_1 (este pacote)** |
|---|---|---|
| Gate responde "Yes" em cena limpa | 95-100% | **0%** |
| EGO ≠ OUTSIDE em cena limpa | 34-100% | **0%** |
| DESC alucina objetos de obra | 25% | **0%** |
| Vídeo contínuo: tempo preso em APPROACHING | 99% | **0%** (100% OUTSIDE, zero transições) |

**Histórico de bugs corrigidos neste pacote** (detalhes nas seções acima):
- Bug do `tie_word_embeddings` (engine usava embedding no lugar do lm_head
  treinado): 92% das respostas começavam com token-lixo (`.DATA`) → 0%.
- Viés "sempre-sim" (SFT sem exemplos negativos): corrigido no stage7_1
  com hard negatives; o Cosmos-2B base já discriminava (0% falso Yes) e o
  fine-tuning sem negativos tinha destruído essa capacidade.

### Latência real medida (A100, referência — Jetson deve ser mais lenta)

| Etapa | Tempo médio |
|---|---|
| Vision Encoder (por imagem) | 21.6–22.0 ms |
| LLM Prefill (contexto médio 324 tokens) | 27.1–27.4 ms |
| Geração (por token) | 2.5–2.7 ms (~375–398 tok/s) |
| Pico de memória GPU | 3.4–3.5 GB |

Comparado a BF16 via `transformers.generate()` no mesmo checkpoint: 108 ms
(INT4 TensorRT) vs 515 ms (BF16 HF) para 20 tokens de saída — **speedup de
4.8x**, conteúdo idêntico. Números de A100; refazer a medição no Jetson de
destino antes de comprometer com um orçamento de latência final (a
GPU embarcada tem menos SMs, mas o pipeline INT4 foi desenhado justamente
para esse cenário).

Scripts de validação (reprodutíveis): `mi3lab-workzone-vla/diagnostics/`
(`eval_cascade_design.py` — benchmarks positivo/negativo com a cascade;
`eval_stage7_1_bf16.py` — avaliação BF16; `analyze_lmheadfix_batch.py` —
taxas de degeneração de texto) e `mi3lab-workzone-vla/jobs/deploy_stage7_1_full.sbatch`
(pipeline completo quantização→ONNX→engine→batches, reprodutível se o
checkpoint mudar).

## Calibração usada na quantização

`data/roadwork/calib_workzone.jsonl` — 512 amostras de texto do domínio
(mix de work zone/ego-state/placas/direção geral, mesma proporção dos dados
de treino do stage 6), não o dataset genérico de notícias (default do
Edge-LLM seria `cnn_dailymail`). Testado contra 2000 amostras e contra
calibração multimodal (imagem+pergunta) — 512 amostras de texto puro
performou melhor nos dois casos (77% vs 72% e 70% de match com BF16,
respectivamente).

## Limitações conhecidas

- **Visual tower fica em FP16**, não quantizado — calibração multimodal
  (imagem+texto) para quantizar a visão também degradou resultados nos
  testes feitos (ver acima); mantido em FP16 por ora.
- **Edge-LLM só suporta FP16** (não BF16) por decisão de projeto da NVIDIA
  ([issue #54](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/54)) —
  confirmamos que nenhuma ativação deste checkpoint excede o range do fp16
  (máx observado 20352 contra limite 65504), então não é um risco aqui.
- **Sem `repetition_penalty`** no runtime base — mitigar com sampling leve
  (ver seção de parâmetros acima).
- **Engine travado na compute capability** — recompilar sempre que o
  hardware de destino mudar (Orin ↔ Thor ↔ x86 são todos incompatíveis
  entre si).

## Origem

- Checkpoint fonte: `checkpoints/sft_stage7_1/checkpoint-1394` (stage6 +
  hard negatives BDD100k + gate + positivos DESC/TRAFFIC)
- Export HF: `models/workzone-2b-stage7-1-hf/` (Qwen3VLForConditionalGeneration,
  vocab 155.697 tokens, `tie_word_embeddings: false`)
- Quantizado: `models/workzone-2b-stage7-1-int4awq/` (W4A16_AWQ, group_size
  128, visual tower e lm_head preservados fora do INT4)
- Este ONNX: `models/workzone-2b-stage7-1-onnx/` (llm E visual — a visual
  tower também treinou no stage7_1 com lr 0.1x, os dois engines precisam
  ser recompilados)
- Dataset do stage 7.1: `data/roadwork/lingoqa_stage7_1/` (builder:
  `mi3lab-workzone-vla/data_prep/build_stage7_negatives.py` + `build_stage7_1.py`)
