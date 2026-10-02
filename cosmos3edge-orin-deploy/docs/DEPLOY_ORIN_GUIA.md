# Guia de deploy do Cosmos3-Edge no Jetson (Orin / Thor)

Última atualização: 2026-07-26 · Tudo medido em A100 80GB, salvo onde indicado.
**Nada foi executado em hardware Jetson real ainda.**

---

## 1. O que foi medido, e o que corrigir antes de começar

### 1.1 O critério de tempo real NÃO é 100 ms

O modelo usa **action chunking**: uma inferência produz um bloco de ações
executado ao longo do tempo. A NVIDIA declara *"32 actions per inference on
Jetson Thor — while achieving real-time control at 15 Hz"* (32 ÷ 15 = 2,13 s
de controle por inferência).

Nesta configuração: **60 ações a 10 Hz = 6,0 s de controle por inferência.**
O orçamento é **6,0 s**, não 100 ms.

> ⚠️ O campo `meets_realtime_deadline` em `cosmos3edge_inverse_onnx_10hz.py`
> compara contra `1/hz` = 100 ms e por isso reporta `False` mesmo quando a
> execução é folgadamente tempo real. **Não use esse campo como critério.**

### 1.2 A resolução padrão estava 6,5x acima da recomendada

A NVIDIA usa `image_size: 256` para tarefas de policy. O pipeline vinha fixo
em `resolution_tier=480`, que para o aspecto do `boston.mp4` (1920×1080) dá
**480×832**; o tier 256 dá **192×320**.

### 1.3 Números medidos (boston.mp4, PyTorch nativo, A100)

| config | latência | cabe em 6 s | desloc. Z | vs referência |
|---|---|---|---|---|
| tier480 / 30 passos | 6,61 s | não | 60,1 m | referência |
| tier256 / 30 passos | 2,28 s | ✅ | 52,8 m | 0,156 |
| tier256 / 16 passos | 1,66 s | ✅ | 53,7 m | 0,141 |
| **tier256 / 8 passos** ← recomendado | **1,31 s** | ✅ | 53,2 m | 0,148 |
| tier256 / 4 passos | 1,14 s | ✅ | 49,8 m | 0,203 (degrada) |

Para comparação, o pacote ONNX atual (tier480, 30 passos) leva **15,06 s**.

### 1.4 Referência oficial da NVIDIA

| plataforma | Policy DROID |
|---|---|
| Jetson Thor T5000 (128 GB) | **6,32 s** |
| Jetson Thor T3000 (32 GB) | 8,63 s |
| Jetson AGX Orin (64 GB) | *não benchmarkado para policy* |

Ancorando pela config equivalente (6,32 s no Thor ÷ 2,28 s no A100 ⇒ ~2,8x),
`tier256 / 8 passos` daria **≈3,7 s no Thor** — dentro do orçamento de 6 s.
**O Orin não é benchmarkado pela NVIDIA para policy**, o que sugere que o alvo
pretendido é o Thor.

---

### 1.5 Resultado final medido — configuração tier 256 pronta

Os grafos do tier 256 foram exportados, validados e têm engines construídos.
Medições ponta a ponta no `boston.mp4` (A100, amostras 60–62):

| configuração | latência/tick | cabe em 6 s | desloc. Z |
|---|---|---|---|
| ONNX tier480 / 30 passos (**pacote original**) | 15,06 s | ❌ | 60,6 m |
| ONNX tier256 / 30 passos | 3,07 s | ✅ | 53,0 m |
| **ONNX tier256 / 8 passos** ← recomendado | **1,99 s** | ✅ | 53,0 m |
| nativo tier256 / 8 passos (referência) | 1,31 s | ✅ | 53,2 m |

**7,5x sobre o pacote original**, com 3x de folga no orçamento. A trajetória
do caminho ONNX bate com a do nativo (53,0 m vs 53,2 m; erro máx. de ação
0,011), confirmando que o acoplamento VAE-novo + transformer-novo está correto.

**Engines TensorRT do VAE (tier 256, 192×320):**

| grafo | nós (pós-fold) | RAM p/ buildar | latência |
|---|---|---|---|
| chunk0 | 448 | 3,34 GB | 12,26 ms |
| chunk1 | 533 | 3,37 GB | 16,63 ms |
| chunkN | 533 | 3,37 GB | 16,81 ms ×14 |
| **VAE por tick** | | **máx 3,37 GB** | **264,2 ms** |

Projeção pro Thor (fator ~2,8x ancorado no número oficial da NVIDIA):
ONNX tier256/8 ≈ **5,6 s**, dentro do orçamento; trocando o VAE pelos engines
TensorRT deve ficar confortavelmente abaixo.

### 1.6 Resolução real por tier (medida, não deduzida)

Deduzir da tabela de bins **erra**. Medido interceptando `vae.encode`:

| tier | entrada do VAE | latente |
|---|---|---|
| 256 | `(1,3,61,192,320)` | `(1,48,16,12,20)` |
| 480 | `(1,3,61,480,832)` | `(1,48,16,30,52)` |

O transformer recebe `vision_tokens (1,48,16,11,20)` no tier 256 — o pipeline
faz uma transformação interna entre VAE e transformer. Cada componente é
exportado com o que **de fato recebe**; não tente casar os dois na mão.

## 2. ⚠️ Decisão pendente: qual resolução é a correta

**Mudar de 480 para 256 altera a trajetória**: 60,1 m contra 52,8 m de
deslocamento em 6 s (~12%). Não é ruído numérico — dentro do tier 256, variar
os passos (30/16/8) mantém ~53 m de forma consistente. **Quem muda a resposta
é a resolução.**

Qual está correta exige verdade de campo (GPS/odometria do `boston.mp4`) ou
comparação contra `pose_rel_to_abs`. **Resolver isto antes de fixar a
configuração** — não faz sentido otimizar uma configuração errada.

Além disso: este é o **modelo base da NVIDIA**, sem fine-tuning nos dados de
zona de obra do projeto.

---

## 3. O que precisa ser reexportado

O pacote ONNX está **congelado em 480×832**: o transformer espera
`position_ids` com 4969 tokens e recusa qualquer outra resolução
(`Got: 1145 Expected: 4969` ao tentar tier 256). Mudar de resolução exige
reexportar.

| artefato | estado | como reexportar |
|---|---|---|
| VAE encoder (por chunk) | pronto em 480×832 | `diagnostics/export_wan_vae_chunk_all.py --height H --width W` |
| MoT inverse transformer | congelado em 480×832 | `.../export_cosmos3edge_diffusers_inverse_transformer_onnx.py` — precisa de um **fixture** capturado na resolução alvo |
| Reasoner INT4 | pronto, independe de resolução | já empacotado |
| Visual tower | pronto | já empacotado |

### Fixture do transformer

O export do transformer parte de um fixture (entradas reais capturadas de uma
execução do pipeline), em
`models/cosmos3edge-full-onnx/denoiser/fixtures/inverse_dynamics_boston.pt`.
Para tier 256 é preciso capturar um novo fixture rodando o pipeline nativo
naquela resolução.

---

## 4. Passo a passo do deploy

### 4.1 No cluster (x86_64) — preparar os grafos

```bash
# 1. VAE encoder por chunk, na resolução escolhida
#    (192x320 para tier 256 no aspecto do boston; 480x832 para tier 480)
python mi3lab-workzone-vla/diagnostics/export_wan_vae_chunk_all.py \
    --output-dir models/cosmos3edge-vae-encoder-chunkHxW \
    --height 192 --width 320 --num-frames 61
# valida sozinho: encadeia os 16 chunks e compara com o _encode monolítico

# 2. MoT transformer — capturar fixture na resolução alvo e exportar
#    (script em COSMOS3EDGE_COMPLETE_IMPLEMENTATION_KIT/code/diagnostics/)
```

### 4.2 No Jetson — construir os engines

Engines do TensorRT são **travados ao hardware e à versão**; não copie os
engines do A100.

```bash
# Reasoner INT4 (já existia, feito pelo Codex)
./deploy/build_reasoner_on_jetson.sh <TensorRT-Edge-LLM-dir> <ONNX-dir>
./deploy/build_reasoner_visual_on_jetson.sh <...>

# VAE encoder por chunk (novo)
./deploy/build_vae_chunk_on_jetson.sh <chunk-onnx-dir> [engine-dir]
```

O script do VAE faz duas coisas por grafo:
1. `runtime/fold_vae_chunk_graph.py` — remove o nó `If` e dobra `Pad`→`Conv`
   via constant folding do ONNX Runtime. **Etapa obrigatória**: sem ela o
   parser do TensorRT rejeita o grafo.
2. `trtexec` — constrói o engine.

**Custo do build medido (A100, 480×832):** pico de **3,82 GB** de RAM do host,
~3 min por grafo. Cabe folgado nos 64 GB do Orin.

> Por que existe o modo por chunk: o encoder monolítico traz as 16 iterações
> do loop causal desenroladas (495 Conv, 44.366 nós) e exige **125,8 GB** de
> RAM para construir o engine — **impossível no Orin**. Os grafos por chunk
> têm ~533 nós e 32 Conv.

### 4.3 Rodar

```bash
export COSMOS3EDGE_ORT_TENSORRT=1
export COSMOS3EDGE_TRT_CACHE=<dir>

python runtime/cosmos3edge_inverse_onnx_10hz.py \
    --checkpoint pipeline-config --manifest <timeline.json> \
    --vae-encoder <dir-com-encoder-chunk0-fp16.onnx> \
    --inverse-transformer <onnx do transformer> \
    --output <result.json> \
    --steps 8 --resolution-tier 256
```

A seleção entre encoder monolítico e por chunk é automática
(`build_vae_encoder`): se `--vae-encoder` apontar para um **diretório**
contendo `encoder-chunk0-fp16.onnx`, usa o modo por chunk. Forçável com
`COSMOS3EDGE_VAE_CHUNKED=1|0`.

---

## 5. Armadilhas de ambiente (todas encontradas na prática)

Estas custaram várias horas em x86_64 com tudo instalado; no aarch64 do Jetson
tendem a ser piores.

| sintoma | causa | solução |
|---|---|---|
| `Cosmos3OmniPipeline.__init__() got an unexpected keyword 'default_use_system_prompt'` | diffusers padrão em vez do fork Cosmos3 | `PYTHONPATH=<pkg>/third_party/diffusers-cosmos3/src:...` **primeiro** |
| `ModuleNotFoundError: cv2` | opencv ausente | só é usado por `prepare_cosmos3edge_video_10hz.py`; dá para reusar um `timeline.json` existente |
| `libcudart.so.13` / `libcublasLt.so.12` ausente | versão de CUDA do onnxruntime ≠ a do torch | alinhar as versões, ou emprestar as libs via `LD_LIBRARY_PATH` |
| `TensorrtExecutionProvider not active` | `get_available_providers()` lista o que foi **compilado**, não o que **carrega**; faltava `libnvinfer` | instalar o TensorRT correspondente à build do ORT |
| `Got invalid dimensions for input: position_ids` | grafo congelado em outra resolução | reexportar na resolução alvo |

> **Regra prática:** `get_available_providers()` listando um provider **não**
> significa que ele funciona. Sempre execute um grafo mínimo e confira o
> resultado numérico antes de confiar.

---

## 6. Estado dos artefatos

| item | estado | onde |
|---|---|---|
| Reasoner INT4 + build script Jetson | pronto (Codex), não testado em HW real | `onnx/reasoner-int4`, `deploy/build_reasoner_on_jetson.sh` |
| Visual tower | pronto (Codex) | `onnx/visual-fp16` |
| **VAE chunk 192×320 (tier 256)** | ✅ exportado, validado, engines OK | `models/cosmos3edge-vae-encoder-chunk256` |
| **MoT transformer tier 256** | ✅ exportado, cosseno 0,999988 | `models/cosmos3edge-mot-inverse-tier256` |
| VAE chunk 480×832 (tier 480) | ✅ exportado, validado | `models/cosmos3edge-vae-encoder-chunk480` |
| MoT transformer 480×832 | pronto (Codex) | `onnx/mot-fp16-shared` |
| Fixture tier 256 | ✅ capturado | `models/cosmos3edge-fixtures/inverse_boston_tier256.pt` |
| VAE **decoder** | mesmo defeito de loop desenrolado, **não tratado** — só necessário para modalidades de geração de vídeo | — |
| Acurácia 256 vs 480 | **pendente** (ver §2) | — |
| Execução em Jetson real | **nunca feita** | — |

> ⚠️ No pacote, `onnx/vae-chunk-256` e `onnx/mot-inverse-256` são **symlinks**
> para `models/`. Ao empacotar para transferência, resolva-os
> (`cp -rL` ou `tar -h`) — o transformer tem 5,8 GB.

### Como rodar a configuração recomendada

```bash
export PYTHONPATH=<pkg>/third_party/diffusers-cosmos3/src:<pkg>/runtime

python runtime/cosmos3edge_inverse_onnx_10hz.py \
    --checkpoint pipeline-config --manifest <timeline.json> \
    --vae-encoder <pkg>/onnx/vae-chunk-256 \
    --inverse-transformer <pkg>/onnx/mot-inverse-256/model.onnx \
    --output result.json \
    --steps 8 --resolution-tier 256 \
    --start-sample 60 --stop-sample 63
```

---

## 7. Validações já realizadas (para não repetir)

- **Cirurgia do VAE preserva a matemática**: bit-exato contra o grafo original.
- **Encoder por chunk ≡ monolítico**: erro abs. máx. 4,9e-03 (ruído fp16).
- **Integração ponta a ponta**: no `boston.mp4`, o encoder por chunk produz
  ações **bit-exatas** (erro 0,0) frente ao monolítico, em 3 ticks.
- **A lentidão do ONNX era falta de fusão**, não precisão: o TensorRT resolve
  sem quantizar nada. Mas atenção — o ONNX Runtime é **2,5x mais lento que o
  PyTorch nativo**; só o TensorRT supera o nativo, e por ~1,4x.

## 8. Referências

- [Introducing Cosmos 3 Edge](https://huggingface.co/blog/nvidia/cosmos3edge)
- [nvidia/Cosmos3-Edge — model card](https://huggingface.co/nvidia/Cosmos3-Edge)
- `RELATORIO_COSMOS3EDGE_TENSORRT_FUSAO.md` (raiz do repo) — investigação completa
