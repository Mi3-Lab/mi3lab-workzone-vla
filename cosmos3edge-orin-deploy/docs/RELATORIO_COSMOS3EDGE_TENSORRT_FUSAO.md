# Cosmos3-Edge: TensorRT resolve a lentidão do ONNX (sem quantizar)

Data: 2026-07-25/26 · GPU de teste: A100 (PCIe 40GB e 80GB) · TensorRT 10.3.0

> ## ⚠️ LEIA ISTO PRIMEIRO
>
> ### O critério de tempo real estava errado (corrigido em 2026-07-26)
>
> O modelo usa **action chunking**: uma inferência produz um BLOCO de ações
> executadas ao longo do tempo. A NVIDIA declara *"32 actions per inference
> on Jetson Thor — while achieving real-time control at 15 Hz"* (32/15 =
> 2,13 s de controle por inferência).
>
> Aqui: **60 ações a 10 Hz = 6,0 s de controle por inferência.** O orçamento
> é **6,0 s**, não os 100 ms do período do tick que `meets_realtime_deadline`
> usa em `cosmos3edge_inverse_onnx_10hz.py`. Todas as conclusões anteriores
> de "43x acima do orçamento" partiam dessa premissa errada.
>
> ### A configuração também estava errada
>
> A NVIDIA usa **`image_size: 256`** para tarefas de policy; o pipeline vinha
> fixo em `resolution_tier=480` → 480×832 para o aspecto do boston, **~6,5x
> mais pixels**. Referência oficial no model card: Policy DROID leva
> **6,32 s no Jetson Thor T5000** e 8,63 s no T3000.
>
> ### Varredura medida no boston.mp4 (PyTorch nativo, A100)
>
> | config | latência | cabe em 6 s? | desloc Z | vs ref |
> |---|---|---|---|---|
> | tier480 / 30 passos | 6,61 s | não | 60,1 m | referência |
> | tier256 / 30 passos | **2,28 s** | ✅ | 52,8 m | 0,156 |
> | tier256 / 16 passos | 1,66 s | ✅ | 53,7 m | 0,141 |
> | **tier256 / 8 passos** | **1,31 s** | ✅ | 53,2 m | 0,148 |
> | tier256 / 4 passos | 1,14 s | ✅ | 49,8 m | 0,203 |
>
> Contra os **15,06 s** do pacote ONNX atual (tier 480, 30 passos), o ponto
> de operação tier256/8-passos dá **11,5x**. Reduzir passos de 30→8 quase não
> muda a trajetória (53,2 m vs 52,8 m); só em 4 passos degrada.
>
> **Projeção pro Thor, ancorada no número oficial da NVIDIA** (6,32 s no Thor
> com a config deles ÷ 2,28 s no A100 na mesma config ⇒ razão ~2,8x):
> tier256/8-passos ≈ **3,7 s no Thor**, dentro do orçamento de 6 s.
>
> ### Ressalva de acurácia (não resolvida)
>
> Mudar 480 → 256 altera a trajetória: **60,1 m vs 52,8 m** (~12%). Dentro do
> tier 256 os passos 30/16/8 dão ~53 m consistentemente — **quem muda a
> resposta é a resolução, não os passos**. Qual está correta exige a
> trajetória real do veículo; é validação de acurácia, ainda pendente.
>
> ### Comparação de runtime (61 frames, 480×832)
>
> | componente | PyTorch nativo | ONNX Runtime | TensorRT |
> |---|---|---|---|
> | VAE encoder | 1 002 ms | 8 679 ms (8,7x pior) | 811 ms (1,24x melhor) |
> | passo do MoT | 155 ms | 221 ms (1,43x pior) | 102 ms (1,52x melhor) |
>
> **O pacote ONNX é ~2,5x mais lento que o modelo nativo**; só o TensorRT
> supera o nativo, e por ~1,4x. O trabalho de chunk do VAE (abaixo) é o que
> torna esse TensorRT construível no Jetson (125,8 GB → 3,8 GB).

## Pergunta que motivou o trabalho

O pacote de deploy do Codex (`COSMOS3EDGE_ORIN_COMPLETE_RUNTIME`) media o
grafo ONNX **1,43x mais lento que o PyTorch nativo**, na mesma precisão
FP16. A hipótese inicial era que faltava quantizar o resto do modelo.

**Essa hipótese estava errada**, e os dados do próprio Codex já mostravam:
o benchmark INT4 dele deu **1,256x mais LENTO** que BF16 na decodificação.
Quantizar sem kernels fundidos não acelera.

A hipótese alternativa testada aqui: o problema é **falta de fusão de
kernels**. O ONNX Runtime (CUDA EP) executa operação por operação; o
TensorRT funde tudo num engine otimizado. Mesma precisão, mesmo grafo.

## Resultado

| componente | ONNX Runtime (CUDA EP) | TensorRT | ganho |
|---|---|---|---|
| MoT policy transformer (av60) | ~221 ms | **102 ms** | 2,2x |
| VAE encoder (av61) | 8 679 ms | **686,8 ms** | **12,6x** |

Referência adicional: o MoT em PyTorch nativo dava 155 ms — ou seja, o
engine TensorRT ficou **mais rápido que o nativo**, enquanto o ONNX
Runtime estava mais lento que ele.

**Conclusão: a lentidão era de tooling (falta de fusão), não de precisão.
Nada foi quantizado para obter esses ganhos — é tudo FP16, o mesmo grafo.**

## O bloqueio do VAE e a causa raiz

O VAE não buildava no TensorRT: o builder abortava com
`SliceConvolutionFusion ... Cannot add shapes of unequal rank`.

Corrigir nó a nó não adiantava — o erro apenas pulava para o próximo
(`conv_in` → `conv_in_2` → `conv_in_3` → ...), porque o problema era
sistêmico:

> **O grafo foi exportado com H e W dinâmicos** (`video_height`,
> `video_width`; só batch=1, C=3, T=61 eram estáticos). Isso forçou toda a
> lógica de padding causal do VAE a virar **cálculo em runtime**
> (`Shape→Gather→Div→Cast→Reshape→...`) em vez de constante: **527 nós
> `Pad`**, 463 deles alimentando um `Conv` diretamente. O otimizador do
> TensorRT não consegue inferir o rank dessas cadeias.

### Correção aplicada

1. **Congelar a shape de entrada** em 400×720 → todas as cadeias
   `Shape→...` passam a ser constant-foldable.
2. **Constant folding do ONNX Runtime** (otimização de grafo, não executa
   nenhuma convolução) resolve os 527 paddings dinâmicos de uma vez e
   elimina os 16 nós `If`.
3. **Dobrar os 415 pares `Pad`→`Conv`** restantes no atributo `pads` do
   próprio `Conv`, removendo o nó `Pad`.

Efeito no grafo:

```
ORIGINAL:  44.366 nós  (86% aritmética de shape)  16.504 Constant, 527 Pad, 16 If
CORRIGIDO:  6.912 nós  (50%)                           0 Constant,  64 Pad,  0 If
```

**Segurança da transformação** (verificada antes de aplicar): todos os 527
nós `Pad` usam `mode="constant"`, e dos 48 que passam `const_value`
explícito, todos valem exatamente `0.0`. Logo
`Pad(constant,0) + Conv(pads=0)` ≡ `Conv(pads=P)`.

## Validação numérica

`diagnostics/validate_vae_surgery.py` compara a saída do grafo **original
intocado** contra a do grafo **pós-cirurgia**, mesma entrada, mesmo runtime:

```
erro abs máximo:  0.000000e+00
erro rel p99:     0.000000e+00
bit-exato:        True
VEREDITO: EQUIVALENTE
```

### Por que essa etapa não era opcional

Uma versão anterior do grafo **buildou no TensorRT e produziu 629,6 ms** —
mas estava **semanticamente quebrada**. Uma correção intermediária (fix dos
nós `If`, adicionando `Unsqueeze` para igualar o rank dos ramos) mudou o
rank de um tensor e quebrou um `Transpose` no `mid_block/attentions`
(`perm size: 3 does not match input rank: 4`). O ONNX Runtime recusa
executar esse grafo; o TensorRT o aceitou (com shapes estáticas ele resolve
o `If` para um ramo só e nunca exercita o caminho quebrado).

Esse fix, além de nocivo, era **desnecessário**: congelar H/W + constant
folding já elimina todos os `If` sozinho.

> **Lição:** "buildou e rodou no TensorRT" não é prova de correção. Só a
> comparação de saída contra a referência é.

## A causa raiz real: o export desenrolou o loop de streaming

Investigando por que o build pedia 125,8 GB (desproporcional para um
encoder de **300MB de pesos**, ~150M params), a resposta não é o tamanho do
modelo — é que **o export desenrolou 16 iterações de um loop de streaming**.

Em PyTorch, `AutoencoderKLWan._encode` processa o vídeo em chunks com cache
causal:

```python
iter_ = 1 + (num_frame - 1) // 4          # 61 frames -> 16 iterações
for i in range(iter_):
    if i == 0:  out = encoder(x[:, :, :1], feat_cache=self._enc_feat_map, ...)
    else:       out_ = encoder(x[:, :, 1+4*(i-1):1+4*i], feat_cache=..., ...)
```

O `torch.onnx.export` traçou através desse loop, gerando um grafo com todas
as iterações materializadas. Confirmado por contagem:

```
495 Conv no grafo = 16 iterações × ~31 convs cada
/encoder/conv_in/Conv, conv_in_1, conv_in_2, ..., conv_in_15
   (o MESMO conv, 16 vezes, compartilhando os mesmos pesos)
```

**O encoder real tem ~31 convoluções; o ONNX tem 495.** Isso explica de uma
vez os 44.366 nós, os 527 `Pad` dinâmicos (cada iteração recalcula o seu a
partir de shapes), os 125,8 GB de build e boa parte dos 686,8 ms.

E o desperdício se repete no runtime: em `cosmos3edge_inverse_onnx_10hz.py`,
a cada tick a janela desliza **1 frame** e os **61 são re-encodados**:

```python
history = images[max(0, sample_idx - 60) : sample_idx + 1]
window  = [images[0]] * left_padding + history     # 61 frames
result  = pipe(...)                                 # re-encoda tudo
```

Como o VAE é causal (o próprio docstring diz "causal 61-frame window"), o
correto é manter o `feat_cache` entre ticks e encodar só o chunk novo.

### Nota: a origem dos eixos dinâmicos

Os `dynamic_axes` de H/W que quebraram a fusão do TensorRT foram uma
decisão deliberada do export, não descuido — o comentário no script diz
*"Trace at a small spatial size to avoid retaining ~80 GB of decoder
activations"*. Ou seja, o export já contornava o mesmo problema de memória
que ressurgiu no build.

## Bloqueios em aberto (o que separa isso do Orin)

1. **O build não cabe no Orin.** Pico medido: **125,8 GB de RAM do host**
   (build de ~12min). O AGX Orin tem 64GB unificados — e engines do
   TensorRT são travados ao hardware, têm que ser buildados no alvo.

2. **Ainda longe dos 10Hz.** 686,8 ms contra orçamento de 100 ms/tick —
   ~6,9x acima. Aplicando só o ganho do VAE ao perfil do Codex
   (VAE 8,679s + Transformer 6,913s + host 0,381s ≈ 15,97s), o total cai
   para ~7,98s. É um ganho enorme, mas a meta de 10Hz continua distante.

> **Os bloqueios 1 e 2 têm a MESMA causa e a MESMA solução:** exportar uma
> iteração com o `feat_cache` como I/O explícito do grafo.
> **Implementado e medido — resultados na seção abaixo.**

## Solução: export por chunk com cache causal explícito

`diagnostics/export_wan_vae_chunk_encoder.py` exporta UMA iteração do loop:
entram 4 frames de vídeo + 24 tensores de cache; saem 1 frame latente + 24
tensores de cache. O host mantém o cache entre chamadas.

Sonda de viabilidade prévia (`diagnostics/probe_wan_vae_cache.py`):

```
slots de cache:     26  (24 tensores + 2 sempre None)
shapes:             IDÊNTICOS entre os chunks 1, 2, 3 e 15   OK
memória do cache:   339 MB (fp16)
loop vs monolítico: bit-exato (erro 0.000e+00)
```

### Resultados medidos

| | monolítico (antes) | **por chunk (agora)** | ganho |
|---|---|---|---|
| nós no grafo | 44.366 | **533** | 83x |
| Conv | 495 | **32** | 15x |
| Pad / If | 527 / 16 | **4 / 0** | — |
| **RAM p/ buildar** | **125,8 GB** | **3,74 GB** | **34x** |
| tempo de build | 12min | 3min | 4x |
| latência | 686,8 ms (61 frames) | **40,55 ms** (4 frames) | — |

**O bloqueio do Orin está resolvido:** 3,74 GB cabe com folga nos 64 GB do
AGX Orin, então o engine agora *pode* ser buildado no alvo.

Sanidade: 16 chunks × 40,55 ms = 649 ms, um pouco melhor que os 686,8 ms do
monolítico — o chunking não introduz overhead. **O ganho real vem de não
recomputar 60 frames redundantes por tick.**

### Latência por tick: duas opções, e a rápida NÃO é gratuita

O ganho grande (1 chunk por tick em vez de 16) exige **manter o `feat_cache`
entre ticks**. Mas o pipeline atual reseta o estado causal a cada tick: pega
a janela de 61 frames e chama `vae.encode()`, que começa com `clear_cache()`.
Os dois modos **não são equivalentes**, e a diferença foi medida
(`diagnostics/compare_streaming_vs_window_vae.py`, PyTorch/CUDA dos dois
lados para não haver ruído de backend):

```
tick  60:  erro 0,0        (janela coincide com o início do vídeo -- sanidade)
tick 240:  erro abs máx 2,64   corr 0,962

por posição na janela (dados de escala unitária):
  lat  0 (mais antigo): 2,64      lat  8: 0,39
  lat  3:               1,81      lat 12: 0,070
  lat  7:               1,16      lat 15 (mais recente): 0,0076
```

O erro decai com a posição, mas **não some**: os 8 latentes mais antigos
divergem entre 1,2 e 2,6. A razão é estrutural — no modo janela o latente 0
é calculado como se o vídeo começasse ali (cache zerado); no streaming ele
carrega o histórico causal real.

**Qual é o "correto"?** Provavelmente o modo janela: VAEs de vídeo são
treinados em clipes que começam do zero, então "primeiro frame sem
histórico" é a convenção de treino. Streaming fica fora dessa distribuição.

| | latência/tick | builda no Orin | muda comportamento |
|---|---|---|---|
| **A. manter janela** (16 chunks/tick) | **649 ms** | ✅ 3,74 GB | ❌ nenhuma |
| **B. streaming** (1 chunk/tick) | **~10 ms** | ✅ 3,74 GB | ⚠️ metade da janela muda |

**Recomendação: adotar A.** Ela destrava o Orin sem risco de comportamento e
ainda dá 13x sobre o baseline de 8.679 ms do ONNX Runtime. B fica como
otimização futura, condicionada a validar a *política* (as ações geradas),
não só os números do VAE.

**O ganho de memória do build (125,8 GB → 3,74 GB) vale nos DOIS casos** —
é o desbloqueio do Orin, e não depende dessa escolha.

### Shapes estáticos de propósito

Como o chunk é 16x menor, as ativações caem de 1,41 GB para 92 MB — o que
dispensa o truque de traçar em resolução reduzida com `dynamic_axes` que o
export original precisou usar. O grafo sai **totalmente estático**, que é
exatamente o que o otimizador do TensorRT precisa. **A cirurgia de grafo da
seção anterior deixa de ser necessária** (embora o `If` remanescente ainda
precise de um passe de constant folding do ORT — 1 nó, não 16).

### Validação

`export_wan_vae_chunk_encoder.py` encadeia as chamadas do grafo exportado e
compara contra o `_encode` monolítico do PyTorch:

```
erro abs máximo: 5,86e-03      erro rel p99: 3,15e-02
```

Não é bit-exato porque compara ONNX Runtime/CPU contra PyTorch/CUDA
(backends diferentes acumulam fp16 diferente). A prova de que o *chunking em
si* é exato está no teste em PyTorch puro: **erro 0.000e+00**.
*Pendente:* comparar chunked-ONNX vs monolítico-ONNX no MESMO runtime, para
isolar de vez o chunking do ruído de backend.

### Warmup de 2 chunks

Os chunks 0 e 1 têm caminho distinto do regime permanente: o chunk 0 consome
1 frame (não 4) e parte de cache vazio, deixando 22 dos 24 slots com
dimensão temporal 1 em vez de 2 (`CACHE_T=2`). O grafo é exportado com o
cache já aquecido, então **os 2 primeiros chunks rodam em PyTorch** — uma
inicialização que acontece 1x na vida do processo.

## Opção A implementada e entregue

Números finais medidos (A100, engines FP16, `builderOptimizationLevel=1`):

| grafo | nós (pós-fold) | RAM p/ buildar | latência |
|---|---|---|---|
| `chunk0` (1 frame, sem cache) | 448 | 3,26 GB | 19,12 ms |
| `chunk1` (4 frames, cache T=1) | 533 | 3,42 GB | 40,63 ms |
| `chunkN` (4 frames, cache T=2) | 533 | 3,70 GB | 41,51 ms ×14 |
| **total por tick** | | **máx 3,70 GB** | **640,9 ms** |

**13,5x** sobre os 8.679 ms do caminho ONNX Runtime atual, **sem nenhuma
mudança de comportamento**, e construindo com **3,70 GB** em vez de 125,8 GB
— ou seja, o engine finalmente pode ser construído no próprio Orin.

### Por que três grafos

Como o cache é resetado a cada tick (semântica de janela), todo tick passa
pelos chunks "frios", cujos shapes de cache diferem do permanente: o chunk 0
consome 1 frame e não recebe cache; o chunk 1 recebe cache com `T=1`; do 2 em
diante, `T=2` (`CACHE_T=2`). Exportar os três evita ter que carregar o VAE em
PyTorch no Orin só para os dois primeiros — o deploy fica ONNX/TensorRT puro.

### Arquivos entregues

| caminho | o que faz |
|---|---|
| `runtime/cosmos3edge_vae_chunked.py` | `ChunkedWanEncoder` — substituto direto do `OrtWanEncoder`, mesma assinatura `encode()` |
| `runtime/fold_vae_chunk_graph.py` | remove o nó `If` e dobra `Pad`→`Conv` (obrigatório antes do TensorRT) |
| `deploy/build_vae_chunk_on_jetson.sh` | constrói os 3 engines no próprio Orin |
| `onnx/encoder-chunk{0,1,N}-fp16.onnx` | os grafos exportados |
| `diagnostics/export_wan_vae_chunk_all.py` | export + validação da cadeia completa |
| `diagnostics/probe_wan_vae_cache.py` | sonda a estrutura do `feat_cache` |
| `diagnostics/compare_streaming_vs_window_vae.py` | mede janela vs streaming |

A seleção é automática em `build_vae_encoder()` (em
`runtime/cosmos3edge_inverse_onnx.py`): se o caminho do `--vae-encoder` for um
diretório com `encoder-chunk0-fp16.onnx`, usa o modo por chunk; senão mantém o
monolítico. Forçável com `COSMOS3EDGE_VAE_CHUNKED=1|0`. Já ligado nos três
pontos de entrada (`cosmos3edge_inverse_onnx.py`,
`cosmos3edge_inverse_onnx_10hz.py`, `cosmos3edge_policy_onnx_10hz.py`).

### Ainda não feito

- **Nada rodou em Jetson real.** Tudo aqui é A100 x86_64.
- O `ChunkedWanEncoder` passa por CPU (`.cpu().numpy()`) entre chunks, onde o
  original usava `io_binding`. Deve ser ruído frente aos 640 ms, mas é
  otimização óbvia (bind direto na GPU).
- **Decoder do VAE**: tem o mesmo defeito de loop desenrolado e ainda não foi
  tratado. Necessário para as modalidades de geração; provavelmente não para
  inverse dynamics / policy.

## O novo gargalo: o transformer

Com o VAE em ~10 ms/tick, o perfil do Codex (VAE 8,679 s + Transformer
6,913 s + host 0,381 s) passa a ser dominado pelo transformer, que roda 30
passos de difusão por tick. Mesmo aplicando os 2,2x de fusão medidos no MoT,
seriam ~3,1 s — ainda ~31x acima do orçamento de 100 ms. **A meta de 10 Hz
não está ganha; o gargalo mudou de dono.**

3. **Engine especializado em 400×720** (consequência de congelar a shape).
   Normal para TensorRT, e compatível com o teste Boston 10Hz, mas fecha a
   porta para resolução variável.

4. **Cobertura parcial:** só 2 dos 14 grafos ONNX foram medidos. Faltam 5
   variantes do MoT (inverse-av61, forward-umi16, policy-umi16,
   i2v-121-cond/uncond) e 5 do VAE (decoder-av61, encoder/decoder-umi17,
   encoder/decoder-i2v121).

5. **Oportunidade não explorada:** o log acusou
   `Throughput may be bound by Enqueue Time` — o tempo de enfileiramento
   empata com o de computação, indicando overhead de lançamento de kernel.
   `--useCudaGraph` deve render mais.

6. **Nada foi testado em hardware Jetson real.** Tudo aqui é A100.

## Artefatos

| caminho | o que é |
|---|---|
| `models/cosmos3edge-vae-encoder-av61-clean/encoder-av61-fp16-clean.onnx` | grafo do VAE corrigido e **validado bit-exato** |
| `models/trt-engines-bench/vae_encoder_av61_CLEAN.engine` | engine TensorRT do VAE (A100 40GB) |
| `models/trt-engines-bench/mot_policy_av60.engine` | engine TensorRT do MoT policy |
| `diagnostics/fix_vae_freeze_and_fold.py` | a correção (congelar + constant fold + dobrar Pad→Conv) |
| `diagnostics/validate_vae_surgery.py` | validação numérica original vs corrigido |
| `jobs/trtexec_vae_clean_build.sbatch` | build + medição de pico de RAM |
| `jobs/fix_vae_clean_and_validate.sbatch` | cirurgia + validação em um job |

### Correções descartadas (não reutilizar)

- `diagnostics/fix_vae_if_rank_mismatch.py` — **quebra o modelo** (muda
  rank, quebra `Transpose` no mid_block). Desnecessária.
- `diagnostics/fix_vae_dynamic_pad.py`, `fix_vae_pad_into_conv.py`,
  `fix_vae_all_dynamic_pads*.py` — tentativas pontuais/parciais,
  substituídas por `fix_vae_freeze_and_fold.py`.

## Próximos passos sugeridos

1. Integrar os engines nos scripts de runtime do Codex (`make_session()` já
   suporta `COSMOS3EDGE_ORT_TENSORRT=1`) e refazer o Boston 10Hz ponta a
   ponta, para medir o ganho real de pipeline (não só por componente).
2. Atacar o bloqueio de memória de build antes de qualquer coisa no Orin —
   sem isso, o engine não existe no alvo.
3. Medir os 10 grafos restantes, se o padrão se confirmar.
4. Testar `--useCudaGraph`.
