# Cosmos3-Edge completo no Jetson AGX Orin — arquitetura corrigida

Data: 2026-07-21

## Conclusão da auditoria

O pacote ONNX anterior não é um runtime completo. Ele capturou componentes
isolados, mas não preservou o pipeline que transforma mídia comum em saídas do
Cosmos3-Edge. Em particular, faltavam o VAE encoder, o loop UniPC/CFG, a
integração do encoder visual com o prefill multimodal e os perfis necessários
para as diferentes modalidades.

Também é incorreto exigir que todo componente seja INT4. ONNX é o formato do
grafo, não a precisão. No TensorRT, INT4 é weight-only e aplicável a GEMM; ele
não substitui automaticamente convoluções, normalizações, softmax, VAE e
pré/pós-processamento. A implementação correta no Orin será de precisão mista.

### Evidência executável obtida nesta auditoria

- O pipeline oficial Diffusers fixado executou inverse dynamics completo no
  exemplo AV oficial: `60x9`, 30 passos, 8,08 s no A100. Contra o JSON publicado
  no checkpoint obteve MAE `0,02454`, erro máximo `0,16016` e cosseno
  `0,99682`. A pequena diferença é compatível com mudança de backend/precisão,
  mas será acompanhada valor a valor.
- No `boston.mp4`, a referência Diffusers BF16 completou em 12,55 s no A100,
  pico de 10,15 GB de CUDA e trajetória finita `60x9`.
- O encoder Wan VAE FP16 novo foi carregado e realmente executado pelo ONNX
  Runtime CUDA: 9,48 s sem TensorRT, MAE `0,00317` e erro máximo `0,04860`
  contra o tensor Diffusers BF16.
- Os antigos encoder e denoiser BF16 foram rejeitados pelo ONNX Runtime antes
  da inferência (`Conv` BF16 e `Einsum` BF16 inválidos nesse backend). Logo,
  `onnx.checker=pass` não demonstrava que esses grafos eram utilizáveis.
- O inverse dynamics ONNX completo executou o `boston.mp4` em 22,48 s no A100:
  MP4, preprocessing oficial, VAE encoder ONNX, 30 passos UniPC chamando o MoT
  ONNX e saída `60x9`. Contra o pipeline BF16 obteve MAE `0,00161`, erro máximo
  `0,01025` e cosseno `0,999990` sobre os 540 valores.
- O Wan decoder FP16 ONNX também executou: entrada `[1,48,16,25,45]`, saída
  finita `[1,3,61,400,720]`, 13,48 s no A100 sem TensorRT.
- O forward UMI ONNX completo executou os dois chunks autoregressivos oficiais:
  frame inicial + 16 ações por chunk, seeds 0/1, 17 frames por chamada, descarte
  dos frames condicionantes e vídeo final de 32 frames, 256x256 a 20 FPS. Foram
  `8,16 s` de inferência no A100 (`6,37 s` no primeiro chunk e `1,78 s` no
  segundo, com sessões já carregadas). Contra o vídeo Edge publicado: MAE RGB
  `11,00/255`, PSNR médio `23,03 dB` e SSIM global médio `0,9576`. A divergência
  cresce autoregressivamente, mas conteúdo, geometria, duração e movimento são
  preservados; a comparação por frame está registrada em JSON.
- A exportação forward isolada também foi comparada diretamente com BF16 em um
  passo: cosseno `0,999951`, MAE `0,00872` e erro máximo `0,07715`. Isso localiza
  a maior diferença visual acumulada no rollout/codec de precisão reduzida, não
  em uma quebra estrutural do transformer.
- O modo policy conjunto foi exportado como um único ONNX com duas saídas. No
  passo isolado, visão FP16 contra BF16 obteve cosseno `0,999955`, MAE `0,00809`;
  ação obteve cosseno `0,999988`, MAE `0,00131`. O pipeline ONNX completo gerou
  vídeo de 17 frames e ação UMI `[16,10]` em `6,56 s` no A100. Os vetores inicial
  e final contra o policy BF16 tiveram MAE `0,00295` e `0,00255`, respectivamente.
- A deduplicação real dos três grafos MoT foi validada pelo `onnx.checker`: os
  pesos externos lógicos somavam `17,36 GiB`; o armazenamento compartilhado por
  SHA-256 contém `5,82 GiB`, economizando `11,55 GiB` (razão `2,98x`). Os três
  grafos permanecem separados por interface, mas apontam para os mesmos 475
  blobs únicos dentro da árvore ONNX.
- A referência image-to-video oficial (121 frames, 480x832, 24 FPS, 50 passos,
  CFG 5) executou em `54,26 s` no A100 e produziu latente finito
  `[1,48,31,30,52]`. O VAE encoder FP16 desse perfil foi exportado e validado.
  Como condicional e incondicional têm pacotes estáticos diferentes, foram
  exportados dois grafos de interface que compartilharão os mesmos pesos: o
  ramo condicional obteve cosseno `0,999958`, MAE `0,00905`; o incondicional,
  cosseno `0,999958`, MAE `0,00900` contra BF16. O VAE decoder nativo do perfil
  também produziu RGB finito `[1,3,121,480,832]`. O ONNX Runtime completou o
  pipeline inteiro e gravou 121 frames em `179,24 s`, depois de `264,91 s` de
  carregamento no A100. Contra o vídeo Edge publicado obteve PSNR médio
  `15,15 dB` e SSIM médio `0,7022`: o primeiro frame é muito próximo
  (`28,27 dB`, SSIM `0,9866`), mas o erro FP16 se acumula nas 100 chamadas CFG.
  Portanto a implementação estrutural passou, mas este perfil FP16 ainda não
  passou o critério de paridade temporal para produção.
- O reasoner INT4 W4A16 foi integrado de verdade ao visual TensorRT. Foi
  acrescentado ao TensorRT Edge-LLM o tipo `cosmos3_edge_vision`, preprocessing
  SigLIP2 na ordem raster correta, inserção dos 299 embeddings, mRoPE não
  intercalado `[24,20,20]` e suporte aos bindings do ONNX estático. No frame
  Boston, o patch tensor contra o processor oficial teve cosseno `0,999960` e
  os embeddings TensorRT contra BF16, `0,997344`. A inferência gerou uma
  descrição coerente da rua e reconheceu `Road Work Ahead`, com encoder visual
  `60,61 ms`, prefill de 339 tokens `24,07 ms`, decode `2,18 ms/token` para 90
  tokens e pico de `3,37 GB` no A100. Este teste cobre uma imagem no perfil
  fixo 416x736; ainda não valida packing de vários frames em vídeo.

### Causa do ZIP excessivo

O diretório anterior continha 8,6 GB do checkpoint BF16 original, 1,8 GB do
reasoner INT4 e duas exportações do mesmo MoT, 6,4 GB para inverse e mais 6,4 GB
para forward. Somados ao visual, VAE, runtime-fonte, fixtures e relatórios, o
diretório chegou a 28 GB e o ZIP a 11 GB. Essa duplicação não faz parte da
arquitetura do modelo. Na entrega corrigida, o ZIP operacional não inclui o
checkpoint Hugging Face integral e os grafos de modo referenciam pesos ONNX
externos compartilhados. BF16, TorchAO INT4 e AWQ ficam em um segundo ZIP
opcional de referência, atendendo ao pedido de preservar esses pesos sem
inflar o arquivo necessário para executar no Orin.

## Pipeline oficial que deve ser preservado

### Reasoner multimodal

1. Decodificar imagem/vídeo e selecionar frames.
2. Executar o processor Cosmos3-Edge: resize, patchify, grid THW e tokens
   especiais de visão.
3. Executar SigLIP2 + projector para produzir embeddings de largura 2048.
4. Inserir esses embeddings nas posições dos tokens visuais.
5. Construir mRoPE multimodal.
6. Executar prefill e depois decode autoregressivo com KV-cache.
7. Decodificar os tokens de texto preservando o texto exato.

O ONNX W4A16 existente do reasoner possui as interfaces corretas para
`inputs_embeds`, 28 pares de KV, cache mRoPE e logits. O visual ONNX atual fixa
o grid Boston `[1,26,46]`. A integração implementada estende a infraestrutura
Qwen do Edge-LLM somente nas partes compartilháveis e trata separadamente o
SigLIP2: bindings `pixel_values`/`visual_embeddings`, patch layout raster,
ausência de rotary visual/cu-seqlens no grafo, expansão do placeholder para IDs
sintéticos e mRoPE Cosmos `[24,20,20]`. Isso já executa imagem + INT4 reasoner
ponta a ponta. O próximo incremento necessário para vídeo é exportar profiles
com múltiplos frames/grades e construir o reasoner com `max_input_len` suficiente;
o engine estático de uma imagem não deve ser apresentado como reasoner de vídeo.

### Inverse dynamics AV

1. Ler `chunk_size + 1` frames; para AV são 61 frames a 10 FPS.
2. Redimensionar/padronizar para um canvas de action resolution tier.
3. Executar o Wan VAE encoder: RGB `[1,3,61,H,W]` para latentes normalizados
   `[1,48,16,H/16,W/16]`.
4. Construir o prompt JSON de ação, IDs, position IDs, sequence indexes,
   condition masks e domínio AV (`domain_id=1`, `raw_action_dim=9`).
5. Inicializar ruído determinístico `[60,64]` com a seed solicitada.
6. Executar 30 passos UniPC, shift 10, guidance 1.0. Cada passo chama o
   transformer MoT e atualiza os latentes de ação.
7. Manter zerados os canais 9:64 e devolver os primeiros 9 canais. Para AV não
   há um decoder neural adicional: o pós-processamento é slice e, quando
   configurado, desnormalização.

### Forward dynamics e image-to-video

1. Preparar frame inicial e, em forward dynamics, ações condicionantes.
2. Executar VAE encoder no prefixo condicionado.
3. Inicializar latentes de vídeo com ruído nas posições não condicionadas.
4. Executar o loop UniPC/CFG chamando o transformer a cada passo.
5. Desnormalizar e executar o Wan VAE decoder.
6. Remover padding, converter `[-1,1]` para RGB e codificar o vídeo.

### Policy

O modo policy gera simultaneamente latentes de vídeo e ação. O engine precisa
retornar as duas velocidades no mesmo passo; dois engines independentes não
preservam o acoplamento do MoT.

## Fronteiras ONNX/TensorRT corretas

O scheduler e a lógica de pipeline permanecem no host Python/C++; eles não têm
pesos e não justificam colocá-los dentro de um único ONNX gigante.

| Engine | Entrada principal | Saída | Precisão alvo no Orin |
|---|---|---|---|
| Visual reasoner | patches + grid/profile | embeddings 2048 | INT8 Q/DQ, fallback FP16 |
| Reasoner | embeddings + mRoPE + KV | logits + KV | W4A16 INT4 Edge-LLM |
| Wan VAE encoder | RGB normalizado | latentes 48 canais | INT8/FP16 misto |
| MoT inverse | visão + ação ruidosa + timestep | velocidade de ação | INT8 Q/DQ; INT4 WoQ experimental nos GEMMs |
| MoT forward/I2V | visão ruidosa + ação + timestep | velocidade de visão | INT8 Q/DQ; INT4 WoQ experimental nos GEMMs |
| MoT policy | visão e ação ruidosas + timestep | duas velocidades | INT8 Q/DQ |
| Wan VAE decoder | latentes normalizados | RGB | INT8/FP16 misto |

Os ONNX de cada modo devem referenciar um único blob externo de pesos do MoT
quando possível. Duplicar o blob em cada grafo recria o problema de tamanho do
pacote. No dispositivo, constrói-se apenas o conjunto de engines usado pelo
perfil selecionado, ou remove-se o engine temporário após a medição.

## Quantização correta

- Reasoner: manter o W4A16 AWQ já validado, pois o TensorRT Edge-LLM possui
  plugins próprios e declara INT4 no Orin.
- Transformer de geração/ação: calibrar com vídeos, timesteps e latentes reais,
  não apenas texto. O alvo seguro inicial é INT8 Q/DQ, preservando
  normalizações, softmax, embeddings, projeções pequenas e saídas em BF16/FP16.
- Visual: INT8 Q/DQ com calibração de frames roadwork; medir erro dos embeddings
  e concordância das respostas completas.
- VAE: INT8 apenas onde o TensorRT selecionar kernels válidos; manter camadas
  sensíveis em FP16. INT4 genérico não cobre convoluções do VAE.
- INT4 do MoT: experimentar somente MatMul/GEMM com blocos 64/128 e comparar
  engine, latência e trajetória. Se o TensorRT do JetPack não selecionar kernel
  INT4 no Orin, manter INT8; um arquivo menor que dequantiza lentamente não é
  uma otimização válida.

## Perfis mínimos da primeira entrega

1. Reasoner texto, imagem e vídeo com KV-cache real.
2. Inverse dynamics AV: 61 frames, 10 FPS, 480 tier, ação 60x9.
3. Forward dynamics UMI: 16 ações e 17 frames, conforme o exemplo oficial.
4. Image-to-video no perfil oficial Edge.
5. Policy com saída conjunta de vídeo e ação.

Depois da equivalência nesses perfis, os grids/aspect ratios e comprimentos são
ampliados por optimization profiles. Um perfil Boston estático não deve ser
apresentado como modelo genérico.

## Critérios de aceite

- Entrada deve ser JPEG/PNG/MP4, nunca fixture latente obrigatória.
- Mesmo input e seed devem repetir exatamente a saída dentro do backend.
- Reasoner deve preservar texto completo, EOS, TTFT e ms/token.
- Inverse dynamics deve produzir `60x9` finito e comparar todos os 540 valores
  com a referência BF16.
- Forward/I2V/policy devem produzir vídeo decodificado, não somente velocity.
- Relatar erro por etapa: VAE encode, velocity por timestep, scheduler, ação
  final e VAE decode.
- Medir cold build/load, warm p50/p95, RAM compartilhada, potência e temperatura
  no Orin real.
- Não aprovar quantização apenas com `onnx.checker`; executar o pipeline inteiro.

## Versões fixadas para reprodução

- Cosmos Framework: cópia presente neste repositório.
- Checkpoint: `nvidia/Cosmos3-Edge` local.
- Diffusers Edge correto: commit
  `86e6dac5360703ddf09fe250db50be667eb93662` (2026-07-20).
- O Diffusers 0.39 instalado anteriormente é incompatível: interpretou o Edge
  como Qwen/gated e criou parâmetros aleatórios. Ele não pode ser usado como
  referência.

## Fontes primárias

- Modelo e exemplos oficiais: https://huggingface.co/nvidia/Cosmos3-Edge
- Pipeline Cosmos3 Diffusers: https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3
- Implementação oficial Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
- Transformer MoT oficial: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos3.py
- TensorRT INT4/INT8 e Q/DQ: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html
- ModelOpt ONNX INT4: https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.quantization.int4.html
- TensorRT Edge-LLM no Orin: https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html

## Limite de hardware

A NVIDIA publica Generator Edge no H100/H20/RTX PRO/DGX e image-to-video em
Jetson Thor, mas não publica números do Generator completo no AGX Orin. Portanto
inverse/forward/I2V/policy no Orin continuam experimentais até medição no
dispositivo. O reasoner possui um caminho oficial de edge deployment mais
maduro. Isso não impede a implementação, mas impede prometer latência antes do
teste físico.
