# Auditoria de pesos, ONNX e inferência a 10 Hz

Data: 2026-07-22

## Correção objetiva

O nome anterior “kit de implementação completa” induz ao erro. O ZIP de 17 MB
é um kit de código, relatórios e resultados; ele **não contém pesos** e não roda
sozinho. A quantização INT4 e a exportação ONNX foram concluídas para o
**reasoner**, não para todo o Cosmos3-Edge. Os componentes de dinâmica, VAE e
visão exportados posteriormente estão em FP16.

Também são conceitos diferentes:

- uma trajetória `60x9` representa 60 pontos espaçados a 10 Hz, produzidos por
  uma única execução de inverse dynamics sobre uma janela de 61 frames;
- inferência online a 10 Hz significa iniciar uma nova execução a cada 100 ms.

Os testes anteriores comprovaram o primeiro caso. Eles não comprovaram uma nova
inferência completa do Cosmos3-Edge a cada 100 ms.

## Pesos realmente existentes

| Artefato | Caminho | Tamanho aproximado | Precisão e escopo |
|---|---|---:|---|
| Checkpoint oficial completo | `COSMOS3EDGE_ORIN_FULL_PACKAGE/weights/bf16-full` | 8,6 GB | BF16; transformer, VAE, vision encoder, tokenizers e configurações |
| Language model TorchAO | `models/cosmos3edge-lm-int4-torchao/lm_int4_state_dict.pt` | 4,25 GB | INT4 weight-only, grupo 128; somente a torre de entendimento, exclui `*_moe_gen`; não é ONNX |
| Reasoner AWQ | `models/cosmos3edge-reasoner-int4awq/model.safetensors` | 1,82 GB | W4A16 AWQ, grupo 128; somente reasoner, `lm_head` excluído |
| Reasoner ONNX | `models/cosmos3edge-reasoner-int4awq-onnx/llm` | 1,7 GB | ONNX/TensorRT Edge-LLM; somente reasoner |
| Visual ONNX | `models/cosmos3edge-full-onnx/visual/model.onnx` | 981 MB | FP16; perfil estático de uma imagem Boston |
| MoT inverse/forward/policy | `models/cosmos3edge-complete-onnx/transformer` | 5,82 GiB de pesos únicos quando deduplicados | FP16; não INT4 |
| VAE encoder/decoder | `models/cosmos3edge-complete-onnx/vae` | depende do perfil; encoder AV 294 MB e decoder AV 1,1 GB | FP16; não INT4 |

Os arquivos duplicados dentro de `COSMOS3EDGE_ORIN_FULL_PACKAGE/weights` têm os
mesmos SHA-256 dos originais de trabalho:

- AWQ: `4fbb4a9fe1dbf6dac32f42f1529f4b223ba5040b4c4192ca73c1fbbfc4c368d6`;
- TorchAO: `30f396f2f038c5965d602bb39668a75553fe6a161c33246c13f3915f04be92b0`.

## O que foi confirmado diretamente nos grafos

O reasoner ONNX recebe `inputs_embeds` FP16, 28 pares de KV-cache FP16 e mRoPE;
ele não recebe pixels ou vídeo. Seu grafo contém 168 nós
`Int4GroupwiseGemmPlugin`. O arquivo de quantização declara
`W4A16_AWQ`, grupo 128, sem zero point e com `lm_head` excluído. Portanto este é
um ONNX INT4 real, mas apenas da torre textual.

O visual ONNX recebe patches `[1196,768]` FP16 e devolve 299 embeddings de
largura 2048. O inverse ONNX recebe latentes visuais FP16, ação ruidosa FP16 e
devolve `action_velocity [60,64]` FP16. Forward, policy e VAE também expõem
entradas/saídas FP16. Esses grafos não podem ser chamados de INT4.

## Pesos corretos para cada teste

### Referência BF16 completa

Usar `COSMOS3EDGE_ORIN_FULL_PACKAGE/weights/bf16-full`. Esse é o único
checkpoint local autossuficiente para a referência nativa completa.

### Comparação nativa INT4

Carregar o checkpoint BF16 completo para os componentes não quantizados e
substituir somente a language model pelo estado TorchAO
`models/cosmos3edge-lm-int4-torchao/lm_int4_state_dict.pt`. Esta execução é
precisão mista e PyTorch; não é ONNX INT4 completo.

### Reasoner ONNX INT4 multimodal

Usar em conjunto:

1. `models/cosmos3edge-full-onnx/visual/model.onnx`;
2. `models/cosmos3edge-reasoner-int4awq-onnx/llm/model.onnx` e seu external data;
3. `embedding.safetensors`, tokenizer, processor, montagem dos tokens visuais,
   mRoPE e KV-cache do runtime TensorRT Edge-LLM corrigido.

O perfil visual atual suporta uma imagem redimensionada para o grid fixo usado
no teste Boston. Ele não é ainda um perfil genérico de vídeo multiframes.

### Inverse dynamics ONNX

Usar em conjunto:

1. VAE encoder AV FP16;
2. transformer inverse AV FP16;
3. scheduler UniPC e preparação de entrada no host;
4. pós-processamento dos primeiros 9 canais para a trajetória `60x9`.

Esse caminho foi estendido com uma janela causal deslizante de 61 frames e uma
nova chamada para cada um dos 151 ticks do Boston, de 0,0 a 15,0 s. O teste é
offline: depois do carregamento, cada chamada leva aproximadamente 15 s no
A100 e nenhuma cumpre o deadline de 100 ms. Isso comprova a execução por tick,
mas também comprova que o backend atual não é tempo real a 10 Hz. O Orin ainda
precisa ser medido fisicamente.

## Estado honesto da entrega

- Existe reasoner ONNX INT4 W4A16 funcional.
- Existem visual, cinco interfaces MoT e seis interfaces VAE em ONNX FP16.
- Existe checkpoint BF16 completo e um estado TorchAO INT4 parcial.
- Não existe hoje um Cosmos3-Edge completo, todo INT4, em ONNX.
- O ZIP antigo de 17 MB não contém pesos e foi substituído.
- O ZIP operacional corrigido contém aproximadamente 10 GiB: todos os ONNX,
  pesos externos deduplicados, configuração e runtime. BF16/TorchAO/AWQ ficam
  em um segundo ZIP opcional de aproximadamente 16,86 GiB.
- No A100, o Reasoner executou 151/151 imagens, uma por tick de 100 ms da linha
  do tempo; inverse AV executa outra chamada independente para cada tick.
- A execução é sequencial/offline porque não termina em 100 ms. Não existe
  ainda evidência física de tempo real no Jetson AGX Orin.
