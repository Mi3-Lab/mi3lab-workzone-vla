# Cosmos3-Edge: runtime completo de pesquisa para Jetson AGX Orin

O ZIP operacional contém os ONNX, os pesos externos desses ONNX, runtimes e
testes necessários para executar as modalidades sem baixar o checkpoint BF16.
O segundo ZIP adiciona checkpoints de referência para comparação/reexportação.
Na árvore local de montagem, os dois conjuntos aparecem juntos. A precisão é
mista e está identificada por componente; somente o reasoner é INT4 W4A16.

Para transferência, a entrega é separada em dois ZIPs: o runtime operacional
contém `onnx`, `pipeline-config`, `runtime`, `deploy`, mídia e testes; os pesos
BF16/TorchAO/AWQ de referência ficam em um arquivo adicional. O runtime ONNX
não carrega o checkpoint BF16 antes da inferência.

- `COSMOS3EDGE_ORIN_OPERATIONAL_AV_FUTURE_SE3_FINAL.zip`: runtime operacional corrigido para o Orin;
- `COSMOS3EDGE_REFERENCE_WEIGHTS_BF16_INT4.zip`: referências opcionais.

## Conteúdo

- `onnx/reasoner-int4`: reasoner W4A16 AWQ e KV-cache FP16;
- `onnx/visual-fp16`: SigLIP2/projector FP16;
- `onnx/mot-fp16-shared`: inverse AV, policy AV, forward, policy UMI e dois ramos I2V, todos
  referenciando um conjunto compartilhado de pesos;
- `onnx/vae-fp16-shared`: encoders/decoders AV, UMI e I2V com pesos
  compartilhados;
- `pipeline-config`: tokenizer, scheduler e configurações sem pesos BF16
  duplicados no carregamento ONNX;
- `weights`: presente somente no ZIP adicional; referências BF16, TorchAO INT4
  e AWQ para comparação/reexportação;
- `runtime`: pipelines completos e orquestrador temporal a 10 Hz;
- `deploy`: build do reasoner/visual e teste Boston completo no Orin;
- `tests/boston_10hz`: 151 timestamps de 0,0 a 15,0 s;
- `third_party/tensorrt_edge_llm_patch`: alterações necessárias no Edge-LLM.
- `third_party/diffusers-cosmos3`: fonte Diffusers fixada no commit testado,
  usada somente como orquestrador host; os pesos pesados executam nos ONNX.

A decomposição exata do tamanho está em `docs/CONTEUDO_E_TAMANHO.md`.
O inventário dos 14 grafos está em `docs/ONNX_INVENTORY.md`.
As versões testadas estão em `docs/SOFTWARE_VERSIONS.md`.

## Preparação do software no Orin

Instale PyTorch e ONNX Runtime GPU para a versão do JetPack do dispositivo; o
ONNX Runtime precisa listar `TensorrtExecutionProvider` e
`CUDAExecutionProvider`. O host também precisa de `ffmpeg`, `rsync` e das
dependências de compilação oficiais do Edge-LLM. Depois instale
`requirements-runtime.txt` e aplique o patch sobre um checkout TensorRT
Edge-LLM v0.9.0:

```bash
python3 -m pip install -r requirements-runtime.txt
./deploy/apply_edgellm_cosmos3edge_patch.sh "$PWD" /caminho/TensorRT-Edge-LLM
```

`requirements-reference-benchmark.txt` é opcional e só serve para comparar o
estado TorchAO INT4 do segundo ZIP com a execução nativa BF16. Esse benchmark
opcional (`runtime/benchmark_cosmos3edge_reasoner_10hz.py`) usa um checkout do
Cosmos Framework com `COSMOS_TRAINING=0`; ele não faz parte do caminho ONNX
operacional.

Recompile o Edge-LLM no próprio Orin. Engines serializados em A100 não são
portáveis e, por isso, não estão no pacote.

Após extrair, valide os hashes antes de construir engines:

```bash
./deploy/verify_package.sh "$PWD" operational
```

## Teste obrigatório no Orin

Depois de compilar o TensorRT Edge-LLM com o patch desta pasta e instalar o
ONNX Runtime com TensorRT Execution Provider:

```bash
./deploy/run_boston_complete_10hz_on_jetson.sh \
  "$PWD" /caminho/para/TensorRT-Edge-LLM
```

Por padrão, o teste produz uma resposta multimodal e um **rollout AV
experimental de 60 passos/6 segundos** independente para cada tick de 100 ms.
Ele usa a cabeça `policy` do modelo-base Cosmos3-Edge, mas não deve ser tratado
como uma VLA pós-treinada ou uma policy segura para controlar o veículo. Se a chamada ultrapassar 100 ms, a
execução ocorre offline/sequencial e registra a falha de deadline. Inverse AV
continua disponível com `COSMOS3EDGE_TRAJECTORY_MODE=inverse`; ele reconstrói o
movimento observado e não deve ser rotulado como previsão futura. Cada atualização guarda timestamp, texto exato,
`action_60x9`, latência e cumprimento ou não do deadline de 100 ms. O renderer
integra as poses relativas por composição SE(3), convenção
`backward_framewise`, e não interpola nem fabrica inferências ausentes.

Importante: uma versão anterior do renderer somava somente XYZ com `cumsum` e
ignorava as rotações 6D. Isso estava matematicamente errado e foi substituído
por `runtime/cosmos3edge_av_pose.py`, conferido contra o `pose_rel_to_abs`
oficial com erro absoluto máximo de `3,50e-6`.

Para testar também forward dynamics, policy conjunta e image-to-video usando
somente o pacote operacional:

```bash
./deploy/run_all_generation_modalities_on_jetson.sh "$PWD"
```

## Limite físico

“Executar em cadência de 10 Hz” não significa automaticamente “terminar cada
execução em menos de 100 ms”. O JSON registra ambas as coisas separadamente. O
resultado A100 e a medição física do Orin determinam se há tempo real.
