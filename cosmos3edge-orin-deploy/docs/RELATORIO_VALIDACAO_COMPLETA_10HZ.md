# Cosmos3-Edge completo: auditoria final e Boston a 10 Hz

## Resultado verificável

- Vídeo: `/data/wesleyferreiramaia/wokzone-alpamayo/videos/boston.mp4`, SHA-256 `b7f4b75d7a6d009c616298c009fc158bfdce11d63aa61e53cef69a774d056313`.
- Fonte: `451` frames, `30.0` FPS,
  `15.033` s.
- Linha do tempo: `151` inferências independentes, timestamps de `0,0` a
  `15,0` s, exatamente uma entrada por tick de 100 ms. A execução é offline e
  sequencial porque as chamadas excedem o deadline.
- Reasoner: `151/151` textos preservados.
- Finalização do Reasoner: `{'end-of-sequence': 150, 'max-length': 1}`.
- AV inverse: `151/151` saídas `60x9`, todas finitas: `True`.
- AV policy-mode: `151/151` saídas `60x9`, todas finitas, com horizonte
  nominal de `60` transições/`6,0 s` a partir de cada frame atual. É um rollout
  experimental do modelo-base, não uma policy AV pós-treinada/validada.
- Nenhuma inferência ausente, interpolada ou fabricada.

## Reasoner ONNX INT4 + visual FP16 no A100

| Métrica | Valor |
|---|---:|
| Vision encoder médio | 9.640 ms |
| Prefill médio | 21.580 ms |
| Decode | 2.1535 ms/token |
| Tokens gerados | 7882 |
| Pipeline GPU médio estimado | 143.630 ms/chamada |
| Parede média incluindo overhead/carga | 232.367 ms/chamada |
| Pico GPU | 3374 MB |
| Deadline de 100 ms | **FAIL** |

## AV inverse ONNX FP16 no A100

| Métrica | Valor |
|---|---:|
| Latência média | 14.663 s |
| p50 | 15.138 s |
| p95 | 15.267 s |
| máxima | 19.578 s |
| Deadlines de 100 ms cumpridos | 0/151 |

Cada chamada usa uma janela causal de 61 frames a 10 FPS. Antes de 6 s, o
histórico ausente é preenchido com o primeiro frame e isso é registrado por
timestamp. Esta saída é **inverse dynamics**, isto é, reconstrução do movimento
AV observado; não é uma previsão de trajetória futura.

## Rollout AV experimental ONNX FP16 no A100

| Métrica | Valor |
|---|---:|
| Atualizações independentes | 151/151 |
| Ações por atualização | `60x9` |
| Horizonte futuro | 6,0 s |
| Latência média | 11,347 s |
| p50 | 11,003 s |
| p95 | 11,864 s |
| máxima | 15,702 s |
| Deadlines de 100 ms cumpridos | 0/151 |

Esta é a saída `policy` AV conjunta do modelo-base Cosmos3-Edge: a partir do
frame RGB atual, ela faz rollout de vídeo/ação e devolve 60 poses 9D. Cada pose
contém translação `[0:3]` e rotação Zhou 6D `[3:9]`. A NVIDIA apresenta o Edge
como inicialização para downstream e publica Policy DROID como benchmark de
policy; para AV, o exemplo oficial documentado é inverse dynamics. Portanto,
esta execução é uma sondagem experimental da cabeça policy, não evidência de
uma VLA AV pós-treinada ou segura para controle.

### Correção da integração geométrica

A implementação anterior desenhava `cumsum(action[:, :3])`. Isso era um erro:
ignorava as rotações e não respeitava a convenção das poses relativas. O runtime
agora converte rot6d para SO(3) e compõe transformações SE(3) na convenção
`backward_framewise`, igual ao `pose_rel_to_abs` oficial. A comparação numérica
das 151 saídas apresentou erro absoluto máximo de `3,5003e-6`.

Mesmo após essa correção, a série continua instável entre ticks: a distância
entre endpoints consecutivos tem média `1,6399`, p95 `5,6431` e máximo
`12,6343` nas unidades nativas. `149/151` endpoints têm eixo frontal positivo;
dois são negativos. Não há calibração que permita chamar as unidades de metros,
nem ground truth/GPS neste teste para calcular erro de trajetória.

O vídeo `tests/boston_10hz/boston_cosmos3edge_policy_future_10hz.mp4`
preserva os 451 frames/15,033 s e troca a trajetória a cada tick de 100 ms.
`POLICY_FUTURE_ACCEPTANCE_PASS.json` verifica timestamps, `151 x [60,9]`,
finitude, horizonte e vídeo completo; essa aceitação é estrutural, não de
qualidade ou segurança de controle.

As 151 saídas têm hashes distintos; portanto a curva não é uma trajetória
repetida sobre o vídeo. A mudança absoluta média das ações entre ticks foi
`0,005029` e todos os tensores permaneceram finitos. Os intervalos e
estatísticas completos estão em `POLICY_FUTURE_TRAJECTORY_DIAGNOSTICS.json`.

### Perfil do gargalo

Uma chamada instrumentada separou os `15,973 s` assim:

| Etapa | Tempo |
|---|---:|
| VAE encoder ONNX | 8,679 s |
| Transformer policy ONNX, 30 passos | 6,913 s |
| Host, pré/pós-processamento e scheduler | 0,381 s |

No perfil isolado, todos os `42.228/42.228` eventos de nós do transformer
executaram no CUDA Execution Provider: não há fallback de nós para CPU. Um
passo ONNX levou `221,13 ms`, contra `155,05 ms` no PyTorch FP16 nativo
(`1,43x` mais lento). Tornar a atenção exportável adicionou apenas `2,4%` no
PyTorch. CUDA Graph mediu `218,34 ms` e não trouxe ganho material. Isso aponta
para fragmentação/falta de fusões do grafo ONNX; o caminho de otimização é
TensorRT/fusões de atenção e um engine construído no próprio Orin.

## Comparação nativa BF16 e INT4

| Backend | Tempo total com carga | Prefill médio | Decode ponderado | Pico CUDA |
|---|---:|---:|---:|---:|
| BF16 | 344.857 s | 0.054 s | 22.462 ms/token | 9.392 GB |
| TorchAO INT4 weight-only (lineares de entendimento) | 444.481 s | 0.103 s | 28.206 ms/token | 6.928 GB |

Correspondência textual byte a byte: `0/151`. Similaridade média de sequência:
`0.1957`.

| Comparação de texto | Iguais byte a byte | Similaridade média |
|---|---:|---:|
| ONNX W4A16 vs BF16 nativo | 0/151 | 0.0449 |
| ONNX W4A16 vs TorchAO INT4 | 0/151 | 0.0574 |

BF16 e TorchAO INT4 usam a mesma execução nativa com pensamento habilitado e
limite de 96 tokens, portanto formam a comparação controlada. O Edge-LLM ONNX
gera a resposta direta e pode terminar por EOS; as linhas ONNX-vs-nativo são
somente diagnósticas e não representam paridade de configuração de geração.

Estado TorchAO: `169` tensores
INT4 no arquivo e `169` no modelo carregado; chaves ausentes `0`;
inesperadas `0`.
Essa referência quantiza os `nn.Linear` da torre de entendimento; embeddings,
normalizações e os pesos `*_moe_gen*` permanecem BF16. "INT4" aqui significa
weight-only misto, não que cada byte do modelo tenha quatro bits.

Todos os textos integrais permanecem nos JSONs e no TXT, sem substituição por resumos.

Neste ambiente, o INT4 usou `26.2%`
menos memória CUDA, porém o decode foi
`1.256x` o BF16
(`25.6%` mais lento).
A baixa correspondência textual mostra que esta quantização altera o raciocínio;
ela não deve ser tratada como paridade de qualidade sem avaliação semântica do
caso de uso.

## Demais modalidades ONNX

| Modalidade | Modo | Frames | Inferência | Resultado |
|---|---|---:|---:|---|
| Forward UMI | forward_dynamics | 32 | 8.231 s | PASS |
| Policy UMI | policy | 17 | 6.761 s | PASS |
| I2V 121 | image2video | 121 | 188.791 s | PASS |

## Pacote

`/data/wesleyferreiramaia/wokzone-alpamayo/COSMOS3EDGE_ORIN_COMPLETE_RUNTIME`
separa `10.759.761.656` bytes (`10,0208 GiB`) operacionais de
`18.105.005.161` bytes (`16,8616 GiB`) de referência.
O ZIP operacional exclui `weights/`; o ZIP de referência contém BF16,
TorchAO INT4 e a fonte AWQ. Os ONNX MoT e VAE usam pesos
externos compartilhados. Engines A100 foram removidos porque não são portáveis
para o Orin. O reasoner permanece W4A16 INT4; visual, MoT e VAE permanecem FP16.

## Conclusão de desempenho

A implementação executa cada timestamp exigido, mas **não é tempo real a
10 Hz** no A100: tanto o reasoner médio quanto inverse e o rollout AV
ultrapassam 100 ms. O rollout AV também não passou por validação de trajetória
contra ground truth e continua temporalmente instável após a correção SE(3).
O Jetson AGX Orin precisa repetir a medição com engines construídos no próprio
dispositivo; não há base técnica para prometer 10 Hz nem qualidade de controle.
