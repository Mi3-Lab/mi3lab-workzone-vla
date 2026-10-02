# Reformulação com Cosmos3-Edge — relatório contínuo

Objetivo: mover o paper de *"um VLM único compete com detector+CLIP"* para
**estimativa de travessia ancorada em trajetória**, resolvendo o gap de
ego-relevância que a versão atual apenas nomeia, e as pendências dos revisores
(generalização, night/weather, nuisance rate).

Tudo executado no Jetson AGX Orin (A100 permanece desligado).

---

## Contexto: por que reformular

O paper atual identifica ego-relevância como a causa única do gap de precisão e
propõe como *future work*: *"supervision that grounds state in the vehicle's
trajectory"*. O Cosmos3-Edge **gera a ego-trajetória em tempo de inferência**,
o que permite resolver isso **em runtime, sem retreino** — compatível com a
restrição de não usar o A100.

---

## FASE 0 — Gate de viabilidade

### 0.0 Preparação (feito)

| item | resultado |
|---|---|
| Engines Cosmos3-Edge no Orin | reasoner INT4 AWQ 1.27 GB + visual FP16 983 MB |
| Edge-LLM v0.9.0 + patch Cosmos3 | 6 arquivos, rebuild OK |
| **Regressão do nosso 2B** | **8/8 idênticos** — pipeline do paper intacto |
| onnxruntime-gpu 1.23.0 | instalado isolado em `~/c3traj-libs` (`--target`), numpy do sistema |
| TensorrtExecutionProvider | **ativo e validado executando grafo real** (não só listado) |

Correção necessária no Jetson: `torch.linalg.solve` em CUDA falha
(`cusolverDnXsyevBatched_bufferSize` ausente). Os sistemas do UniPC são
minúsculos (≤4×4), então resolvemos na CPU — patch em
`scheduling_unipc_multistep.py` (backup `.pre-jetson-fix`). Sem custo de
performance.

### 0.1 Latência da trajetória no Orin (ORT + CUDA EP)

Config recomendada pelo pacote: **tier 256 / 8 steps**, `boston.mp4`, samples 60–62.

| plataforma | latência/tick | cabe em 6 s? |
|---|---|---|
| A100 (medido pelo pacote) | 1,99 s | sim |
| Thor T5000 (projeção do pacote) | ~5,6 s | sim |
| **Orin (medido por nós, CUDA EP)** | **8,8 – 9,6 s** | **NÃO** |

Deslocamento reconstruído em 6 s: 48,7 / 48,2 / 47,2 m — consistente entre ticks.

**Achado:** o pacote nunca tinha sido executado em Jetson real e projetava
~5,6 s no Thor por um fator 2,8×. No Orin o fator medido é **~4,5×**, e a
configuração recomendada **não atinge o orçamento de 6 s** com ORT/CUDA.
A própria documentação indica o caminho: *"o ONNX Runtime é 2,5× mais lento que
o PyTorch nativo; só o TensorRT supera o nativo"*. Próxima medição: TRT EP.

### 0.2 Ground truth de trajetória via GPS (nosso diferencial)

O pacote deixou explicitamente em aberto: *"qual resolução é correta exige
verdade de campo (GPS/odometria)"*. **Nós temos o GPS** — os `.map` que
acompanham a footage da Califórnia.

Parser implementado em `pipeline/gps_map_parser.py` (1 Hz, lat/lon/velocidade,
projeção equirretangular local). Validação de sanidade em `Merced_Day_01`:

| janela | deslocamento GPS | esperado pela velocidade | erro |
|---|---|---|---|
| t=60 s, +6 s | 46,0 m | 48,1 m (29 km/h) | 4% |
| t=0 s, +6 s | 92,1 m | 93,3 m (56 km/h) | 1% |

Parser validado. Permite comparar a trajetória gerada contra a real — algo que
o pacote não podia fazer.

---

## Estado das fases

- [x] 0a — ambiente ONNX Runtime GPU + TRT EP validado
- [~] 0b — latência no Orin: medida em CUDA EP (8,8–9,6 s); TRT EP em curso
- [x] 0c — parser/validação de GPS pronto (falta cruzar com trajetória gerada)
- [ ] 1 — ego-relevância geométrica
- [ ] 2 — avaliação nos 208 vídeos
- [ ] 3 — benchmark OOD Califórnia
- [ ] 4 — reescrita

---

## Resultados já consolidados (do teste anterior, mesma sessão)

### Generalização OOD — footage Califórnia, 25 sinais anotados
| sistema | geral | dia | noite | chuva | névoa |
|---|---|---|---|---|---|
| nosso 2B fine-tunado | 4% | 10% | 0% | 0% | 0% |
| detector YOLO+CLIP | 72% | 100% | 50% | 50% | 40% |
| **Cosmos3-Edge base (zero-shot)** | **92%** | 100% | 100% | 75% | 80% |

### Calibração (ROADWork validation)
- positivos: 6/6 detectados (100% recall)
- negativos puros: **3/6 falsos positivos** → affirmation bias forte
- ego-state: erra (respondeu OUTSIDE em frame claramente INSIDE)

### Latência do reasoner (Orin, INT4, greedy, 736×416)
GATE 162 ms · EGO 182 ms · SIGN 556 ms · DESC 488 ms
(vs 90 **segundos** no caminho BF16/HF — ~550× mais rápido)

---

## FASE 1 — Ego-relevância: resultado que redefine a contribuição

### 1.1 O reasoner Cosmos3-Edge sabe localizar (o nosso 2B não sabia)

| capacidade | resposta | latência |
|---|---|---|
| bounding boxes 2D | `[{"bbox_2d":[195,292,209,310],"label":"traffic cones"}, ...]` (JSON) | 1446 ms |
| pontos 2D | `[{"point_2d":[105,696],"label":"cones"}, ...]` | 1397 ms |
| pergunta de faixa | *"The traffic cones are on the ego vehicle's driving lane."* | **306 ms** |

Isso abriu a hipótese de uma ego-relevância **barata** (306 ms) por prompt,
dispensando a trajetória (8,8 s). Testamos — e a hipótese caiu.

### 1.2 TESTE DECISIVO: prompt de faixa NÃO resolve ego-relevância

Prompt: *"Are the work-zone objects on the ego vehicle's driving lane, on an
adjacent lane, or off-road/sidewalk?"*, nos casos exatos que o paper aponta
como a família de falha.

| ground truth | respostas obtidas |
|---|---|
| **outside** (6 negativos puros) | offroad/sidewalk · offroad/sidewalk · adjacent lane · offroad/sidewalk · on road · offroad/sidewalk |
| **inside** (6 positivos, carro *dentro* da zona) | offroad/sidewalk · adjacent lane · adjacent lane · adjacent lane · offroad/sidewalk · offroad/sidewalk |

**Não há discriminação**: com o veículo comprovadamente dentro da zona de obra,
o modelo responde majoritariamente "offroad/sidewalk". Ou seja, a resposta é
praticamente independente do ground truth.

**Por que isso importa (e fortalece o paper):** a versão atual documenta que
prompts de distância/relevância não funcionam no nosso 2B fine-tunado, e atribui
isso à cobertura do fine-tuning. Agora mostramos que **um world model de
physical-AI, que percebe a cena corretamente** (92% OOD, descrições precisas,
bboxes corretas) **falha exatamente do mesmo jeito**. A conclusão deixa de ser
sobre o nosso checkpoint e vira geral:

> Ego-relevância não é alcançável por *prompting*. Perceber corretamente os
> objetos e julgar se eles reivindicam o caminho do ego são capacidades
> distintas; a segunda exige **geometria** (trajetória), não linguagem.

Isso converte a trajetória de "escolha de design" em **necessidade demonstrada**
— e é a justificativa central da reformulação.

### 1.3 O corredor geométrico com bboxes do próprio modelo também não resolve

Implementado `pipeline/ego_corridor.py`: corredor trapezoidal (converge no
horizonte, alarga na base) e teste de sobreposição com os bboxes do reasoner.

Bug encontrado e corrigido no caminho: o reasoner emite coordenadas
**normalizadas 0–1000** (convenção Qwen-VL), não pixels — um `point_2d` com
y=696 num frame de 416 px denuncia a escala. O parser agora detecta isso.

Mesmo com a escala correta, nos mesmos 12 casos:

| | falsos positivos | detectados |
|---|---|---|
| corredor + bboxes do reasoner | **6/6** | 4/6 |

**Diagnóstico:** o prompt *"Locate all work-zone objects"* **pressupõe que os
objetos existem**, e o modelo obedece mesmo em cenas limpas — o mesmo
affirmation bias medido no GATE (3/6). O corredor herda o viés da fonte, então
não pode corrigi-lo.

**Consequência de projeto:** a ego-relevância precisa de uma fonte de evidência
**independente da linguagem**. A trajetória do action head é essa fonte: ela não
é condicionada por um prompt que pressupõe a existência de obra.

### 0.3 TensorRT EP está bloqueado para o transformer (achado técnico)

A documentação do pacote aponta o TensorRT como caminho para caber no orçamento
(*"só o TensorRT supera o nativo"*). Tentamos `COSMOS3EDGE_ORT_TENSORRT=1`.
Após ~10 min de compilação silenciosa (19 GB de RSS), o erro real:

```
[libprotobuf ERROR] onnx.ModelProto exceeded maximum protobuf size of 2GB: 5915463748
```

O TensorRT Execution Provider precisa serializar o **modelo inteiro** em
protobuf, cujo limite é 2 GB; o transformer MoT tier-256 tem **5,9 GB**.
**Bloqueio fundamental de formato, não de hardware** — nenhum ajuste de tier ou
de steps resolve, porque o problema é o tamanho do grafo serializado.

Consequências:
- resta o **CUDA EP**: 8,8–9,6 s/tick no Orin, contra orçamento de 6 s;
- o caminho viável seria construir engines TensorRT *standalone* por subgrafo
  (como o pacote já faz para o VAE via `trtexec`), o que exige reexportar o
  transformer em partes — trabalho fora do escopo desta sessão;
- portanto, **no Orin, a trajetória a 10 Hz não é tempo real hoje**. Ela
  permanece utilizável como evidência *sob demanda* (ver §1.4).

Isto é exatamente o tipo de informação que só aparece em hardware real — o
pacote projetava ~5,6 s no Thor e nunca havia sido executado num Jetson.

### 0.4 Validação da trajetória contra GPS real (resolve a pendência do pacote)

Rodamos o inverse dynamics no clipe da Califórnia (`Merced_Day_01`, t=60–80 s,
2560×1440) e comparamos com o GPS.

**Metodologia — cuidados que mudaram o resultado:**
1. *Janela causal.* O modo é `inverse_dynamics` (*"observed AV motion over a
   causal 61-frame window; not a future planner"*), então a janela GPS correta é
   **[t−6 s, t]**, não [t, t+6 s]. Comparar com a janela futura seria errado.
2. *GPS a 1 Hz vs trajetória a 10 Hz.* Diferenciar posições é instável: deslocar
   a janela em 0,1 s troca 46,0 m por 33,0 m (um fix inteiro entra/sai).
   Passamos a **integrar o canal de velocidade** com interpolação linear, que é
   suave (47,5 → 46,5 → 45,6 m para os mesmos deslocamentos).

**Resultado (velocidade ~28 km/h):**

| tick | trajetória | GPS | erro |
|---|---|---|---|
| t=6,0 s | 35,1 m | 47,5 m | −26,1% |
| t=6,1 s | 34,9 m | 46,5 m | −25,1% |
| t=6,2 s | 34,2 m | 45,6 m | −25,1% |

MAE 11,8 m · **bias −11,8 m (sistemático)** · MAPE 25,4% · **correlação r=+0,95**

**Segundo ponto de medição (velocidade ~48 km/h), no mesmo clipe:**

| tick | trajetória | GPS | erro |
|---|---|---|---|
| t=14,0 s | 86,0 m | 80,0 m | +7,5% |
| t=14,1 s | 85,4 m | 80,7 m | +5,8% |
| t=14,2 s | 86,0 m | 81,4 m | +5,7% |

MAE 5,1 m · MAPE **6,3%**

**Leitura corrigida.** Com apenas o primeiro ponto, o bias de −25% sugeria um
fator de escala constante (e portanto que o tier 256 subestimava). **O segundo
ponto refuta isso**: o erro troca de sinal com a velocidade.

| velocidade | erro da trajetória |
|---|---|
| ~28 km/h | **−25%** (subestima) |
| ~48 km/h | **+6%** (superestima levemente) |

Ou seja, a reconstrução é **boa em velocidade de cruzeiro (≈6%) e degrada em
velocidade baixa (−25%)** — consistente com o inverse dynamics ter menos sinal
visual quando o deslocamento entre frames é pequeno. Não é um simples fator de
escala, então **não podemos concluir a partir daqui que o tier 480 seja o
correto**; a pendência §2 do pacote continua parcialmente aberta, agora com uma
caracterização útil: o erro depende do regime de velocidade.

Nota metodológica: a correlação r=+0,95 do primeiro grupo e r=+0,07 do segundo
não são comparáveis — cada grupo tem 3 ticks separados por 0,1 s, com variação
quase nula. A correlação aqui não é informativa; os números que valem são o
erro relativo e seu sinal.

Latência no clipe da Califórnia: **19,9–20,8 s/tick** (contra 8,8–9,6 s no
`boston.mp4`), consistente com o vídeo de entrada maior (2560×1440).

---

## Balanço da Fase 0+1 — o que a reformulação pode e não pode prometer

### Confirmado (forte, reproduzível)

1. **Cosmos3-Edge INT4 roda no Orin em latência competitiva** — GATE 162 ms,
   contra 90 s no caminho BF16/HF. Quantização deixou de ser bloqueio.
2. **Generalização OOD muito superior** — 92% na Califórnia (dia/noite/chuva/
   névoa) contra 4% do nosso 2B fine-tunado e 72% do detector.
3. **Ego-relevância não é alcançável por prompt** — nem num world model de
   physical-AI que percebe a cena corretamente. Isso *generaliza* a tese do
   paper para além do nosso checkpoint.
4. **Validação de trajetória contra GPS** — inédita (o pacote não tinha ground
   truth): erro ≈6% a 48 km/h, −25% a 28 km/h.

### Bloqueado (medido, não contornável nesta sessão)

5. **TensorRT EP inviável** para o transformer (protobuf 2 GB vs grafo 5,9 GB).
6. **Trajetória não é tempo real no Orin**: 8,8–9,6 s/tick (boston) e
   19,9–20,8 s/tick (Califórnia, 2560×1440), contra orçamento de 6 s.
7. **Corredor geométrico com bboxes do próprio modelo não corrige o viés** —
   o prompt de localização pressupõe existência e herda o affirmation bias.

### Consequência honesta para a tese

A versão mais ambiciosa — *"ego-relevância resolvida por trajetória em tempo
real"* — **não se sustenta no Orin hoje** (item 6). O que se sustenta:

- a **impossibilidade de resolver ego-relevância por linguagem** (item 3) é um
  resultado negativo forte e generalizável, que **eleva** a contribuição do
  paper atual em vez de substituí-la;
- o **Cosmos3-Edge como percepção** (itens 1–2) é uma troca de backbone com
  ganho enorme de generalização, mensurável no mesmo protocolo;
- a trajetória permanece viável como **evidência sob demanda** (não por frame),
  com custo declarado — e a validação por GPS (item 4) sustenta o quanto dela
  se pode confiar, por regime de velocidade.

Ou seja: a reformulação continua valendo, mas com o eixo em **percepção +
generalização + o resultado negativo sobre ego-relevância**, e não em
"trajetória em tempo real".

---

## FASE 2 — Avaliação no benchmark (208 vídeos)

Harness estendido: `evaluate_cascade.py --infer-size WxH` (a torre visual do
Cosmos3-Edge é congelada num grid 26×46 = **736×416**, então o redimensionamento
proporcional usado para o nosso 2B não serve).

**Custo real:** ~33 s/vídeo → **~1,9 h para os 208**, com cache de predições em
`~/eval_cache/cosmos3edge/`.

> Nota de método: uma estimativa inicial minha de "~6 min/vídeo → 21 h" estava
> **errada** — era artefato de output bufferizado (faltava `python3 -u`), não
> lentidão real. Registrado aqui porque quase levou a reduzir o escopo sem
> necessidade.

Configuração: greedy (`--temperature 0`), mesmo protocolo latency-honest, mesmo
split de validação, mesmas métricas paper-exact — comparável à Tabela 2.

---

## FASE 3 — Benchmark OOD da Califórnia (formalização)

Footage gravada pela equipe, **fora do ROADWork** — logo, sem qualquer
possibilidade de leakage com o treino de nenhum dos sistemas comparados.

### Composição

| dimensão | conteúdo |
|---|---|
| vídeos | 27 (26 com GPS 1 Hz; falta apenas `CA99_Night_06`) |
| geografia | CA50, CA99, I5, Merced, Fresno, Sacramento, Shasta Lake |
| câmera | COOAU dashcam (fisheye + overlay de timestamp/GPS) — **diferente** da usada no ROADWork |
| condições | 13 dia · 11 noite · 1 evening · 1 night-fog · 1 night-rain → **52% não-diurno** |
| categorias | 15 With_SpeedLimit · 6 Only_Signs · 6 Lane_Change |
| ground truth | 39 placas de obra anotadas por timestamp em 14 vídeos + GPS |

Distribuição dos 39 sinais: 15 dia · 9 noite · 5 névoa · 4 chuva · 4 sunset · 2 evening.

### Por que isso importa para os revisores

A limitação declarada no paper atual é *"the benchmark is two daytime U.S.
cities (night/weather untested)"*. Este conjunto cobre exatamente o que faltava:
**noite, chuva e névoa**, em outra geografia e outra câmera, com zero leakage.

### Limitação honesta deste benchmark

Ele tem ground truth de **presença de evidência de obra** (timestamps de placa),
**não** de estado de travessia por frame. Portanto sustenta uma avaliação de
*percepção/detecção sob domain shift*, e **não** substitui o benchmark ROADWork
para métricas de traversal state (IoU por estado, timing de entrada). Anotar
traversal states aqui é trabalho futuro.

### Estado

Medição preliminar já feita em 25 sinais / 6 vídeos (ver tabela acima:
Cosmos3-Edge 92% · detector 72% · nosso 2B 4%). Extensão para os **39 sinais /
14 vídeos** fica pendente da GPU (ocupada pela Fase 2).

### 2.1 Resultado parcial (44/208 vídeos) — a divisão de trabalho fica explícita

Script: `pipeline/compare_all_systems.py` (tabela a partir dos caches).

| sistema | n | acc | F1 | IoU out | IoU **appr** | IoU **in** | in P | in R | FA/h |
|---|---|---|---|---|---|---|---|---|---|
| nosso 2B (sampled) | 208 | 62,5% | 0,464 | 0,586 | 0,152 | 0,535 | 54,1% | 96,8% | 60,1 |
| nosso 2B (greedy) | 208 | 62,0% | 0,453 | 0,581 | 0,152 | 0,532 | 56,8% | 97,3% | 56,2 |
| detector+CLIP | 208 | 62,9% | 0,470 | 0,617 | 0,124 | 0,544 | 79,9% | 95,7% | 30,9 |
| detector +textFE | 208 | 62,9% | 0,473 | 0,594 | 0,140 | 0,555 | 81,9% | 95,1% | 52,3 |
| **Cosmos3-Edge INT4** | 44* | 52,3% | 0,316 | **0,683** | **0,273** | **0,002** | 40,0% | **4,2%** | **32,8** |
| base ckpt zero-shot | 208 | 48,8% | 0,289 | 0,638 | 0,216 | 0,012 | 100% | 0,5% | 29,2 |

\* parcial, execução em andamento.

**Leitura — e é o achado central da Fase 2.** O Cosmos3-Edge não é "pior"; ele é
**assimétrico** de um jeito muito informativo:

- **melhor que todos** em `OUTSIDE` (0,683 vs 0,617 do detector) e em
  **`APPROACHING` (0,273 vs 0,152 do nosso 2B — quase o dobro)**, que é
  justamente o estado de fronteira que o paper identifica como o mais difícil e
  mais útil;
- **nuisance baixo** (32,8 FA/h vs 60,1 do nosso 2B) — comparável ao detector;
- **colapsa em `INSIDE`** (IoU 0,002; recall 4,2%): o sistema quase nunca
  transiciona para dentro da zona.

A causa é direta e já havia aparecido no teste pontual: o canal `EGO` do modelo
base responde `OUTSIDE` mesmo com o veículo dentro da obra. **Percepção e estado
de travessia são capacidades separadas**: o pré-treino físico entrega a
primeira (detecção, descrição, fronteira, generalização OOD), e a segunda exige
o fine-tuning de ego-state do nosso currículo.

Isso fecha o argumento com o resultado da Fase 1: nem *prompting* (§1.2) nem
*troca de backbone* (§2.1) resolvem ego-relevância/estado — só supervisão
dedicada. O que a troca de backbone **entrega de graça** é generalização (92%
OOD) e localização de fronteira (`APPROACHING`).

---

## FASE 4 — Proposta de reescrita

### O trade-off medido (o novo núcleo do paper)

Os dois eixos que ninguém mediu junto antes, no mesmo protocolo e hardware:

| | estado (`INSIDE` IoU) | generalização (OOD Califórnia) |
|---|---|---|
| nosso 2B fine-tunado | **0,535** | **4%** |
| Cosmos3-Edge base | **0,002** | **92%** |
| detector+CLIP | 0,544 | 72% |

Um modelo sabe *onde o carro está* mas só no domínio de treino; o outro
*enxerga qualquer obra em qualquer condição* mas não sabe onde o carro está.
**Especialização e generalização estão em lados opostos**, e a fronteira entre
elas é exatamente a capacidade de ego-state.

### Tese proposta

> Estimativa de travessia de zona de obra decompõe-se em duas capacidades com
> comportamentos de transferência **opostos**: **percepção** (detectar,
> descrever, localizar) transfere do pré-treino físico e generaliza para
> domínios novos; **ego-state** (o veículo está dentro?) **não** é alcançável
> por prompting nem por troca de backbone — exige supervisão dedicada, e essa
> supervisão custa generalização.

### Evidência que sustenta cada elo

| afirmação | evidência |
|---|---|
| ego-state não vem de prompting | §1.2 — world model responde independente do GT |
| ego-state não vem de backbone melhor | §2.1 — Cosmos3-Edge: `INSIDE` IoU 0,002, recall 4,2% |
| ego-state vem de fine-tuning | nosso 2B: `INSIDE` IoU 0,535, recall 96,8% |
| fine-tuning custa generalização | 4% OOD vs 92% do base |
| percepção transfere e generaliza | 92% OOD (dia/noite/chuva/névoa), `APPROACHING` IoU 0,273 |

### O que se mantém do paper atual

- o **protocolo latency-honest** (contribuição metodológica, agora usado para
  comparar 6 sistemas);
- o **cascade evidence-first** e as descobertas de prompt (affirmation bias,
  transcrição fiel), agora **confirmadas num segundo modelo**;
- a comparação pareada com o detector, incluindo o híbrido +textFE.

### O que muda

- a pergunta deixa de ser *"um VLM substitui o detector?"* e passa a ser
  *"quais capacidades transferem e quais precisam ser ensinadas?"*;
- o gap de ego-relevância deixa de ser limitação do nosso checkpoint e vira
  **resultado**: é propriedade da tarefa;
- entra a dimensão de **generalização OOD**, que responde à limitação
  "two daytime U.S. cities" apontada pelos revisores.

### Título candidato

*"Perception Transfers, Ego-State Does Not: Decomposing Embedded Work-Zone
Traversal Estimation"*

### 2.2 RESULTADO FINAL (208/208) — com correção do parcial

| sistema | n | acc | F1 | IoU out | IoU **appr** | IoU **in** | in P | in R | FA/h |
|---|---|---|---|---|---|---|---|---|---|
| nosso 2B (sampled) | 208 | 62,5% | 0,464 | 0,586 | 0,152 | 0,535 | 54,1% | 96,8% | 60,1 |
| nosso 2B (greedy) | 208 | 62,0% | 0,453 | 0,581 | 0,152 | 0,532 | 56,8% | 97,3% | 56,2 |
| detector+CLIP | 208 | 62,9% | 0,470 | **0,617** | 0,124 | 0,544 | 79,9% | 95,7% | **30,9** |
| detector +textFE | 208 | 62,9% | 0,473 | 0,594 | 0,140 | **0,555** | **81,9%** | 95,1% | 52,3 |
| **Cosmos3-Edge INT4** | 208 | 43,6% | 0,290 | 0,609 | **0,233** | 0,007 | 38,9% | 3,8% | 46,1 |
| base ckpt zero-shot | 208 | 48,8% | 0,289 | 0,638 | 0,216 | 0,012 | 100%* | 0,5% | 29,2 |

\* precisão degenerada: 1 único evento predito.

#### Correções sobre o parcial de 44 vídeos

Duas afirmações que fiz com o parcial **não sobreviveram** ao conjunto completo:

| afirmação (parcial, n=44) | valor final (n=208) | veredito |
|---|---|---|
| "melhor que todos em `OUTSIDE` (0,683)" | 0,609 vs **0,617** do detector | **falso** — fica atrás |
| "nuisance baixo (32,8 FA/h), comparável ao detector" | **46,1** vs 30,9 | **falso** — 49% pior |

O que **se manteve**:

| afirmação | parcial | final |
|---|---|---|
| melhor `APPROACHING` de todos os sistemas | 0,273 | **0,233** (vs 0,152 nosso, 0,124 detector) |
| colapso em `INSIDE` | IoU 0,002 / recall 4,2% | IoU 0,007 / recall **3,8%** |

**Conclusão que sobrevive aos 208 vídeos:** o Cosmos3-Edge base localiza a
fronteira `APPROACHING` melhor que qualquer sistema treinado para a tarefa
(+53% sobre o nosso 2B, +88% sobre o detector), e **colapsa completamente** em
`INSIDE`. Isso sustenta a tese da decomposição — percepção de fronteira
transfere, ego-state não — **sem** as alegações de OUTSIDE e nuisance, que eram
artefato de amostra pequena.

---

## FASE 2.3 — CORREÇÃO IMPORTANTE: a comparação estava injusta

Levantado pelo usuário: *"o que está sendo feito zero-shot? se usarmos algo
parecido com o 2B ele vai melhor"*. **Procede, e em dois níveis.**

### Nível 1: a tabela compara fine-tunado vs não-fine-tunado

Todo o Cosmos3-Edge é **zero-shot**; o nosso 2B passou por 6 estágios de SFT.
A tabela §2.2 não compara arquiteturas — compara regimes de treino. Isso precisa
estar explícito em qualquer conclusão.

### Nível 2: usei os prompts do 2B no modelo base (erro metodológico)

O `EGO_PROMPT` foi desenhado para o nosso checkpoint fine-tunado. Aplicado ao
modelo base, ele **induz viés**, não mede capacidade. Verificado primeiro que
não era erro de parsing (o modelo responde `OUTSIDE<|im_end|>`, parseado
corretamente) — o erro é real, mas a causa é o prompt.

Teste balanceado (10 INSIDE + 10 OUTSIDE puros):

| prompt do canal EGO | INSIDE | OUTSIDE | acurácia balanceada |
|---|---|---|---|
| A — o do nosso 2B | 1/10 | 9/10 | 50,0% |
| B — explícito p/ base | 9/10 | 3/10 | **60,0%** |
| C — simétrico | 1/10 | 10/10 | 55,0% |
| D — "já entrou?" | 0/10 | 9/10 | 45,0% |

Todos ≈ acaso: **o modelo segue o viés do prompt, não a imagem**. O prompt B
"resolve" o INSIDE apenas trocando o viés de lado.

### O que efetivamente funciona: geometria, não julgamento verbal

Em vez de pedir *julgamento de estado*, usar a capacidade que o modelo
comprovadamente tem — **localizar**. Objetos próximos aparecem na parte
inferior do quadro, então a coordenada `y` máxima das bboxes é um proxy de
proximidade.

| método | acurácia balanceada |
|---|---|
| prompts de julgamento (A–D) | 45–60% |
| prompt de distância **verbal** | incoerente (responde "NEAR (>10M)", "FAR" em frames INSIDE) |
| **proximidade geométrica (ymax das bboxes)** | **80,0%** |

`ymax` médio: **INSIDE 0,653 vs OUTSIDE 0,361**.

> **A tese se refina:** não é que ego-state "não transfere". É que ele **não é
> acessível por linguagem** — nem pedindo julgamento, nem pedindo distância em
> palavras — mas **é recuperável da geometria que o próprio modelo já produz**.

Isso é acionável e Jetson-only (sem A100): um estimador de estado por
proximidade, calibrado, sobre as detecções do modelo base.

**Rigor:** os 80% acima foram obtidos com o threshold ajustado nos mesmos dados
— enviesado. Uma calibração honesta (threshold no split de *calibração*,
avaliado no de *validação*) está em curso; o número que vale é esse.

### 2.4 Calibração honesta do estimador geométrico — número final

Protocolo correto: threshold ajustado **no split de calibração**, avaliado
**no de validação** (nunca nos mesmos dados).

| | threshold | acurácia balanceada |
|---|---|---|
| ajustado e testado nos mesmos dados (§2.3) | ymax≥0,62 | 80,0% *(enviesado)* |
| **calibração → validação (honesto)** | **ymax≥0,30** | **69,2%** |

Detalhe na validação: INSIDE **14/14**, OUTSIDE **5/13**.

**Diagnóstico:** o threshold ótimo da calibração caiu muito baixo (0,30), o que
degenera o critério para *"detectou algum objeto?"*. Nesse regime ele herda o
affirmation bias do detector-por-prompt: acerta todo INSIDE e erra a maioria dos
negativos. A separação real de `ymax` existe (0,653 vs 0,361 nas médias), mas
com n=14+14 na calibração a escolha do threshold é instável.

### Onde isso deixa a questão do usuário

| abordagem | acurácia balanceada de ego-state | precisa de A100? |
|---|---|---|
| prompt do 2B no modelo base | 50,0% | não |
| melhor prompt calibrado p/ o base | 60,0% | não |
| geometria (bbox), calibração honesta | **69,2%** | não |
| **fine-tuning (nosso 2B)** | `INSIDE` IoU 0,535 / recall 96,8% | **sim** |

**Conclusão honesta:** calibrar para o Cosmos3 **ajuda de verdade** (50% → 69%),
e confirma que a informação de estado *está* no modelo — mas em forma
geométrica, não linguística. Ainda assim, **não substitui o fine-tuning**: o
nosso 2B treinado continua muito superior em estado. O que a troca de backbone
entrega sem treino é generalização (92% OOD) e fronteira (`APPROACHING` 0,233).

Limitação declarada: n pequeno (14+14 calibração, 14+13 validação), um frame por
vídeo. Um estimador geométrico sério exigiria calibração em muito mais frames.

---

## FASE 2.5 — Busca sistemática pelo melhor resultado

Pedido: *"procura o melhor jeito pra ter os melhores resultados"*. Testadas 11
estratégias de fusão + varredura de debounce, tudo sobre os caches (sem
re-executar inferência). **Protocolo honesto:** validação partida ao meio —
estratégias escolhidas no `dev` (104 vídeos), reportadas no `test` (104).

### Fusão: não funciona

| estratégia | dev F1 | test F1 |
|---|---|---|
| vote(2B, det, c3e) — **melhor no dev** | 0,470 | **0,452** |
| det sozinho | 0,462 | **0,478** |
| detFE sozinho | 0,468 | **0,478** |
| c3e entrada + 2B `INSIDE` | 0,429 | — |
| det gate + c3e estado | 0,296 | — |

A vencedora no `dev` **não se sustentou** no `test` (0,452 < 0,478) — o
protocolo dev/test pegou exatamente o overfitting de seleção que existe para
pegar. E as fusões com o Cosmos3 na *entrada* ficaram **piores que os
componentes**, refutando a hipótese de que ele ajudaria na fronteira apesar do
`APPROACHING` IoU superior isoladamente.

Único ganho da fusão: recall `INSIDE` 100% e +1,3 pp de acurácia — ao custo de
−0,026 de F1 e −14 pp de precisão. Não compensa.

### Debounce: confirma o achado do paper atual

| sistema | inP sem debounce | inP com debounce |
|---|---|---|
| nosso 2B | 58,4% | **76,7%** (d=1 s) |
| detFE | 84,7% | **86,4%** (d=2 s) |
| det | 82,7% | 84,2% (d=2 s) |

O ganho é grande justamente onde o paper já dizia que seria (nosso 2B, flicker);
no detector é pequeno porque o EMA dele já faz esse papel.

### Melhor configuração medida

| objetivo | melhor sistema | números (test) |
|---|---|---|
| **macro-F1 e nuisance** | `det + debounce 2s` | F1 **0,479** · FA/h **30,3** |
| **precisão de evento** | `detFE + debounce 2s` | inP **86,4%** · IoU `in` **0,570** |
| **generalização OOD** | **Cosmos3-Edge** | **92%** (vs 72% det, 4% 2B) |

### Conclusão da busca (honesta)

**No benchmark ROADWork, o Cosmos3-Edge não melhora o resultado** — nem sozinho,
nem em fusão. O melhor sistema continua sendo o **detector+CLIP com debounce**
(opcionalmente com text fast-entry, se o objetivo for precisão).

O valor do Cosmos3-Edge está **fora do domínio de treino**, onde ele é o único
que não degrada. Isso não é uma contradição: é o trade-off central medido —
*especialização vence no domínio, pré-treino vence fora dele*.

---

## FASE 5 — Detector + Cosmos3-Edge como verificador (ideia do usuário)

> *"E se eu substituísse o CLIP por ele no outro sistema? Já que teria o YOLO
> para detectar rápido e chamando o Cosmos pra ajudar na frequência menos de
> 1 Hz"*

### Por que a arquitetura faz sentido

No pipeline detector+CLIP portado, o CLIP **já é** o componente de baixa
frequência (roda a cada `CLIP_INTERVAL=3` ciclos) e é o **elo mais fraco sob
domain shift**. Trocá-lo pelo Cosmos3-Edge combina:

- **velocidade do YOLO** — 43 ms/ciclo, ~23 Hz;
- **percepção do Cosmos3** — 92% OOD (dia/noite/chuva/névoa) contra um CLIP
  genérico;
- **custo amortizado** — o verificador só roda 1 a cada N ciclos.

Implementação: `pipeline/evaluate_yolo_cosmos3.py`. `Cosmos3Verifier` é
drop-in de `ClipFusion.global_score`; o `GATE` retorna score com sinal
(+1/−1/0) para casar com o `logistic(3x)` que o pipeline já aplicava ao cosseno
do CLIP. Todo o resto (EMA, boost laranja, máquina de estados, thresholds
publicados) permanece intacto.

### Latência medida no Orin — o ponto forte

| sistema | ciclo | taxa |
|---|---|---|
| detector+CLIP | 43 ms | 23 Hz |
| **detector + Cosmos3 (interval 3)** | **67 ms** | **15 Hz** |
| nosso 2B (patrol) | 238 ms | 4,2 Hz |
| Cosmos3 puro (patrol) | 718 ms | 1,4 Hz |

**10× mais rápido que o Cosmos3 puro** e na mesma ordem do detector. A
amostragem fica densa (≈15 Hz ⇒ ~1,9 m entre observações a 100 km/h, contra
20 m no Cosmos3 puro), o que importa muito sob o protocolo latency-honest.

### Smoke test (4 vídeos — amostra mínima, não conclusiva)

acurácia 57,5% · macro-F1 0,405 · `INSIDE` IoU 0,497 · **eventos `INSIDE`
precisão 100% / recall 100% (4/4)** · `APPROACHING` IoU 0,040.

A execução completa nos 208 vídeos **não** foi feita: o usuário interrompeu
essa linha ("Não! Eu não quero complementar agora! Só perguntei") para não
disputar a GPU com o dump de evidências que estava em curso. O híbrido fica
registrado como hipótese medida em amostra mínima, **não** como resultado.

### Por que isso pode ser o melhor sistema

É a única configuração testada que combina as duas forças medidas em vez de
escolher entre elas: o detector traz precisão de evento (82–86%) e nuisance
baixo; o Cosmos3 traz robustez de percepção fora do domínio. E, ao contrário
da fusão de predições (§2.5, que falhou), aqui a combinação é **arquitetural**
— o verificador entra dentro do laço do detector, não como voto a posteriori.

---

## FASE 6 — Calibração da máquina de estados PARA o Cosmos3-Edge

### O erro que o usuário apontou

> "vc não utilizou os vídeos do roadwork igual vc fez com o 2b para calibrar e
> depois testar nos vídeos não vistos, eu sei disto porque eu lembro que esta
> etapa levou quase um dia pra fazer e vc rodou agora em questão de minutos"

Correto, e é a crítica mais séria feita até aqui. Os limiares publicados
(`N=5, K_ENTER=3, K_EXIT=4, K_FADE=2`), as matrizes do filtro Bayesiano e o
léxico de qualificadores foram todos fixados no split de **calibração** para o
**nosso 2B fine-tunado**. Rodar outro modelo com esses números mede os
hiperparâmetros, não o modelo. Toda a tabela da Fase 2 — inclusive o "colapso
do INSIDE" (IoU 0,007) que eu tratei como propriedade do Cosmos3 — estava
contaminada por isso.

### Como refazer sem gastar um dia de GPU por configuração

Calibrar por força bruta rodando a cascata inteira por configuração é inviável
(centenas de configurações × ~1,9 h). A solução foi separar **evidência** de
**política**:

1. `dump_evidence.py` roda os quatro canais (GATE/SIGN/DESC/EGO) em **todo**
   frame amostrado (stride fixo de 1 s), em vez do agendamento dependente de
   estado que a cascata ao vivo usa, e grava a evidência bruta por ciclo.
   Custo: uma passada de GPU, 100 vídeos de calibração, **3.068 ciclos**.
2. `calibrate_cascade.py` reproduz essa evidência sob qualquer política. Uma
   varredura completa passa a custar **segundos de CPU**.

Ponto de método que quase escapou: a primeira versão do replay era uma
reimplementação à mão da lógica de estados. Isso calibra um sistema e faz
deploy de outro. Refatorei `CascadeStateMachine.update()` extraindo
`update_evidence(gate, corroborated, ego_i, fast_entry)` — a entrada
text-facing agora delega para ela, e o replay chama **exatamente a mesma
máquina que roda ao vivo**. Os números abaixo são todos da versão fiel.

### Resultado da varredura (split de CALIBRAÇÃO, 100 vídeos)

Grade: `N ∈ {3,5,7,9}` × `K_ENTER, K_EXIT, K_FADE ∈ 1..N` × `use_ego` ×
`use_sign`.

| | herdada do 2B | calibrada p/ Cosmos3 |
|---|---|---|
| config | N=5, KE=3, KX=4, KF=2, ego=on | **N=3, KE=3, KX=2, KF=1, ego=off** |
| acurácia | 46,3% | **60,5%** |
| macro-F1 | 0,291 | **0,484** |

Por estado (IoU / recall):

| estado | herdada | calibrada |
|---|---|---|
| OUTSIDE | 0,569 / 59,9% | **0,675 / 74,2%** |
| APPROACHING | 0,241 / 89,3% | **0,294** / 77,8% |
| INSIDE | 0,021 / 2,1% | **0,298 / 34,7%** |
| EXITING | 0,005 / 0,7% | **0,121 / 13,6%** |

### O achado principal: o colapso do INSIDE era artefato de hiperparâmetro

`INSIDE` sai de IoU 0,021 → 0,298 (**14×**) e recall de 2,1% → 34,7%, com
precisão subindo de 39,4% para 67,7%. Ou seja, a conclusão que eu havia
registrado na Fase 2 — "o Cosmos3 lidera APPROACHING mas colapsa em INSIDE" —
**não sobrevive à calibração justa**. A metade da frase sobre APPROACHING se
mantém; a metade sobre INSIDE era eu medindo os limiares do 2B.

Isso muda a leitura do modelo no paper: o Cosmos3-Edge base não é um modelo que
"não sabe estar dentro". Ele precisa de uma janela mais curta (`N=3` em vez de
5) e de saída mais ágil (`K_EXIT=2`), porque sua evidência por frame é menos
ruidosa que a do 2B e não se beneficia de tanto voto majoritário.

### Segundo achado: desligar o canal EGO é melhor

`use_ego=False` vence em **todas** as oito melhores configurações. Isso é
consistente e independente do que já havíamos medido na Fase 1 (ego-relevância
não é alcançável por prompt: todos os prompts ≈ acaso). Aqui a mesma conclusão
aparece por outro caminho — via seleção de política em dados de calibração, sem
olhar respostas do EGO uma a uma. Com `use_ego=False` a emissão do filtro fica
uniforme e o estado passa a ser conduzido só por gate + corroboração; a
transição APPROACHING→INSIDE vira função do tempo de permanência (deriva de
`TRANS`) em vez de um julgamento verbal que o modelo não sabe fazer.

### Mudanças de código

- `cascade_state_machine.py`: `__init__` aceita `N, k_enter, k_exit, k_fading,
  use_ego`; `update_evidence()` extraído de `update()`. Defaults inalterados —
  o pipeline do 2B não muda de comportamento.
- `evaluate_cascade.py`: flag `--csm N=..,k_enter=..,...`.
- `calibrate_cascade.py`: novo, varredura sobre evidência gravada.

### Validação (em andamento)

A config calibrada está sendo avaliada nos **208 vídeos de validação não
vistos**, mesmo protocolo latency-honest, mesmas métricas paper-exact. Os
números da calibração acima **não** entram no paper como resultado — são a
etapa de seleção. Só o que sair da validação é reportável.

### Ressalva de método (registrada antes de ver o resultado)

O dump de evidência usa stride **fixo** de 1 s com todos os canais em todo
frame; a cascata ao vivo usa agendamento dependente de estado e relógio
latency-honest, com ciclo variável (~1–2 s). Logo, a janela `N=3` escolhida
equivale a ~3 s de evidência no dump, mas a uma duração diferente ao vivo. A
varredura seleciona a política sob uma amostragem idealizada e a validação a
executa sob a amostragem real — é justamente por isso que a validação nos 208
vídeos não vistos é o único número reportável. Se o ganho não se transferir,
essa discrepância é a primeira suspeita.

### Bug encontrado na primeira tentativa de validação (e por que quase passou)

A primeira execução da config calibrada nos 208 vídeos previu **OUTSIDE em
100% dos frames** — em todos os 9 vídeos concluídos, inclusive vídeos com
`inside` no ground truth. Interpretei isso inicialmente como "a config
calibrada não transfere". Estava errado, e a sequência de diagnóstico vale
registrar porque três hipóteses plausíveis caíram antes da certa:

1. *A máquina de estados quebrou com `N=3, K_ENTER=3`* — descartado: teste
   unitário com evidência sintética entra em APPROACHING no 3º ciclo.
2. *O runner cacheia o embedding visual pelo caminho do arquivo* (o avaliador
   escreve sempre em `/dev/shm/eval_frame.jpg`) — descartado: escrevendo duas
   imagens diferentes no mesmo caminho, as respostas mudam corretamente.
3. *A config calibrada é a culpada* — descartado: rodando **sem** `--csm` o
   sintoma é idêntico.

Causa real: `evaluate_cascade.py` tem **duas** flags de engine —
`--engine-dir` (reasoner) e `--multimodal-engine-dir` (torre visual, com
default fixo apontando para `jetson-deploy/engines/visual`, a torre do
**nosso 2B**). Eu passei apenas `--engine-dir` para o Cosmos3. O reasoner do
Cosmos3-Edge estava então recebendo tokens visuais produzidos pelo encoder do
2B — features sintaticamente válidas e semanticamente sem sentido. O sintoma é
traiçoeiro porque **não** há erro: o modelo responde com fluência, só que a
resposta é praticamente constante e descorrelacionada da imagem (`GATE='No.'`
em 86/86 ciclos, texto do SIGN idêntico em todos). A consulta direta ao mesmo
frame, com a torre certa, respondia `Yes`.

Com `--multimodal-engine-dir ~/cosmos3edge-engines/visual`: `GATE='Yes'` em
37/56 ciclos, 2 transições de estado, acurácia 55,5% (contra 29,5%) no mesmo
vídeo.

**A Fase 2 foi verificada e NÃO está contaminada.** Reproduzi o vídeo
`boston_042e…07800` com a torre correta e a config **herdada** (idêntica à da
Fase 2): 620 frames `approaching` contra 521 no cache original — mesma ordem,
diferença compatível com a variância de amostragem do protocolo
latency-honest (o descarte de frames depende da latência medida, que varia
entre execuções). A execução com a torre errada dava 901 frames `outside`, um
regime completamente diferente. Ou seja, a Fase 2 usou a torre certa; foram só
as execuções de hoje que herdaram o default.

Lição registrada: um default de CLI silencioso é mais perigoso que uma exceção.
Vale mudar o default de `--multimodal-engine-dir` para derivar de
`--engine-dir` quando este for passado explicitamente.

### RESULTADO DA VALIDAÇÃO — 208 vídeos não vistos

Config calibrada (`N=3, K_ENTER=3, K_EXIT=2, K_FADE=1, ego=off`), selecionada
**apenas** no split de calibração, executada nos 208 vídeos de validação com o
protocolo latency-honest e as métricas paper-exact.

| sistema | acc | macro-F1 | IoU out | IoU appr | IoU inside | IoU exit | inP | inR | entrada MAE | FA/h |
|---|---|---|---|---|---|---|---|---|---|---|
| nosso 2B (sampled) | 62,5% | 0,464 | 0,586 | 0,152 | 0,535 | 0,085 | 54,1% | 96,8% | 2,53 s | 60,1 |
| nosso 2B (greedy) | 62,0% | 0,453 | 0,581 | 0,152 | 0,532 | 0,062 | 56,8% | 97,3% | 2,52 s | 56,2 |
| detector+CLIP | 62,9% | 0,470 | 0,617 | 0,124 | 0,544 | 0,106 | 79,9% | 95,7% | 2,52 s | **30,9** |
| detector +textFE | 62,9% | 0,473 | 0,594 | 0,140 | 0,555 | 0,104 | **81,9%** | 95,1% | 2,42 s | 52,3 |
| Cosmos3 herdada (Fase 2) | 43,6% | 0,290 | 0,609 | 0,233 | 0,007 | 0,006 | 38,9% | 3,8% | 2,33 s | 46,1 |
| **Cosmos3 CALIBRADO** | **65,5%** | **0,515** | 0,615 | **0,278** | **0,574** | 0,071 | 71,5% | 87,0% | **2,15 s** | 64,6 |

O ganho **transferiu** para dados não vistos, e mais forte que na calibração:
macro-F1 0,290 → **0,515**, `INSIDE` IoU 0,007 → **0,574** (82×), recall de
`INSIDE` 3,8% → **87,0%**.

#### O que isso significa para o paper

O Cosmos3-Edge INT4 **zero-shot**, com a máquina de estados calibrada, é o
melhor sistema medido em acurácia, macro-F1, IoU de `APPROACHING`, IoU de
`INSIDE` e erro de timing de entrada — superando tanto o detector portado
quanto o nosso 2B **fine-tunado no próprio ROADWork**. A tese do paper muda: o
gargalo medido não era capacidade de percepção do modelo nem falta de
fine-tuning; era (a) supervisão de ego-relevância e (b) hiperparâmetros de
temporalização casados ao modelo errado.

#### Onde ele continua perdendo (não esconder)

1. **Nuisance**: 64,6 FA/h, o pior da tabela — mais que o dobro do
   detector+CLIP (30,9). Consistente com o affirmation bias medido na Fase 0
   (3/6 falsos positivos em negativos puros).
2. **Precisão de `INSIDE`**: 71,5% contra 81,9% do detector+textFE. Ele troca
   precisão por recall (87,0% vs 95,1% — aqui perde nos dois, mas o salto
   partiu de 3,8%).
3. **`EXITING`**: IoU 0,071, abaixo do detector (0,106). Saída continua sendo o
   estado mais fraco de todos os sistemas — o ground truth de `exiting` é curto
   e a evidência de gate desaparece devagar.
4. **Custo**: continua um VLM no laço (~1,4 Hz), contra ~23 Hz do detector.
   O argumento de trade-off compute do paper atual permanece válido.

#### Protocolo (para o camera-ready)

Calibração e teste são **disjuntos**: os limiares vieram de 100 vídeos do split
de calibração (3.068 ciclos de evidência gravada) e nunca viram os 208 vídeos
de validação. Isso reproduz para o Cosmos3-Edge exatamente a etapa que já havia
sido feita para o nosso 2B — e que, sem ela, tornava a comparação da Fase 2 uma
medida dos hiperparâmetros do 2B, não do modelo.

### Fusão, revisitada com o Cosmos3 calibrado

A §2.5 concluiu que fusão em nível de predição não funciona — mas aquela busca
usou o cache do Cosmos3 com a config herdada (`INSIDE` IoU 0,007). Refeita com
o cache calibrado, mesmo protocolo honesto (validação partida ao meio: seleção
no `dev`, reporte no `test`, `rng` semente 0):

**Seleção no dev por macro-F1 puro** → vencedor `c3e only` (F1 0,531).
No teste: F1 **0,498** contra 0,478 do detector. Ou seja, com o Cosmos3
calibrado a conclusão da §2.5 se inverte parcialmente: agora o melhor sistema
único **é** melhor que o detector, mas a fusão ainda não bate o melhor único
sob esse critério.

**Sob restrição de nuisance**, porém, aparece a melhor configuração da sessão:

| estratégia (TEST) | acc | macro-F1 | IoU appr | IoU inside | inP | inR | FA/h |
|---|---|---|---|---|---|---|---|
| det only | 64,5% | 0,478 | 0,136 | 0,562 | **82,7%** | 94,7% | 30,3 |
| detFE only | 64,0% | 0,478 | 0,150 | 0,571 | 84,7% | 94,7% | 52,8 |
| c3e only | 63,8% | 0,498 | 0,258 | 0,547 | 70,6% | 85,3% | 64,0 |
| **det gate + c3e state** | **64,9%** | **0,503** | **0,276** | 0,530 | 74,1% | 85,3% | **30,3** |

`det gate + c3e state` = o **detector** decide se há zona ativa (portão), o
**Cosmos3** decide qual estado. Fica com o melhor macro-F1 do teste **e** com o
nuisance do detector (30,3 FA/h, metade do c3e sozinho), porque o portão nunca
abre onde o detector não vê nada — exatamente onde está o affirmation bias do
VLM. É a combinação arquitetural que a §2.5 procurou e não achou; ela só
aparece depois que o canal de estado do Cosmos3 passa a funcionar.

> **Ressalva de honestidade (importante para o camera-ready):** essa estratégia
> **não** venceu a seleção no `dev` por macro-F1 puro (0,511 contra 0,531 do
> `c3e only`). Ela só vence sob um critério com restrição de nuisance
> (`max F1 s.a. FA/h ≤ 32`), e esse critério foi formulado **depois** de ver os
> resultados. A restrição em si é defensável a priori — o paper já trata FA/h
> como métrica de primeira linha e o argumento de implantação depende dela —
> mas o limiar específico não é. Antes de reportar isso como resultado
> principal, o correto é fixar o critério e **re-selecionar num split de
> desenvolvimento que não seja este**. Registrado como candidato forte, não
> como número final.

---

## FASE 4 — Plano de reescrita do paper

### O que os novos resultados contradizem no texto atual

| §atual | afirmação atual | evidência nova |
|---|---|---|
| `sec:analysis` | *"This is a **fine-tuning coverage** problem, not an architecture problem"* | **Falso como escrito.** O Cosmos3-Edge — que percebe a cena corretamente (92% OOD, bboxes corretas, descrições precisas) — falha na ego-relevância exatamente do mesmo jeito (§Fase 1). Não é cobertura do nosso fine-tuning; é uma limitação de *prompting* como interface. |
| `sec:limitations` | *"a specialized small VLM as competitive on correctness while behind on precision"* | O melhor sistema agora é um VLM **não especializado** (zero-shot) que **vence** em acc, macro-F1 e IoU de `APPROACHING`/`INSIDE`. |
| `sec:conclusion` | *"the pragmatic path is to **add** language-model evidence to a detector, not replace it"* | Direção **confirmada e fortalecida**, mas por outro motivo: o melhor candidato (`det gate + c3e state`) usa o detector como portão de nuisance e o VLM como estimador de estado — não o VLM como fast-entry do detector. |
| `sec:limitations` (iv) | limitação sobre isolar efeitos de protocolo | acrescentar a limitação nova e mais séria: **portabilidade de hiperparâmetros**. |

### Contribuições reformuladas

1. **Ego-relevância não é alcançável por prompting** — demonstrado em dois
   modelos de famílias e escalas diferentes, um deles um world model de
   physical-AI com percepção comprovadamente boa. Isso promove o achado de
   "limitação do nosso checkpoint" para **afirmação sobre a interface**.
2. **Hiperparâmetros de temporalização não são portáveis entre modelos** — e
   ignorar isso produz conclusões qualitativamente erradas. Demonstração
   quantitativa: macro-F1 0,290 → 0,515 e `INSIDE` IoU 0,007 → 0,574 no mesmo
   modelo, mesmos vídeos, mudando só os limiares. Um leitor que visse só a
   Fase 2 concluiria "o Cosmos3 não sabe reconhecer estar dentro de uma obra",
   o que é falso.
3. **Um world model quantizado, zero-shot, supera sistemas ajustados ao
   domínio** em hardware embarcado — e o protocolo latency-honest continua
   sendo o que torna a comparação legítima.
4. **Protocolo de calibração barato** (dump de evidência + replay) que torna a
   calibração por modelo viável: segundos de CPU em vez de horas de GPU por
   configuração.

### O que se mantém intacto

Protocolo latency-honest, métricas paper-exact, o split, a análise de nuisance,
a caracterização de custo/energia, e a tese de que somar evidência de linguagem
a um detector é o caminho pragmático.

### Ordem de execução

1. Re-seleção honesta da estratégia num split de desenvolvimento disjunto
   (100 vídeos de calibração), critério fixado **antes**: `max macro-F1 s.a.
   FA/h ≤ FA/h do detector no mesmo split`. → define o sistema principal.
2. Fase 3: OOD Califórnia formalizado, 39 sinais anotados × 3 sistemas.
3. Reescrita: tabela principal, `sec:analysis`, `sec:limitations`, conclusão,
   título e abstract.

---

## FASE 7 — Re-seleção honesta do sistema principal

A §"Fusão, revisitada" deixou pendente exatamente o vício que ela mesma
apontou: a estratégia `det gate + c3e state` só vencia sob uma restrição de
nuisance formulada **depois** de ver os resultados, num split que era metade da
validação. Refeito corretamente.

### Protocolo

- **Conjunto de desenvolvimento**: os 100 vídeos do split de **calibração**
  (`eval_split_dev.json`), disjunto da validação e nunca reportado. Rodados de
  ponta a ponta com os quatro sistemas (Cosmos3 calibrado, detector+CLIP,
  detector+textFE, nosso 2B) sob o mesmo protocolo latency-honest — sequencial,
  porque rodar dois na mesma GPU distorceria o descarte de frames dos dois.
- **Critério fixado antes** de olhar qualquer número do dev, escrito no
  cabeçalho de `select_strategy.py`:

  > maximizar macro-F1 sujeito a `FA/h ≤ FA/h do detector no mesmo split`.

  Justificativa registrada junto: o paper já trata nuisance como métrica de
  primeira linha, e o detector é o sistema em produção — "não piorar o nuisance
  do que já está implantado" é a barra que um substituto precisa passar.
- **Conjunto de candidatos fixado** (as 11 estratégias de `fuse_systems.py`).
  Mantive as variantes com `detFE` na fila mesmo suspeitando que não passariam
  na restrição: descartar um candidato porque eu previa que perderia é a mesma
  falha que este passo existe para corrigir.

### Seleção (DEV, 100 vídeos)

| estratégia | acc | macro-F1 | IoU appr | IoU inside | inP | inR | FA/h |
|---|---|---|---|---|---|---|---|
| 2B only | 62,1% | 0,462 | 0,179 | 0,466 | 52,4% | 89,9% | 40,8 |
| det only | 66,8% | 0,499 | 0,143 | 0,516 | **85,8%** | 85,3% | **16,3** |
| detFE only | 66,1% | 0,465 | 0,098 | 0,501 | 80,3% | 91,7% | 67,5 |
| c3e only | 66,2% | 0,517 | **0,302** | 0,521 | 73,5% | 76,1% | 78,0 |
| c3e entry + det inside | 62,9% | 0,487 | 0,167 | 0,521 | 75,0% | 89,9% | 75,7 |
| c3e gate + det state | 64,0% | 0,474 | 0,167 | 0,522 | 84,2% | 85,3% | 78,0 |
| **det gate + c3e state** | **68,3%** | **0,521** | 0,288 | **0,523** | 74,6% | 73,4% | **16,3** |

Elegíveis sob `FA/h ≤ 16,3`: apenas `det only` e `det gate + c3e state`.
**Selecionado: `det gate + c3e state`.**

### Reporte (VALIDAÇÃO, 208 vídeos, nunca usados na seleção)

| sistema | acc | macro-F1 | IoU appr | IoU inside | inP | inR | FA/h |
|---|---|---|---|---|---|---|---|
| nosso 2B (greedy) | 62,0% | 0,453 | 0,152 | 0,532 | 56,8% | 97,3% | 56,2 |
| detector+CLIP | 62,9% | 0,470 | 0,124 | 0,544 | 79,9% | 95,7% | 30,9 |
| detector +textFE | 62,9% | 0,473 | 0,140 | 0,555 | **81,9%** | 95,1% | 52,3 |
| Cosmos3 calibrado sozinho | **65,5%** | **0,515** | 0,278 | **0,574** | 71,5% | 87,0% | 64,6 |
| **det gate + c3e state** (selecionado) | 65,0% | 0,507 | **0,280** | 0,553 | 72,7% | 87,0% | **30,9** |

### Por que essa é a arquitetura certa, e não só a que ganhou

O detector decide **se** existe zona ativa; o world model decide **qual**
estado. Cada componente é usado onde é forte e neutralizado onde é fraco: o
nuisance do sistema passa a ser exatamente o do detector (30,9 FA/h, idêntico —
o portão nunca abre onde o detector não vê nada, que é precisamente onde mora o
affirmation bias do VLM), enquanto a estimativa de estado fica com o modelo que
tem o melhor IoU de `APPROACHING` (0,280 vs 0,124) e de `INSIDE`.

Ganho sobre o detector em produção: macro-F1 0,470 → **0,507** e IoU de
`APPROACHING` 0,124 → **0,280** (2,3×), **a custo zero de nuisance**. O preço é
recall de `INSIDE` (95,7% → 87,0%) e compute (o VLM entra no laço).

### Nota sobre `c3e only`

O Cosmos3 sozinho tem macro-F1 mais alto (0,515 vs 0,507) e o melhor IoU de
`INSIDE` (0,574). Ele **não** foi selecionado porque dobra o nuisance (64,6 vs
30,9 FA/h) e o critério pré-registrado exigia não piorar esse número. Reportar
`c3e only` como sistema principal seria escolher a métrica depois do resultado
— justamente o que a Fase 6 mostrou dar errado. Ambos ficam na tabela.

---

## FASE 3 — Benchmark OOD Califórnia, formalizado

Nota de correção: os "25 sinais" de rodadas anteriores eram **cobertura
parcial**, não limite de anotação. O arquivo `Speed_Sign_Timestamps.txt` tem 40
timestamps, um marcado "Impossible" → **39 sinais avaliáveis**, e os 14 vídeos
existem todos em disco. Os três avaliadores já usavam o mesmo parser e a mesma
janela (6 s, passo 1 s), então bastou executá-los na íntegra.
Consolidação em `ood_california_table.py`.

Detecção = `GATE` afirmativo **ou** transcrição de placa com palavra-chave de
obra, em qualquer amostra da janela de aproximação. Mesmos 39 sinais para os
três sistemas (pareado).

| sistema | geral | dia (15) | entardecer (2) | pôr-do-sol (4) | noite (9) | noite+chuva (4) | noite+névoa (5) |
|---|---|---|---|---|---|---|---|
| nosso 2B (fine-tunado) | **2/39 = 5%** | 13% | 0% | 0% | 0% | 0% | 0% |
| detector+CLIP | 28/39 = 72% | 93% | 100% | 100% | 44% | 50% | 40% |
| **Cosmos3-Edge INT4 (zero-shot)** | **37/39 = 95%** | 100% | 100% | 100% | **100%** | 75% | 80% |

### Leitura

1. **O fine-tuning comprou desempenho em domínio e destruiu generalização.**
   O nosso 2B, o melhor sistema puro-VLM em ROADWork, detecta 5% dos sinais
   numa gravação nossa noutro estado — e **zero** em qualquer condição que não
   seja dia. Isso é mais forte que "degradação": é colapso.
2. **O detector é robusto de dia e frágil à noite** (93% → 40–50%), o que
   bate com a intuição de que cor/forma de cone e barril dependem de
   iluminação.
3. **O world model quantizado generaliza sem ter visto nada disto**: perfeito
   em dia, entardecer, pôr-do-sol e noite; cede só em chuva (75%) e névoa
   (80%), que são também as condições onde a placa fica fisicamente ilegível.

Isso fecha a pendência de revisor sobre night/weather **com dados próprios**,
e dá a justificativa empírica para a arquitetura escolhida na Fase 7: o
componente que generaliza é o que decide o estado.

---

## FASE 4 — Reescrita executada

### Mudanças no `main.tex`

| item | o que mudou |
|---|---|
| título | → *"Evidence-First VLM Cascades for Embedded Work-Zone State Estimation: Hyper-Parameter Portability and the Cost of Specialization"* |
| abstract | reescrito como narrativa única (estava virando dois abstracts colados: tese antiga + achados novos) |
| introdução | realinhada; contribuições passam de 4 para 6 |
| `sec:related` | novo parágrafo sobre world models para physical AI |
| `tab:main` | duas colunas novas: `c3e` (world model calibrado) e `+gate` (híbrido selecionado); linha de FA/h acrescentada; coluna `(pap)` removida da tabela e explicada no texto |
| `sec:worldmodel` | **seção nova**: deploy INT4, não-portabilidade de hiperparâmetros, OOD, híbrido, trajetória |
| `tab:ood` | **tabela nova**: 39 sinais × 3 sistemas |
| `sec:analysis` | reescrita — o gap deixa de ser "cobertura do fine-tuning" e passa a ser limitação da interface |
| `sec:limitations` | 4 → 5 limitações; entra portabilidade de hiperparâmetros e a limitação do próprio critério de seleção |
| `sec:conclusion` | reescrita |

### Citações que faltavam (achado do usuário)

O modelo novo era mencionado **7 vezes sem uma única citação**. Adicionadas:

- `nvidia2026cosmos3` — *Cosmos 3: Omnimodal World Models for Physical AI*,
  NVIDIA, arXiv:2606.02800 (2026)
- `alpamayo2026` — *Alpamayo-R1: Bridging Reasoning and Action Prediction for
  Generalizable Autonomous Driving in the Long Tail*, NVIDIA, arXiv:2511.00088

### Inconsistências encontradas na revisão e corrigidas

1. **Anáfora quebrada**: *"Answering **it** requires more than swapping
   models"* — o "it" apontava para uma pergunta a dois parágrafos de distância
   depois que inseri o parágrafo novo.
2. **Afirmação obsoleta**: o texto dizia que o `+FE` "wins the most rows of the
   three deployed configurations", mas a tabela passou a ter mais colunas e
   quem ganha a maioria das linhas agora é o world model.
3. **Fecho do abstract** ainda descrevia 3,2–10,6 Hz / 37 W como se o 2B fosse
   o sistema recomendado, o que deixou de ser verdade.
4. **`\emph{pap}` órfão**: a coluna foi removida da tabela mas o texto ainda a
   referenciava.

### Estado de tamanho — PENDÊNCIA REAL

Corpo em **~8,6 páginas** contra o limite de 8 do WACV. Já foram removidos:
a figura qualitativa (backup em `/tmp/fig_qualitative_removed.tex`), a
`tab:latency` (backup em `/tmp/tab_latency_removed.tex`) e ~6k caracteres de
prosa em 7 seções. Cortar mais atinge conteúdo que revisores pediram
(ablações) ou o teaser. **Decisão pendente do usuário.**

### Material suplementar (`supp.tex`) — atualizado

Estava de 20/07, anterior a toda a reformulação. Mudanças:

**Corrigido (estava enganoso):**
- `supp:bayes` apresentava $N{=}5, K_\text{enter}{=}3, K_\text{exit}{=}4, K_\text{fade}{=}2$
  como se fossem constantes do método. Agora diz explicitamente que são
  calibradas **para o nosso 2B** e que não devem ser herdadas por outro modelo.
- `supp:repro` não listava nenhum dos artefatos novos.

**Seções novas:**
1. `supp:calib` — engines INT4 do Cosmos3-Edge (arquitetura do reasoner, mRoPE,
   grid 26×46), os seis patches no runtime, a regressão 8/8 do pipeline do 2B,
   a tabela pareada herdada-vs-calibrada na calibração (IoU `INSIDE` 0,021 →
   0,298), o mecanismo de dump+replay, e **a ressalva de amostragem** (dump a
   1 s uniforme vs. cadência real dirigida por latência).
2. `supp:select` — o protocolo de seleção em quatro passos, na ordem em que
   foi executado, incluindo o registro de que sob macro-F1 irrestrito o
   vencedor seria o world model sozinho.
3. `supp:ood` — as 14 gravações, os 39 sinais (40 menos 1 ilegível), a regra
   de decisão, o aviso de que isso mede detecção de placa e **não** estimativa
   de quatro estados, e a validação de trajetória por GPS com a justificativa
   de integrar velocidade em vez de diferenciar posição.

Quatro ponteiros para essas seções foram inseridos no `main.tex`.
Suplemento compila em 5 páginas, sem referências indefinidas.

### Figuras — atualizadas

**`fig_teaser.png` (refeita tres vezes).** A original mostrava so o nosso 2B —
um revisor abrindo a pagina 2 recebia a tese que o paper agora refuta. As duas
primeiras tentativas de conserto foram rejeitadas pelo usuario, com razao:

1. **Frame in-domain com o EGO divergindo.** Mostrava o Cosmos3 errando e o 2B
   acertando — contradizia a manchete do paper. Pior: destacava um canal que a
   configuracao calibrada **desliga**. E o argumento que usei ("ilustra que
   ego-relevancia nao e alcancavel por prompt") nao se sustentava, porque a
   figura mostrava apenas *um* modelo falhando.
2. **Frame noturno com transcricao de placa de velocidade.** O usuario apontou
   que "SPEED LIMIT 55" nao contribui em nada para saber que ha obra.
   Correto. A verificacao que isso motivou trouxe uma boa noticia — "SPEED
   LIMIT" **nao** esta em `SIGN_WZ_KEYWORDS`, entao o benchmark OOD nao esta
   inflado por placas de velocidade comuns; aquele caso contou pelo canal GATE.
   Mas confirmou que a figura destacava o canal errado. Uma varredura
   subsequente sobre os frames noturnos/entardecer nao achou **nenhuma**
   transcricao com palavra-chave de obra: naquelas gravacoes as deteccoes vem
   todas do GATE.

**Versao final** — frame diurno, `CA50_Day_01` @0:23, onde os cones laranja
atravessando as faixas e a barreira de concreto sao visiveis a olho nu:

| canal | nosso 2B (fine-tunado em obras) | Cosmos3-Edge zero-shot |
|---|---|---|
| GATE | `No.` | `Yes.` |
| SIGN | "No temporary traffic control devices visible" | "no visible temporary signs; a green highway sign reading EXIT ONLY" |
| **DESC** | **"No work zone elements visible in this scene."** | **"A concrete barrier on the right side, and orange traffic cones placed along the edge of the highway."** |

O criterio que faltava nas duas primeiras tentativas: a evidencia precisa ser
**verificavel pelo leitor na propria foto**. Aqui esta — os cones e a barreira
estao la, o modelo zero-shot os nomeia, e o modelo que treinamos em zonas de
obra afirma que nao ha nenhum.

**`fig_cascade` (legenda corrigida).** Cravava "gate ativo em ≥3 de 5 ciclos"
como se fosse constante do método — exatamente a afirmação que o paper agora
refuta. Passou a $\geq K_\text{enter}$ de $N$, com os dois conjuntos citados
(5/3/4/2 para o 2B, 3/3/2/1 para o world model) e a observação de que o que
transfere é a estrutura, não os números.

**`fig:qualitative` (removida)** por espaço; backup em
`/tmp/fig_qualitative_removed.tex`.

---

## Passe de coerência estrutural (crítica do usuário: "é um paper, não ir adicionando coisas por último")

Li o texto inteiro antes de editar. Problemas **estruturais** encontrados:

1. **Contradição interna.** A §5.3 afirmava *"Fine-tuning, not prompting, does
   the work"* — exatamente o que a §6 refuta. Virou pergunta ("Does fine-tuning
   do the work?"), com o confundidor exposto: o controle troca o checkpoint mas
   herda os limiares ajustados ao checkpoint antigo, então mede o par, não o
   modelo.
2. **Rótulo duplicado.** `\label{sec:results}` aparecia na seção *e* na
   subseção; toda referência cruzada resolvia para o lugar errado.
3. **Tabela fora de ordem.** A `tab:main` trazia colunas do Cosmos3 na §5,
   antes de o modelo existir no texto — o sintoma exato que o usuário apontou.
   Separada: `tab:main` (2B vs detector) na §5, `tab:worldmodel` (herdado vs
   calibrado) na §6, cada uma junto da sua discussão.
4. **Referência órfã** à figura qualitativa removida.
5. **Título da §6** ("Does the Cascade Need a Specialized Model?") soava
   apêndice → "Varying the Model: Calibration and Generalization".
6. **Pontes narrativas**: a §5 agora fecha explicando por que é insuficiente,
   a §6 abre retomando o confundidor, e a introdução anuncia o arco.

### Nada foi deletado — foi movido

Por instrução do usuário ("lembra que temos o supplementary"):

| conteúdo | destino |
|---|---|
| figura qualitativa | `supp:qual` (tinha sido deletada; recuperada) |
| tabela de ablações + análise por toggle | `supp:ablations` (corpo mantém os dois achados que importam) |
| currículo de fine-tuning detalhado | `supp:curriculum` |
| tabela de latência do corpo | já era subconjunto de `supp:tab:latency`, que tem p50/p95/p99/max e energia por decisão — nada se perdeu |

### Tamanho

Corpo em **~8,6 páginas** (limite 8). Compressões feitas sem perder conteúdo:
Related Work, canais de prompt, deployment, protocolo, análise, introdução e as
três legendas mais longas. Restam ~0,6 página de estouro.

---

## Resposta ao review externo

### Item 1 — contradição de FA/h (o mais grave)

`tab:main` dizia 56,2 FA/h para a coluna *rec.*; a §5.2 dizia 37,7. Rastreei:
a célula trazia o valor do greedy **sem** debounce numa coluna cujas outras
células (precisão 76,7%, recall 96,8%, MAE 4,16 s) já eram as **com** debounce.
Recomputado dos caches: 0 s → 56,2; 1 s → **37,7**; 2 s → 27,5, batendo com o
texto. Corrigido em `tab:main` e `tab:worldmodel`. A conclusão não muda —
37,7 continua acima dos 30,9 do detector.

### Item 6 — e uma afirmação FALSA que o revisor não pegou

O paper afirmava que a vantagem do 2B em IoU de `APPROACHING` tinha
*"paired 95% CI excludes zero"*. Com 10.000 reamostragens no nível de vídeo o
intervalo é **[−0,005, +0,058]** — **contém zero** (p unilateral 0,049).
Afirmação reescrita. Em contrapartida a vantagem do C3E é decisiva:
**+0,154, IC [+0,118, +0,190]**; a de `INSIDE` não se resolve
(+0,030, [−0,009, +0,067]) e isso agora está dito. Harness em `paired_ci.py`.

| par (validação, 208) | acc | macro-F1 | IoU appr | prec. INSIDE | FA/h |
|---|---|---|---|---|---|
| 2B − detector | n.s. | n.s. | n.s. | −0,231 * | +25,3 * |
| C3E − detector | +0,026 * | +0,044 * | +0,154 * | −0,084 * | +33,7 * |
| C3E − 2B | +0,035 * | +0,062 * | +0,126 * | +0,147 * | n.s. |

### Item 3 — experimento pareado (a objeção mais forte)

O revisor pediu: *"same checkpoint before and after fine-tuning, with
independent temporal calibration for both"*. Feito, no split de calibração
(100 vídeos), com os engines do checkpoint base:

| | acurácia | macro-F1 |
|---|---|---|
| base, limiares herdados | 53,2% | 0,281 |
| base, **calibrado para ele** | 55,3% | 0,383 |
| fine-tunado, limiares herdados | 57,7% | 0,451 |
| fine-tunado, **calibrado para ele** | **64,1%** | **0,496** |

Família, escala, arquitetura e torre visual idênticas; varia só o currículo.
Com cada um usando os próprios limiares, o fine-tuning vale **+0,113 de
macro-F1 e +8,8 pp** — real, mas metade do que o controle sem calibração
sugeria. A §5.3 mudou de *"a conclusão não sobrevive"* para *"sobrevive, com
metade do tamanho aparente"*.

### Simetria da varredura (pendência que eu mesmo achei)

A §6 afirmava *"we apply the same search procedure to both models"* — e isso
**não era verdade** quando foi escrito. Agora é. Na validação:

| | macro-F1 herdado | macro-F1 recalibrado |
|---|---|---|
| 2B fine-tunado | 0,453 | 0,459 |
| Cosmos3-Edge | 0,290 | 0,515 |

A assimetria é o argumento: uma busca que apenas premiasse flexibilidade teria
ajudado os dois igualmente.

### Item 2 — decomposição do gap do baseline (em curso)

Não dá para rodar o split original. Mas o pedido era **decompor** o gap
0,600 − 0,470 = **+0,130**:

- **split** (medido): mesmo port, mesmo protocolo, split de calibração →
  0,499 contra 0,470 na validação. **+0,029, ≈22% do gap.**
- **protocolo** (rodando): modo `--dense` novo em
  `evaluate_yolo_baseline.py`, que avança o relógio por um período de frame em
  vez da latência medida. Previsão: contribuição pequena, porque o detector a
  43 ms contra 33 ms de intervalo já vê ~77% dos frames.
- **implementação**: o resíduo, quantificado por diferença.

---

## O sistema recomendado nunca tinha sido executado (achado grave)

O `det gate + c3e state` reportado no paper era uma combinação **post-hoc** de
dois arrays de predições em cache (`fuse_systems.s_gated`), não um sistema. Sob
replay latency-honest o detector (43 ms) e o C3E (160–650 ms) observam frames
**diferentes**; um pipeline realmente encadeado observa um terceiro conjunto.

Implementado de verdade (`evaluate_gated_hybrid.py`) e medido em 123 vídeos:

| | post-hoc | medido | |
|---|---|---|---|
| acurácia | 0,661 | 0,453 | −0,207 |
| macro-F1 | 0,517 | 0,409 | −0,107 |
| IoU appr | 0,277 | 0,200 | −0,076 |
| IoU inside | 0,559 | 0,465 | −0,094 |
| precisão INSIDE | 0,697 | 0,766 | **+0,069** |

**A afirmação do paper não se sustenta**: 0,409 é pior que o detector sozinho
(0,470). Isso é uma segunda instância da tese central — atalhos de avaliação
produzem conclusões erradas — e desta vez o atalho era nosso.

Perfil operacional medido (que era o pedido do revisor): ciclo p50 51 ms com o
portão fechado, p95 1369 ms com ele aberto, 4,4 Hz efetivos, duty cycle do C3E
29,5%.

### Variante v2 — hipótese registrada ANTES de rodar

A precisão de `INSIDE` **subiu** (+0,069) enquanto IoU e cobertura caíram: essa
é assinatura de **sub-observação**, não de decisão ruim. O sistema acerta quando
dispara, mas vê frames de menos. Causa no perfil: com o portão aberto o ciclo
paga detector (43 ms) + C3E gate (162 ms) + C3E desc (488 ms) ≈ 700 ms, e o
DESC sozinho é ~70% disso.

**Variante (uma só, escolhida por raciocínio antes da execução):** com o portão
aberto, não pagar o VLM por evidência que o detector já fornece — o C3E roda só
o GATE, e a corroboração vem do próprio portão que abriu. Ciclo ativo cai de
~700 ms para ~205 ms. Junto, preservar a máquina de estados entre fechamentos
de portão (resetar foi escolha arbitrária da v1, e o detector oscila).

**Previsão registrada:** se sub-observação for a causa dominante, a v2 recupera
cobertura (IoU inside e macro-F1 sobem) e possivelmente perde um pouco de
precisão de `INSIDE`. Se a v2 **não** melhorar, a conclusão é que o portão não
sobrevive ao custo de encadear, e reportamos o negativo.

Compromisso metodológico: **uma** variante. Se ela falhar, não haverá uma v3
buscando um vencedor — isso seria exatamente o vício que o paper denuncia.
