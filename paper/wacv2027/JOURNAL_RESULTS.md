# Pendências experimentais fechadas para a versão de revista (2026-09-28)

Tudo medido no Jetson AGX Orin desta máquina, split de validação de 208 vídeos
(`pipeline/eval_split_full.json`), métricas paper-exatas de `pipeline/paper_metrics.py`.
Nenhum número aqui depende de re-treino nem do A100.

## 0. Ponto de partida: o sistema que não estava no manuscrito

`main.tex` (28/07) conclui que *"the obvious ways of combining the two did not
resolve that trade-off"*. O estimador conjunto de `pipeline/joint_filter.py`
(30/07) é posterior ao texto e é uma execução encadeada real sob replay
latency-honest — não a combinação post-hoc de caches que já produziu um fantasma
de +0,107 macro-F1 neste projeto.

## 1. Qual conjunto de parâmetros reportar (pendência: `joint_params_full` nunca avaliado)

Replay exato sobre a evidência gravada (`pipeline/replay_joint.py`; reproduz
`joint_v2_val` célula por célula, o que valida o harness):

| ajuste | acc | macro-F1 | IoU appr | IoU in | IoU exit | prec. IN | rec. IN | FA/h |
|---|---|---|---|---|---|---|---|---|
| 100 vídeos de calibração | 65,5% | **0,550** | 0,262 | 0,569 | 0,163 | 76,3% | 89,7% | 33,2 |
| **312 vídeos (todo o split)** | **66,1%** | 0,546 | 0,246 | **0,578** | 0,160 | 66,8% | 85,9% | **25,9** |

As duas são estatisticamente indistinguíveis em qualidade (macro-F1 −0,004,
IC [−0,017, +0,008]); o ajuste de 312 tem FA/h menor (−7,3, IC [−14,4, −0,5]).
**Reportar o de 312**: usa todo o split de calibração pelo mesmo procedimento,
é escolha a priori, e é o único que satisfaz o critério de seleção que este
projeto já adotava (maximizar macro-F1 com FA/h não pior que os 30,9 do
detector implantado) — o ajuste de 100 vídeos o viola com 33,2.

## 2. Intervalos de confiança pareados (pendência: o sistema novo não tinha nenhum)

10.000 reamostragens no nível de vídeo, mesmos índices para os dois sistemas
(`pipeline/paired_ci_fast.py` — mesmo estimador de `paired_ci.py`, reduzido a
estatísticas suficientes por vídeo; o par `C3E − det` reproduz os IC já
registrados, o que valida a reescrita). `*` = IC exclui zero.

| joint 312 − | acc | macro-F1 | IoU appr | IoU in | IoU exit | prec. IN | FA/h |
|---|---|---|---|---|---|---|---|
| detector+CLIP | +0,032 * | **+0,076 \*** | +0,122 * | +0,034 * | **+0,053 \*** | −0,131 * | −5,1 n.s. |
| C3E calibrado | +0,006 | +0,031 * | −0,032 | +0,004 | **+0,089 \*** | −0,047 | **−38,8 \*** |
| 2B rec. | +0,041 * | +0,093 * | +0,094 * | +0,046 * | +0,098 * | +0,100 * | −30,4 * |

O trade-off que o manuscrito declara irresolvido **resolve-se in-domain**: ganha
do detector em acurácia, macro-F1 e nos três IoUs, com FA/h estatisticamente
empatada. O custo é precisão de evento `INSIDE` (−0,131). E é o primeiro sistema
a mover `EXITING`, que todos os do paper deixam em 0,07–0,11.

## 3. Perfil operacional (pendência aberta desde a crítica ao híbrido com portão)

`pipeline/profile_joint.py`, recuperado dos tempos gravados na própria corrida:

- detector dita o relógio e nunca é bloqueado: p50 44 ms, p95 59 ms → **22,1 Hz**
- world model assíncrono, repolled assim que responde: **0,62 Hz** (p50 1538 ms entre respostas)
- 35,5 atualizações de estado por resposta do world model; a idade da resposta
  entra na observação
- 137.583 ciclos sobre 103,7 min de condução simulada

## 4. Ablação de canais (pendência: a alegação de emissão conjunta nunca foi testada)

`pipeline/ablate_joint.py` — refit na calibração **e** reavaliação na validação
com o canal removido nos dois lados, então mede um sistema construído sem o
canal, não um sistema surpreendido pela ausência dele.

| variante | acc | macro-F1 | IoU appr | IoU in | IoU exit | prec. IN | FA/h |
|---|---|---|---|---|---|---|---|
| completo | 66,1% | 0,546 | 0,246 | 0,578 | 0,160 | 66,8% | 25,9 |
| sem detector | 64,2% | 0,460 | 0,211 | 0,560 | **0,017** | 69,3% | 42,7 |
| sem GATE | 67,1% | **0,558** | 0,268 | 0,569 | 0,173 | 67,9% | 32,0 |
| sem SIGN | 66,1% | 0,546 | 0,245 | 0,578 | 0,160 | 68,0% | 25,3 |
| sem DESC | 64,8% | 0,530 | 0,210 | 0,564 | 0,157 | 76,7% | 29,8 |
| **sem world model** | 67,3% | 0,536 | 0,233 | 0,561 | 0,148 | 76,5% | **18,5** |
| sem idade | 66,8% | 0,547 | 0,254 | 0,592 | 0,150 | 76,2% | 23,6 |

**Achado que precisa ser reportado**: o ganho in-domain é quase todo do *filtro*,
não da fusão. O detector sozinho dentro do mesmo estimador contado dá 0,536 —
contra 0,470 da máquina de estados ajustada à mão do mesmo detector — e o world
model acrescenta só +0,010, com FA/h pior (25,9 contra 18,5) e precisão pior
(66,8% contra 76,5%). O canal GATE chega a atrapalhar in-domain (0,558 sem ele).
A decomposição honesta é:

| | macro-F1 |
|---|---|
| detector + máquina de estados ajustada à mão (paper) | 0,470 |
| detector + filtro contado, sem world model | 0,536 |
| detector + world model + filtro contado | 0,546 |
| world model + cascata calibrada (paper) | 0,515 |

O canal DESC é o único canal do world model que se paga in-domain (−0,016 de
macro-F1 ao remover), e o canal SIGN não muda nada — coerente com a varredura
que não achou nenhuma transcrição com palavra-chave de obra nos frames noturnos.

## 5. Estabilidade por cidade

| sistema | Boston (154) | Seattle (54) |
|---|---|---|
| detector+CLIP | 0,469 | 0,462 |
| 2B rec. | 0,446 | 0,462 |
| C3E calibrado | 0,518 | 0,495 |
| **joint 312** | **0,551** | **0,519** |

Melhor nas duas cidades, sem colapso — mesmo nível de evidência que o paper já
reportava para os outros sistemas (estabilidade dentro do dataset, não
generalização geográfica).

## 6. Generalização fora de domínio (pendência crítica: nunca rodada)

39 aproximações de placa anotadas na footage da Califórnia, mesma janela de 6 s.

**Primeiro, uma correção de método.** Os três sistemas da `tab:ood` são
pontuados por **evidência bruta** — ativação do GATE ou transcrição com
palavra-chave em qualquer ponto da janela. `eval_external_joint.py` pontuava o
estimador conjunto por **saída do estado OUTSIDE**, que é uma barra estritamente
mais alta; comparar os dois lado a lado teria sido a comparação pareada que a
legenda promete não ser. A corrida agora registra os dois critérios nas mesmas
amostras.

| sistema | todas (39) | dia (15) | ent+pôr (6) | noite (9) | chuva (4) | neblina (5) |
|---|---|---|---|---|---|---|
| 2B fine-tunado | 5% | 13% | 0% | 0% | 0% | 0% |
| detector+CLIP | 72% | 93% | 100% | 44% | 50% | 40% |
| C3E calibrado (evidência) | **95%** | 100% | 100% | 100% | 75% | 80% |
| joint 312 — **evidência recuperada** | **95%** | 100% | 100% | 100% | 75% | 80% |
| joint 312 — **estado age sobre ela** | 69% | 80% | 100% | 67% | 50% | 20% |
| joint 100 (mesmo critério de estado) | 74% | 87% | 100% | 56% | 50% | 60% |
| joint sem detector (refit, critério de estado) | 79% | 87% | 100% | 78% | 50% | 60% |

Três leituras, todas necessárias:

1. **A evidência bruta dentro da corrida conjunta é idêntica à do C3E sozinho**
   (37/39 nos dois). Não há perda de percepção: os mesmos prompts sobre as
   mesmas amostras recuperam as mesmas placas.
2. **O estimador descarta um quarto dessa evidência** (27/39). A máquina
   temporal contada in-domain é o filtro que a joga fora, não o detector: o
   ablado sem detector recupera só até 79%. Em ROADWork a combinação "detector
   cego + world model lendo placa" cobre 124 de 377.288 frames de calibração, e
   é exatamente o regime da noite/neblina da Califórnia.
3. **O mesmo botão controla nuisance e recall OOD**: o ajuste de 312 vídeos, que
   é o mais conservador in-domain (25,9 contra 33,2 FA/h), é também o que mais
   descarta fora de domínio (69% contra 74%).

Isto **não** contradiz o achado central do manuscrito — o modelo não
especializado continua sendo o que enxerga fora de domínio (95% contra 5% do
fine-tunado). O que se acrescenta é que perceber e agir são camadas separáveis,
e a segunda não transfere sozinha.

## 7. Segmentador temporal causal (pendência: treinado em 31/07, nunca avaliado)

`pipeline/tcn_segmenter.py`, MS-TCN causal de 100.780 parâmetros sobre os dois
fluxos, com *modality dropout*; treinado nos 312 vídeos de calibração (20 de
held-out, parada antecipada na época 55), avaliado nos 208 de validação.

**Correção necessária antes de reportar**: `evaluate()` reconstruía a verdade de
campo por forward-fill dos rótulos da grade de 10 Hz, o que quantiza as
fronteiras de intervalo na mesma linha temporal em que vivem as predições e
infla toda métrica de fronteira. Passou a pontuar contra a anotação por frame,
como todos os outros sistemas (`tcn_segmenter.py.bak` guarda a versão anterior).

| sistema | acc | macro-F1 | IoU appr | IoU in | IoU exit | prec. IN | FA/h |
|---|---|---|---|---|---|---|---|
| joint 312 | 66,1% | 0,546 | 0,246 | 0,578 | 0,160 | 66,8% | **25,9** |
| **TCN causal** | **67,8%** | **0,550** | 0,260 | 0,558 | **0,169** | 55,1% | 71,9 |

Empata em macro-F1 e leva o melhor `EXITING` de todos, mas a **2,8× a taxa de
alarmes falsos** e com 11,7 pontos a menos de precisão de evento — falha o
critério de seleção deste projeto (macro-F1 máximo sujeito a FA/h não pior que
os 30,9 do detector). Não é o sistema a recomendar; é o candidato que a
literatura de segmentação sugeria e que, medido, não se sustenta nessa métrica.

### 7.1 O TCN fora de domínio — o remédio proposto, medido

O *modality dropout* do segmentador existe precisamente para fabricar o regime
"detector cego, world model lendo placa" que ROADWork quase não contém. Medido
nas mesmas 39 aproximações, com o mesmo critério de estado do estimador
conjunto:

| sistema (critério de estado) | todas | dia | noite | chuva | neblina |
|---|---|---|---|---|---|
| joint sem detector | 79% | 87% | 78% | 50% | 60% |
| joint 100 | 74% | 87% | 56% | 50% | 60% |
| joint 312 | 69% | 80% | 67% | 50% | 20% |
| **TCN causal** | **44%** | 53% | 33% | 50% | 20% |

**O remédio não funciona.** O modelo temporal aprendido é o que menos transfere
de todos — abaixo até do detector (72%), que é o componente cuja fragilidade
fora de domínio motivou a fusão. Mascarar modalidade no treino não substitui ver
o regime: o que o TCN aprendeu em 377 mil frames de ROADWork foi a
co-ocorrência dos dois fluxos, e é isso que ele leva para a Califórnia.

## 8. O quadro que fecha (corrigido)

Uma versão anterior desta seção concluía que "quanto mais ajustada a camada
temporal, pior ela transfere", comparando a cascata do C3E por **evidência**
contra os estimadores por **ação**. Medimos a cascata por ação
(`pipeline/eval_external_cascade_action.py`, mesma janela, mesmas amostras,
a `CascadeStateMachine` real com os limiares do C3E):

| sistema | macro-F1 | FA/h | OOD evidência | OOD ação |
|---|---|---|---|---|
| 2B fine-tunado, rec. | 0,453 | 37,7 | 5% | — |
| detector+CLIP | 0,470 | 30,9 | 72% | — |
| C3E, cascata calibrada (4 inteiros) | 0,515 | 64,6 | 95% | **62%** |
| joint 312 (108 células contadas) | 0,546 | **25,9** | 95% | 69% |
| TCN causal (100 mil parâmetros) | **0,550** | 71,9 | 95% | 44% |

A cascata de quatro inteiros age sobre **menos** placas que o estimador contado.
A perda não acompanha o número de parâmetros ajustados; ela aparece em **todas**
as camadas temporais (62–69%, e 44% no TCN) sobre a mesma evidência de 95%.
Leitura (não demonstrada — não medimos a duração das rajadas de evidência):
cada camada foi selecionada in-domain para cortar alarmes falsos, e o jeito mais
barato de fazer isso em ROADWork é ignorar evidência que não se repete; à noite
na Califórnia, a evidência verdadeira chega assim.

O sistema a recomendar continua sendo o **joint 312**, com as ressalvas da §4
(o ganho é do filtro, não da fusão) e da §6 (age sobre 69% do que o world model
enxerga fora de domínio).

## Procedência

| resultado | artefato |
|---|---|
| replay e parâmetros | `pipeline/replay_joint.py`, `eval_cache/joint_full_val/` (novo) |
| IC pareados | `pipeline/paired_ci_fast.py` |
| perfil operacional | `pipeline/profile_joint.py` |
| ablação de canais | `pipeline/ablate_joint.py` |
| OOD do conjunto | `eval_cache/ood_california/eval_external_joint{,_312,_nodet}.log` |
| TCN in-domain | `eval_cache/tcn_journal.pt`, log em scratchpad |
| TCN OOD | `eval_cache/ood_california/eval_external_tcn.log` |
| cascata C3E por ação | `eval_cache/ood_california/eval_external_cascade_action.log` |
| TCN com métricas padrão | `eval_cache/tcn_val/`, `pipeline/dump_tcn_predictions.py` |
| correções | `pipeline/tcn_segmenter.py.bak`, `pipeline/eval_external_joint.py.bak`, `pipeline/paired_ci.py.bak` |

## 9. A métrica de "evidência" fora do domínio não controla a taxa de base (2026-09-28)

Com o dump contínuo da Califórnia (`eval_cache/ood_stream/`, 51 min a 1 Hz,
`pipeline/dump_ood_stream.py`; o replay reproduz as cinco medições de GPU:
37, 24, 27, 29 e 31 de 39) foi possível olhar **fora** das janelas de placa.

| | C3E | detector |
|---|---|---|
| "sim" por amostra, a >20 s de qualquer placa | 56,8% (GATE) | 19,9% (algum objeto) |
| janela aleatória de 7 s longe das placas com "evidência" | **84%** | 43% |
| janelas nas 39 placas com "evidência" | 95% | 72% |
| AUC placa vs. longe (contagem de evidência na janela) | **0,55** | 0,59 |

Checagem visual de 16 frames sorteados em que o C3E diz "sim" longe das placas
(`eval_cache/ood_california/far_yes_spotcheck.jpg`): obra clara em 2 (uma é placa
"WORK ZONE 25 MPH" não anotada), ambíguos em 4 (barreiras de concreto), nenhum
sinal de obra em 10.

**Consequência.** Os 95% contra 72% contra 5% medem com que frequência cada
modelo diz "sim" perto de uma placa, sem descontar com que frequência diz "sim"
em qualquer lugar. Descontada a taxa de base, o C3E não separa placa de
não-placa melhor que o detector nesses dados. Fora do domínio, cada modelo se
desloca numa direção de viés diferente: o 2B fine-tunado fica calado, o C3E fica
afirmativo, o detector fica no meio.

Isso invalida, como estão escritas:
- "in-domain results did not predict out-of-domain evidence recovery" e a
  figura 1 (C3E 95% vs. 2B 5%);
- a seção "Perceiving is not acting" e a leitura de que as camadas temporais
  "descartam" evidência: parte do que elas descartam são "sim" espúrios;
- a tese "perception transfers, temporal models do not".

Continuam válidos (medidos com rótulo): tudo in-domain, incluindo a
não-portabilidade dos limiares, o estimador conjunto e sua ablação.

Tentativa de corrigir o parser (§ anterior da conversa): palavras-chave no DESC
inteiro fazem a cascata ficar ativa 100% do tempo (a resposta repete "work zone"
do prompt); só em texto citado, recall 67%→74% com teto de disparos 117→172/h.
Não é um conserto.

**Sem negativos rotulados fora do domínio não dá para afirmar nada comparativo
sobre transferência**, em nenhuma direção.
