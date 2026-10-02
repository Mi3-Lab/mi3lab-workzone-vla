Resumo do trabalho

O artigo apresenta um sistema embarcado para estimar quatro estados de travessia de zonas de obras — OUTSIDE, APPROACHING, INSIDE e EXITING — utilizando um único VLM de 2 bilhões de parâmetros, quantizado em INT4 e executado em uma Jetson AGX Orin. Em vez de uma única consulta, o modelo é operado por quatro canais de prompts com custos diferentes: uma porta binária, transcrição de placas, estimativa do estado egocêntrico e descrição estruturada. Esses sinais são combinados por uma cascata baseada na qualidade da evidência, votação temporal, filtro Bayesiano e histerese.

A avaliação compara o sistema com uma implementação de um pipeline detector+CLIP, utilizando os mesmos 208 vídeos de validação, o mesmo dispositivo e um protocolo que avança o vídeo de acordo com a latência real de cada ciclo. O sistema VLM alcança desempenho global próximo ao baseline e melhor IoU para APPROACHING, mas apresenta precisão de eventos substancialmente inferior sem pós-processamento.

Avaliação geral

Recomendação preliminar: Weak Reject / Revise and Resubmit

O trabalho é interessante, relevante para o Applications Track e, em vários aspectos, mais cuidadoso do que a média dos estudos de implantação embarcada. A principal contribuição não é um novo algoritmo fundamental, mas uma combinação coerente de engenharia de prompts, orçamento de tokens, lógica temporal e avaliação sensível à latência. Isso está alinhado aos critérios do WACV para sistemas e aplicações.

Entretanto, na forma atual, algumas conclusões centrais são mais fortes do que os experimentos sustentam. As maiores preocupações são: possível dependência entre dados de treinamento e avaliação, baixa cobertura de negativos realmente difíceis, taxa de alarmes falsos incompatível com a aplicação declarada e evidência insuficiente de que a melhoria atribuída à transcrição é estatisticamente robusta. A maior parte dessas questões parece corrigível em uma revisão substancial.

Pontos fortes
1. Boa contribuição de sistema para o Applications Track

A formulação de um único VLM como vários “canais” de percepção, diferenciados por prompt e orçamento de saída, é simples, mas conceitualmente útil. O artigo mostra que o custo não depende apenas do tamanho do modelo: o número de tokens de saída altera fortemente a latência, motivando consultas assimétricas e escalonamento condicional. A Tabela 1 torna essa relação clara, com latências que variam aproximadamente de 90 ms a 540 ms por canal.

A Figura 1, na página 4, também comunica bem a arquitetura. A distinção entre evidência fraca, evidência textual verificável, entrada rápida e regras de saída por histerese é clara e justificável.

2. Protocolo de avaliação sensível à latência

O protocolo de “câmera em tempo de parede” é uma contribuição valiosa. Em vez de processar todos os frames e reportar a latência separadamente, cada sistema perde frames proporcionalmente ao seu tempo real de inferência. Essa escolha aproxima a avaliação de uma execução embarcada e impede que um método lento receba observações que não teria em operação.

Esse aspecto tem potencial de impacto além da aplicação específica e deveria ser ainda mais destacado e formalizado.

3. Comparação pareada razoavelmente controlada

A execução do pipeline detector+CLIP nos mesmos vídeos, no mesmo hardware e sob o mesmo simulador é muito mais informativa do que comparar somente com números publicados. A apresentação separa adequadamente os resultados reproduzidos dos resultados do artigo de referência.

4. Análise honesta das limitações

O manuscrito reconhece explicitamente que a precisão bruta de eventos, 54,1%, é inadequada para implantação e que o VLM continua pior em custo computacional, precisão agregada e EXITING. Essa transparência aumenta a credibilidade do trabalho.

Também é positivo que o artigo quantifique a taxa operacional de falsos alarmes, em vez de esconder o problema atrás de uma única métrica agregada.

5. Ablations informativas

As ablações estão bem conectadas às decisões do sistema. Particularmente útil é a comparação entre transcrição e pergunta binária sobre placas: a variante binária reduz a acurácia de 62,5% para 52,5%, apoiando a alegação de que o formato da consulta importa.

6. Escrita e organização

O manuscrito é, em geral, claro, bem estruturado e autoconsciente. A motivação aparece cedo, os componentes são nomeados de forma consistente e a narrativa conecta falhas observadas a decisões de projeto. A Figura 2, na página 5, é especialmente útil por mostrar saídas reais dos quatro canais durante uma sequência completa.

Principais fragilidades
1. Possível dependência entre treinamento e avaliação

Esta é a preocupação mais importante.

O artigo informa que o fine-tuning utiliza imagens derivadas do ROADWork e que o benchmark de vídeo também deriva do ROADWork. Admite-se que algumas imagens de treinamento podem representar as mesmas zonas de obras presentes nos vídeos de avaliação. O argumento apresentado é que a comparação continua “balanceada” porque o detector baseline também foi treinado no ROADWork.

Isso pode preservar parcialmente a equidade relativa entre os dois sistemas, mas não resolve a validade da alegação de generalização. Há pelo menos três formas de dependência possíveis:

frames próximos ou quase duplicados;
a mesma zona, local e configuração física em momentos diferentes;
placas, veículos, geometrias ou fundos memorizáveis.

Um VLM com capacidade de memorizar contexto visual pode se beneficiar de sobreposição de maneira diferente de um detector. Portanto, “ambos foram treinados no ROADWork” não é uma justificativa suficiente.

Revisão necessária: executar uma auditoria explícita de sobreposição e, idealmente, uma divisão por vídeo, local físico ou sequência de captura. Uma avaliação leave-one-city-out, leave-one-location-out ou em um pequeno conjunto externo seria muito valiosa. Caso isso não seja possível, o texto precisa reduzir as alegações de generalização e apresentar uma análise de similaridade entre dados de treinamento e validação.

2. A aplicação de segurança ainda apresenta taxa de alarmes falsos muito alta

O sistema bruto produz 60,1 falsas ativações por hora e permanece incorretamente ativo durante 32,6% dos frames nos 13 vídeos totalmente negativos.

Mesmo após o debounce recomendado, permanecem falsos positivos sustentados associados a objetos de obra irrelevantes para a trajetória. O próprio artigo reconhece que regras temporais não corrigem esse problema.

Isso cria uma tensão com a caracterização do sistema como solução de segurança em tempo real. O resultado é cientificamente interessante como protótipo de pesquisa, mas ainda não constitui evidência de prontidão para implantação. O título, abstract e conclusão deveriam distinguir melhor:

“execução integralmente embarcada”;
“tempo real em sentido computacional”;
“adequação operacional para alerta ao motorista”.

Atualmente, essas ideias ficam próximas demais.

Revisão necessária: reportar a taxa de falsos alarmes também para a configuração recomendada de 1 segundo, não apenas a precisão de eventos. Seria útil apresentar falsos alarmes por hora, duração média, tempo total ativo incorretamente e curvas precision–recall–delay para vários valores de debounce.

3. O conjunto negativo é pequeno e aparentemente pouco representativo

Existem apenas 13 vídeos totalmente negativos na validação. Isso é insuficiente para caracterizar de forma confiável uma aplicação em que falsos positivos são um dos riscos centrais. Além disso, os negativos difíceis não são cenas “limpas”, mas cenas contendo obras em calçadas, veículos estacionados, sinalização de outra via ou estruturas irrelevantes à trajetória. O artigo identifica precisamente essa família de falhas, mas não apresenta um benchmark suficientemente amplo dessa categoria.

Revisão necessária: criar uma avaliação específica de “hard ego-negative examples”, com categorias e contagens. Por exemplo:

construção apenas na calçada;
obra em faixa oposta;
veículo de manutenção estacionado;
cones de estacionamento;
sinalização residual;
obra em rua transversal;
objetos distantes sem impacto na rota.

Uma matriz de erro por categoria fortaleceria bastante a principal conclusão do artigo.

4. A atribuição causal à transcrição não está totalmente demonstrada

O artigo atribui a melhor localização de APPROACHING ao canal SIGN e ao fast entry. Essa hipótese é plausível, mas as ablações apresentadas oferecem suporte limitado:

retirar SIGN muda pouco as métricas agregadas;
retirar fast entry altera o MAE de entrada de 2,53 s para 2,59 s;
não são fornecidos intervalos de confiança para essas diferenças;
não há uma métrica isolando apenas eventos iniciados por placas.

A diferença de 0,06 s em MAE é pequena diante das latências por ciclo e da variabilidade dos eventos. O argumento qualitativo pode ser verdadeiro, mas a força da redação excede a evidência quantitativa.

Revisão necessária: reportar:

quantidade e porcentagem de entradas disparadas por SIGN;
tempo ganho por esses eventos em relação à porta de votação;
distribuição pareada por evento;
intervalo de confiança e teste estatístico;
resultados especificamente em vídeos com sinalização antecipada.
5. O uso de “real-time” precisa ser melhor qualificado

O sistema opera entre 3,2 Hz e 10,6 Hz, dependendo dos canais acionados, com componentes individuais chegando a aproximadamente 540 ms.

Isso pode ser suficiente para uma estimativa semântica de estado, mas “real-time” depende dos requisitos temporais da aplicação. A 100 km/h, um ciclo de 540 ms corresponde a cerca de 15 metros de deslocamento. O texto discute parcialmente distâncias durante a janela de votação, mas não define um requisito de deadline ou frequência mínima.

Revisão necessária: definir o significado operacional de tempo real. Por exemplo, qual é a latência máxima aceitável para o alerta? Qual taxa mínima é necessária? Há deadline misses? Qual é a distribuição p50/p95/p99 da latência, em vez de apenas valores médios ou aproximados?

6. Comparação insuficiente com alternativas mais simples

A comparação principal é com detector+CLIP, o que é adequado, mas falta entender se o ganho vem realmente do uso de um VLM ou de uma forma mais simples de OCR e lógica temporal.

O canal SIGN parece decisivo na narrativa. Um baseline importante seria:

OCR especializado + léxico;
detector de placas temporárias + OCR;
detector puro com regras de antecipação;
detector+CLIP acrescido da mesma fast-entry baseada em texto;
VLM apenas para placas, mantendo o detector para o restante.

Sem esses controles, não está claro se a melhor conclusão é “um único VLM pode substituir o pipeline” ou “OCR de placas é um sinal complementar valioso”.

7. A configuração denominada “single VLM” ainda executa múltiplas inferências sequenciais

Todos os canais compartilham os mesmos pesos, mas cada prompt é uma requisição separada, com nova etapa de prefill/decoding e aparentemente reutilização limitada do embedding visual. Em ciclos em que GATE, SIGN e DESC são executados, o custo total pode ser alto. O artigo descreve um único engine, mas isso não é necessariamente equivalente a uma única passagem de percepção.

Seria importante esclarecer:

o embedding visual é reutilizado entre canais no mesmo frame?
as consultas são sequenciais ou paralelas?
qual é a latência end-to-end real de cada caminho da cascata?
qual é a frequência com que DESC é acionado?
qual é a utilização média de GPU, potência e energia por segundo?

Para um artigo de implantação em edge, energia e consumo médio são métricas importantes.

8. Detalhes estatísticos insuficientes

O artigo apresenta bootstrap CI e um valor de p para acurácia por vídeo, mas não explica:

unidade de reamostragem;
número de replicações;
teste utilizado;
tratamento de vídeos com durações diferentes;
correção para múltiplas comparações;
intervalos de confiança das diferenças em IoU e timing.

Como os frames dentro de um vídeo são fortemente correlacionados, testes em nível de frame seriam inválidos. O manuscrito parece usar vídeo como unidade, o que é apropriado, mas deve especificá-lo claramente.

9. Macro-F1 e acurácia não contam toda a história

O baseline publicado tem macro-F1 0,60, enquanto a reprodução pareada apresenta 0,470. O artigo atribui a diferença ao protocolo sensível à latência, mas também há diferenças de divisão de dados, detalhes da implementação ou amostragem que podem contribuir.

É necessária uma decomposição mais cuidadosa:

baseline original no protocolo original;
port no protocolo original;
port no protocolo latency-honest;
efeito isolado do novo split;
efeito isolado de cada componente da reprodução.

Caso contrário, a conclusão de que toda a diferença decorre do protocolo é prematura.

10. Reprodutibilidade depende de artefatos ainda não disponibilizados

O abstract informa que código, prompts e harness “serão liberados”. O manuscrito inclui vários detalhes úteis, mas uma reprodução completa exige:

checkpoint ou procedimento exato de fine-tuning;
prompts completos;
léxicos e qualificadores;
modelo de emissão do filtro Bayesiano;
thresholds;
implementação do simulador;
engine e comandos de quantização;
split exato por vídeo;
baseline portado.

As diretrizes do WACV incentivam a avaliação de reprodutibilidade e o envio voluntário de código suplementar.

A promessa de liberação futura é positiva, mas uma submissão mais forte incluiria pelo menos os artefatos anonimizados essenciais no suplemento.

Questões para resposta/rebuttal
Há frames, sequências ou locais físicos compartilhados entre o conjunto de fine-tuning e os 208 vídeos de validação? Como essa sobreposição foi medida?
Qual é a taxa de falsos alarmes por hora da configuração recomendada greedy + 1 s debounce?
Quantos eventos de entrada foram disparados diretamente pelo canal SIGN? Qual o ganho mediano de tempo nesses eventos?
Os intervalos de confiança das diferenças de APPROACHING IoU e de entry timing excluem zero?
O baseline detector+CLIP recebeu a mesma possibilidade de fast entry baseada em placas ou um módulo OCR? Caso contrário, a comparação separa arquitetura de política de decisão?
Qual é a latência end-to-end p50, p95 e p99 dos principais caminhos da cascata?
O encoder visual é executado novamente para cada canal ou o embedding é armazenado e reutilizado?
Qual é o consumo de energia médio e de pico no AGX Orin? Em qual modo de potência e frequência o dispositivo foi executado?
Como foram definidos os parâmetros do filtro Bayesiano e o modelo de emissão? Esses valores foram aprendidos ou escolhidos manualmente?
A afirmação de 0% de falso “yes” após hard-negative calibration refere-se a quantas imagens e a qual intervalo de confiança?
O baseline foi validado inicialmente em um pequeno conjunto para confirmar equivalência funcional à implementação original antes da execução latency-honest?
Existem resultados por cidade, velocidade do veículo, distância da obra e presença/ausência de placas antecipadas?
Sugestões concretas de revisão
Experimentos prioritários

A versão revisada deveria, no mínimo:

acrescentar uma auditoria de sobreposição entre treinamento e validação;
reportar falsos alarmes por hora para todas as configurações operacionais;
apresentar uma avaliação específica em negativos egocêntricos difíceis;
comparar com OCR + léxico e com detector+OCR;
fornecer intervalos de confiança pareados para as principais alegações;
reportar latência em distribuição, consumo de energia e frequência de acionamento de cada canal.
Melhorias de apresentação

A Tabela 2 mistura configurações brutas, recomendadas, baseline pareado e números publicados. Uma tabela complementar poderia separar:

comparação científica sem pós-processamento;
comparação das melhores configurações de implantação;
custo computacional;
alarmes falsos por hora.

Também seria útil evitar marcar em negrito apenas o melhor número entre rec. e paired quando algumas diferenças são muito pequenas ou sem significância estatística.

Ajustes nas alegações

Sugiro substituir formulações como:

“the pure-VLM system matches overall accuracy and event recall, localizes the APPROACHING boundary state better, and alerts earlier”

por algo mais cauteloso:

“under the evaluated split and paired latency-aware protocol, the system achieves comparable aggregate accuracy and recall, with higher observed APPROACHING IoU and earlier median entry estimates; the statistical robustness of the latter differences requires further evaluation.”

A conclusão também deveria enfatizar que o sistema demonstra viabilidade computacional e competitividade experimental, não prontidão para implantação de segurança.

Avaliação por critérios
Critério	Avaliação
Relevância para WACV	Alta
Novidade algorítmica	Moderada-baixa
Inovação em nível de sistema	Alta
Qualidade técnica	Moderada
Qualidade experimental	Moderada
Clareza	Alta
Reprodutibilidade	Moderada, condicionada à liberação dos artefatos
Impacto potencial	Alto
Prontidão para aplicação	Baixa na configuração atual
Pontuação simulada

Nota global: 4/10 — Weak Reject / major revision required

Confiança: 4/5

Justificativa da decisão

O trabalho tem uma contribuição de sistema legítima e adequada ao Applications Track: a cascata é bem motivada, o protocolo de latência é valioso e a comparação pareada é promissora. Entretanto, a dependência potencial entre treinamento e avaliação, a pequena cobertura de negativos difíceis e a taxa operacional de falsos alarmes impedem aceitar as conclusões mais fortes na forma atual. Uma revisão que resolva principalmente a separação dos dados e a avaliação de falsos positivos poderia alterar substancialmente a recomendação.