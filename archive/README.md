# Archive

Configs, scripts e jobs superados — já cumpriram seu papel (responderam a
pergunta que motivou sua criação) e não fazem parte da linhagem ativa do
pipeline. Preservados aqui como referência da jornada de debug, não como
algo para rodar de novo.

- `configs/` — estágios de treino superados (Stage 1 v1-v3, Stage 3 KD
  puro, Stage 4 v2-v3) e configs de modelo correspondentes
- `diagnostics/` — testes exploratórios pontuais já concluídos (trajetória,
  ego-state, leitura de placa, avaliação de checkpoint)
- `inference/` — versões antigas do pipeline de inferência (v1, v4-v6, era
  pré-filtro-Bayesiano). Versões mais recentes (v7-v14) que documentam
  correções específicas ficam em `../inference/archive/`; a atual é
  `../inference/video_inference_v15.py`
- `jobs/` — os `.sbatch` correspondentes, com os caminhos já atualizados
  caso alguém queira reexecutar algo daqui

## Fase Alpamayo 2B (cabeça de ação)

- `diagnostics/eval_action_2b_ade.py` + `jobs/eval_action_2b_ade.sbatch` —
  loop de avaliação escrito à mão, abandonado: montava o batch errado
  (KeyError `input_ids`, o `tokenized_data` é um dict aninhado). Substituído
  pelo `evaluate_hf.py` oficial via `training/eval_action_2b.py`
- `diagnostics/debug_addtok.py` + `jobs/debug_addtok.sbatch` — debug pontual
  do patch de vocabulário; o patch em si é `training/alpamayo_r1_vocab_patch.py`
- `jobs/gen_teacher_labels{,_test}.sbatch` — a versão original (superada pelas
  variantes `_a100`/`_l40s`, que rodam 8 shards em paralelo) e a tentativa na
  partição `test`, barrada pelo limite de QOS `MaxSubmit=1`
- `jobs/kdvel_1gpu.sbatch`, `jobs/cotrain_a100.sbatch` — variantes de GPU que
  não chegaram a rodar (fila cheia / nós A100 em `DRAIN`)
- `jobs/{kdvel,cotrain}_smoke.sbatch` — smoke tests gerados por `sed` a partir
  do job principal; regeneráveis

A linhagem ativa está documentada no `README.md`/`PIPELINE.md` da raiz.
