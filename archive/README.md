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

A linhagem ativa está documentada no `README.md`/`PIPELINE.md` da raiz.
