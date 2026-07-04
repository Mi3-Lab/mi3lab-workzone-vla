# Patches para os repositórios vendored

Este projeto depende de dois clones da NVIDIA, tratados como dependências
externas — **não fazem parte deste repositório**:

- `alpamayo-recipes/` — clone de [`NVlabs/alpamayo-recipes`](https://github.com/NVlabs/alpamayo-recipes)
- `alpamayo1.5/` — clone de [`NVlabs/alpamayo1.5`](https://github.com/NVlabs/alpamayo1.5.git) (não modificado, sem patches)

Ambos devem ficar clonados ao lado deste repositório (`../alpamayo-recipes`,
`../alpamayo1.5` relativo à raiz do projeto), com o `.venv` compartilhado
instalado a partir deles.

## O que são estes patches

`alpamayo-recipes` precisou de pequenas correções para o nosso pipeline
funcionar. São modificações genuínas em arquivos **pré-existentes** do
vendor (não confundir com os arquivos 100% novos deste repositório, que
vivem em `training/`, `inference/`, `data_prep/`, `diagnostics/`).

| Patch | Arquivo alvo | O que faz |
|---|---|---|
| `qwen_processor.py.patch` | `src/alpamayo/processor/qwen_processor.py` | Fix de masking de labels para backbones Qwen3 sem tokens `<\|answer_start\|>`/`<\|answer_end\|>` (necessário para o Cosmos-Reason2) |
| `trainer.py.patch` | `recipes/alpamayo1_5_sft/trainer.py` | Suporte a `lr_multiplier` por prefixo de parâmetro e `eval_strategy` |
| `train_hf.py.patch` | `recipes/alpamayo1_5_sft/train_hf.py` | Ajustes de entrypoint |
| `sft_base_model.py.patch` | `recipes/alpamayo1_5_sft/models/sft_base_model.py` | Ajustes no `TrainableReasoningVLA` |

## Symlinks — módulos que precisam viver dentro do pacote vendor

Além dos configs (abaixo), 3 arquivos Python nossos precisam estar
FISICAMENTE dentro da árvore `alpamayo-recipes/recipes/alpamayo1_5_sft/`
porque são importados via caminho de pacote absoluto
(`alpamayo1_5_sft.models.kd_model`, não um script solto) — diferente de
`workzone_state.py`, que só precisa estar em algum diretório do `sys.path`
(resolvido inserindo `mi3lab-workzone-vla/` no `sys.path` dos scripts que o
usam). São symlinks apontando pra cá, mesma lógica dos configs:

| Arquivo neste repo | Symlink em alpamayo-recipes |
|---|---|
| `training/models/kd_model.py` | `recipes/alpamayo1_5_sft/models/kd_model.py` |
| `training/trainer_kd.py` | `recipes/alpamayo1_5_sft/trainer_kd.py` |
| `training/train_kd.py` | `recipes/alpamayo1_5_sft/train_kd.py` |

Se reconstruir o ambiente do zero (clone novo de `alpamayo-recipes`), refaça
esses 3 symlinks antes de rodar `train_kd.py` ou qualquer script de
`inference/`/`diagnostics/` que importe `alpamayo1_5_sft.models.kd_model`.

Os configs (`sft_base.yaml`, `wandb/default.yaml`, e todos os nossos
`sft_stage*.yaml`/`configs/models/*.yaml`/`configs/deepspeed/zero2_fast.json`)
**não** são patches — vivem como arquivos reais em `../configs/` deste
repositório, e o ambiente de trabalho atual os referencia via **symlink**
de dentro de `alpamayo-recipes/recipes/alpamayo1_5_sft/configs/`. Isso
evita duplicar conteúdo e mantém este repositório como única fonte de
verdade. Detalhe importante: `configs/sft_base.yaml` **modifica** um
arquivo que existe no vendor original (adiciona `save_only_model: true` —
ver contexto abaixo) — por isso é tratado como config "nossa" mesmo
tendo equivalente upstream.

## Como reconstruir o ambiente do zero

```bash
# 1. clonar os vendors ao lado deste repo
git clone https://github.com/NVlabs/alpamayo-recipes ../alpamayo-recipes
git clone https://github.com/NVlabs/alpamayo1.5.git ../alpamayo1.5

# 2. aplicar os patches de código
cd ../alpamayo-recipes
git apply ../mi3lab-workzone-vla/patches/qwen_processor.py.patch
git apply ../mi3lab-workzone-vla/patches/trainer.py.patch
git apply ../mi3lab-workzone-vla/patches/train_hf.py.patch
git apply ../mi3lab-workzone-vla/patches/sft_base_model.py.patch

# 3. symlinkar os configs (repete o setup atual)
cd recipes/alpamayo1_5_sft/configs
ln -s ../../../../mi3lab-workzone-vla/configs/sft_base.yaml .
ln -s ../../../../mi3lab-workzone-vla/configs/*.yaml .        # todos os sft_stage*, sft_fasea_trajectory
ln -s ../../../../mi3lab-workzone-vla/configs/models/*.yaml models/
ln -s ../../../../mi3lab-workzone-vla/configs/wandb/default.yaml wandb/
ln -s ../../../../mi3lab-workzone-vla/configs/deepspeed/zero2_fast.json deepspeed/

# 4. instalar dependências (venv compartilhado)
cd ../../../../alpamayo-recipes/recipes/alpamayo1_5_sft
python -m venv .venv && source .venv/bin/activate
pip install -e ../../../alpamayo-recipes -e ../../../alpamayo1.5
```

## Por que essa separação

`alpamayo-recipes/` e `alpamayo1.5/` são clones de terceiros (NVIDIA) —
não são nosso código. Em algum momento anterior um commit local acabou
poluindo o histórico git do clone vendored com trabalho nosso, divergindo
do `origin/main`. Este repositório existe para separar limpamente o que é
nosso do que é vendored: `alpamayo-recipes` foi resetado para
`origin/main` limpo (`git reset --soft` + descarte seletivo), mantendo
só os patches genuínos documentados acima.
