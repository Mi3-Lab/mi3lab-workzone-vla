# Rodando o projeto no NVIDIA Thor

Este guia leva o projeto do Jetson AGX Orin (onde tudo foi desenvolvido) para um Thor, roda um teste de fumaça que compara o Thor com o Orin, e repete os experimentos do paper.

Resumo do caminho:

```
Orin (origem)                         Thor (destino)
--------------                        -------------------------------------------
código   --- git push/clone ------->  ~/jetson-deploy
modelos  --- 01_sync_from_orin ---->  ONNX + .pt (os engines NÃO são copiados)
dados    --- 01_sync_from_orin ---->  ~/workzone/data
                                      02 compila o Edge-LLM com o nosso patch
                                      03 cria o ambiente Python
                                      04 constrói os engines NESTA placa
                                      05 teste de fumaça + regressão vs Orin
                                      06 experimentos do paper
```

Cada script é idempotente: se parar no meio, rode de novo e ele continua de onde estava.

> **A placa deste projeto é a DRIVE AGX Thor (DriveOS).** Siga a seção 4 normalmente, mas compile o Edge-LLM como na **seção 5** (Docker do DriveOS SDK num PC x86) e leia lá o que muda sem PyTorch.

---

## 1. Qual Thor você tem?

São duas placas diferentes, com sistemas diferentes. Confira antes de começar:

```bash
tr -d '\0' < /proc/device-tree/model; echo
ls /etc/nv_tegra_release 2>/dev/null && echo "JetPack -> Jetson AGX Thor" || echo "sem JetPack -> provavelmente DRIVE AGX Thor (DriveOS)"
```

| | **Jetson AGX Thor** | **DRIVE AGX Thor** |
|---|---|---|
| Sistema | JetPack 7.x (Ubuntu 24.04) | DriveOS 7.2 |
| CUDA | 13.0 (JP 7.0/7.1) ou 13.2 (JP 7.2) | 13.x, do DriveOS SDK |
| Onde compila o Edge-LLM | na própria placa | **no Docker do DriveOS SDK, num PC x86**, e copia o `build/` para a placa |
| Alvo do Edge-LLM | `jetson-thor` | `auto-thor` |
| PyTorch (detector YOLO) | wheels CUDA 13 aarch64 | **pode não existir**; sem ele só a parte de linguagem roda |

Os scripts detectam a placa sozinhos. Se a detecção errar, force em `thor/config.sh`: `PLATFORM=jetson-thor` ou `PLATFORM=drive-thor`.

---

## 2. Pré-requisitos

- **Jetson AGX Thor:** JetPack 7 instalado (com CUDA e TensorRT). `nvcc --version` e `dpkg -l | grep tensorrt` devem responder.
- **DRIVE AGX Thor:** DriveOS 7.2 na placa, e um PC x86 com o Docker do DriveOS SDK 7.2.
- **Rede até o Orin** (para copiar modelos e dados por `rsync`/SSH). Teste: `ssh mi3-jetson@IP_DO_ORIN hostname`.
- **Disco:** ~6 GB de modelos, ~5 GB de engines, e de 0,3 a 22 GB de dados conforme o teste (tabela na seção 4).
- `git`, `cmake`, `build-essential`, `rsync`, `python3-venv`:
  ```bash
  sudo apt update && sudo apt install -y git cmake build-essential rsync python3-venv
  ```

---

## 3. Levar o código

**No Orin** (uma vez), publique o repositório num remoto seu:

```bash
cd ~/jetson-deploy
git remote add origin <URL-DO-SEU-REPOSITORIO>   # ex.: git@github.com:Mi3-Lab/jetson-deploy.git
git push -u origin main
```

**No Thor:**

```bash
git clone -b jetson-deploy git@github.com:Mi3-Lab/mi3lab-workzone-vla.git ~/jetson-deploy
cd ~/jetson-deploy
```

Clone exatamente em `~/jetson-deploy`: os scripts do pipeline usam esse caminho. Se clonar em outro lugar, o passo 4 cria um link `~/jetson-deploy` apontando para ele.

O git leva só o código, os papers, a anotação e os resultados em cache (`eval_cache/`, ~40 MB). Modelos, engines e vídeos ficam de fora e vêm pelo passo seguinte.

---

## 4. Passo a passo (Jetson AGX Thor)

Todos os comandos rodam de `~/jetson-deploy`.

### Passo 0 — configurar

Edite as três linhas do topo de `thor/config.sh`:

```bash
PLATFORM="${PLATFORM:-auto}"                        # deixe auto
ORIN_HOST="${ORIN_HOST:-mi3-jetson@192.168.X.Y}"    # usuário@IP do Orin
STORAGE="${STORAGE:-$HOME}"                         # ou um SSD, ex.: /mnt/nvme
```

Veja o estado da máquina (não muda nada):

```bash
bash thor/00_preflight.sh
```

### Passo 1 — copiar modelos e dados do Orin

```bash
bash thor/01_sync_from_orin.sh models smoke
```

| pacote | conteúdo | tamanho | para quê |
|---|---|---|---|
| `models` | ONNX do 2B e do C3E, YOLO `.pt`, CLIP | ~6 GB | sempre |
| `smoke` | 1 vídeo da Califórnia + 1 de Boston + anotações | ~0,3 GB | teste de fumaça |
| `validation` | 208 vídeos de validação | ~4,2 GB | tabela in-domain |
| `california` | 15 vídeos anotados da Califórnia | ~5,5 GB | fora do domínio |
| `sf` | 8 clipes de São Francisco | ~0,4 GB | controle SF |
| `calibration` | 312 vídeos de calibração | ~6,3 GB | só para recalibrar no Thor |

Pode copiar o resto depois, quando for rodar os experimentos.

### Passo 2 — compilar o TensorRT Edge-LLM

```bash
bash thor/02_build_edgellm.sh
```

Clona o Edge-LLM **v0.9.0** (commit `1ac0f2b`), aplica `thor/edgellm_patch/` e compila. O patch tem três partes, e o projeto não roda sem elas:

1. suporte ao Cosmos3-Edge (6 arquivos em `cpp/`: registro do modelo, runners multimodal e ViT, builder visual, cache de RoPE);
2. o binário `llm_stream_video`, um processo persistente que todos os scripts do pipeline usam para conversar com o modelo;
3. uma correção de link do CuTe DSL em `cmake/CuteDsl.cmake`.

Leva de 20 a 40 minutos. O patch foi validado contra uma cópia limpa do v0.9.0 (`git apply --check`).

### Passo 3 — ambiente Python

```bash
bash thor/03_python_env.sh
```

Cria `~/workzone/venv` e instala `torch` com CUDA 13 (`download.pytorch.org/whl/cu130`), depois `ultralytics==8.3.250` (fixado porque o engine do detector guarda essa versão), `open_clip`, `opencv`. Se o índice do torch não servir para a sua imagem do JetPack, informe outro:

```bash
TORCH_INDEX=<url-do-indice> bash thor/03_python_env.sh
```

Ao final o script diz se o torch enxerga a GPU. Sem torch com CUDA, os modelos de linguagem continuam funcionando; o detector não.

### Passo 4 — construir os engines nesta placa

```bash
bash thor/04_build_engines.sh
```

Engines TensorRT ficam presos à GPU onde foram construídos: **os do Orin não rodam no Thor**. O script constrói três, com os mesmos parâmetros usados no Orin:

| engine | origem | parâmetros |
|---|---|---|
| 2B fine-tunado | `onnx/llm`, `onnx/visual` | INT4 AWQ + visual FP16, 128–512 tokens de imagem |
| Cosmos3-Edge | `cosmos3edge-orin-deploy/onnx/{reasoner-int4,visual-fp16}` | INT4 AWQ + visual FP16, **299 tokens fixos (736×416)** |
| detector YOLO12s | `yolo12s_hardneg_1280.pt` | FP16, 960×960, batch 1 |

Para refazer só um: `bash thor/04_build_engines.sh c3e`. Para forçar: `FORCE=1 bash thor/04_build_engines.sh`.

### Passo 5 — teste de fumaça

```bash
bash thor/00_preflight.sh      # deve fechar sem "faltando"
bash thor/05_smoke_test.sh     # ~5 min; --quick para ~2 min
```

O teste faz quatro coisas e imprime PASS/WARN/FAIL em cada:

1. **2B responde direito** a um frame com obra. Se a resposta começar com `.DATA` ou lixo parecido, o ONNX perdeu a correção de `tie_word_embeddings` (ver o README principal).
2. **C3E responde, e não vazio.** O C3E aceita só 736×416; qualquer outro tamanho falha **em silêncio**, com respostas vazias. Isso já aconteceu neste projeto, e o teste existe para pegar.
3. **Regressão contra o Orin:** roda o C3E e o detector em cada segundo de `CA99_Day_01` e compara com o que o Orin gravou. No próprio Orin dá 100%. No Thor, algumas trocas são esperadas (kernels INT4 diferentes); abaixo de 90% vira WARN, abaixo de 75% FAIL.
4. **Latência** por canal, p50/p95, para os dois modelos e o detector.

Resultado em `eval_cache/thor/smoke_<placa>_<data>.json`. A referência do Orin, medida com este mesmo script, é:

| canal | 2B (480 px) p50 / p95 | C3E (736×416) p50 / p95 |
|---|---|---|
| GATE | 140 / 152 ms | 147 / 152 ms |
| SIGN | 158 / 166 ms | 280 / 291 ms |
| EGO | 143 / 157 ms | 172 / 183 ms |
| DESC | 480 / 511 ms | 538 / 579 ms |
| detector YOLO | 29 ms p50 | |

### Passo 6 — experimentos do paper

São longos. Rode em segundo plano e acompanhe pelo log:

```bash
bash thor/01_sync_from_orin.sh validation california sf     # se ainda não copiou
nohup bash thor/06_run_experiments.sh sf california indomain > ~/thor_experiments.log 2>&1 &
tail -f ~/thor_experiments.log
```

| conjunto | o que roda | duração no Orin |
|---|---|---|
| `sf` | evidência do C3E nos 8 clipes de São Francisco | ~10 min |
| `california` | evidência do C3E e do detector nos 14 vídeos da Califórnia | ~1 h |
| `indomain` | 5 sistemas × 208 vídeos (2B amostrado, 2B greedy, C3E calibrado, detector+CLIP, estimador conjunto) | ~8–10 h |

Os resultados vão para `eval_cache/platforms/<placa>/` e nunca sobrescrevem os do Orin. Para comparar:

```bash
bash thor/06_run_experiments.sh compare
```

Imprime a tabela in-domain com a coluna do Orin ao lado da do Thor, e a concordância segundo a segundo da evidência na Califórnia e em São Francisco.

**Por que os números podem mudar, e por que isso é um resultado:** a avaliação é *latency-honest* — cada sistema só vê os frames que teria visto no tempo real que levou. O Thor é mais rápido, então vê mais frames. As constantes temporais (janela, limiares) foram calibradas no Orin. Se a tabela mudar no Thor, é a mesma tese do paper (constantes temporais não transferem), agora entre **hardwares** em vez de entre modelos. Para o IV isso é material novo.

---

## 5. DRIVE AGX Thor (DriveOS)

Igual ao Jetson, exceto por três pontos.

**Compilação do Edge-LLM (no PC x86, dentro do Docker do DriveOS SDK 7.2):**

```bash
# no PC x86, com o repositório clonado em ~/jetson-deploy
mkdir -p ~/TensorRT-Edge-LLM
docker run -it --rm \
    -v ~/jetson-deploy:/root/jetson-deploy \
    -v ~/TensorRT-Edge-LLM:/root/TensorRT-Edge-LLM \
    <imagem-do-DriveOS-SDK-7.2>
# dentro do container:
cd /root/jetson-deploy && bash thor/02_build_edgellm.sh --in-docker
exit
# de volta no PC x86 (o build ficou em ~/TensorRT-Edge-LLM, montado do host):
rsync -a ~/TensorRT-Edge-LLM <usuario>@<IP-DA-DRIVE>:~/
```

Use o mesmo caminho (`~/TensorRT-Edge-LLM`) na placa: o pipeline procura o binário e o plugin ali.

Na placa, `bash thor/02_build_edgellm.sh` apenas lembra desse fluxo. Depois siga do passo 3 em diante. Os engines (passo 4) são construídos **na placa**, como no Jetson.

**PyTorch:** o DriveOS não traz PyTorch, e pode não haver wheel compatível. O `03_python_env.sh` tenta; se não houver, o que roda e o que não roda:

| funciona sem torch | precisa de torch |
|---|---|
| 2B e C3E (todos os canais de linguagem) | detector YOLO e baseline detector+CLIP |
| `06 ... sf` e os sistemas 2B/C3E do `indomain` | estimador conjunto ao vivo, `06 ... california` (usa o detector) |
| replay, tabelas, vídeos de revisão | TCN |

**Memória:** a documentação do Edge-LLM recomenda, no DriveOS, aumentar as huge pages para modelos maiores (não deve ser necessário para os nossos de 2B e 4B):

```bash
echo 15658 | sudo tee /proc/sys/vm/nr_hugepages
```

### Qwen-Drive-1.0-4B na DRIVE

No Orin o Qwen-Drive roda em PyTorch puro (BF16, `pipeline/dump_qwendrive_stream.py`), a ~2,9 s por pergunta de 8 tokens. Isso não vale na DRIVE: sem PyTorch, ele precisa virar engine TensorRT pelo Edge-LLM, como o 2B e o C3E. O Edge-LLM v0.9.0 suporta Qwen3.5, que é o VLM do Qwen-Drive sem mudança de arquitetura, mas:

1. a exportação (quantização INT4 e ONNX) roda num **PC x86 com GPU**, não na placa;
2. o checkpoint é `qwen_drive`, não `qwen3_5`; é preciso extrair só os pesos do VLM para um checkpoint Qwen3.5 antes de exportar;
3. depois disso, o ONNX entra no mesmo `04_build_engines.sh`, com 736×416 (299 tokens de imagem), como o C3E.

Os passos 1 e 2 ainda não foram feitos.

---

## 6. Armadilhas conhecidas

- **Um processo de modelo por vez.** Todos os scripts trocam requisições pelo mesmo arquivo, `/dev/shm/cascade_request.json`. Dois rodando juntos corrompem um ao outro em silêncio. O `06_run_experiments.sh` já roda tudo em sequência.
- **Resposta vazia não é "não".** Quando o engine falha, `PersistentEngine.infer()` devolve `None`, e os scripts tratam como resposta vazia. Se uma taxa de "sim" der 0% exato, desconfie e olhe o texto bruto antes de acreditar.
- **C3E: sempre 736×416.** Nunca redimensione proporcionalmente para o C3E.
- **Nunca copie engines entre placas.** Sempre reconstrua com `04_build_engines.sh`.
- **`ultralytics` fixado em 8.3.250.** Outra versão pode ler o engine com metadados diferentes.
- **Rode os experimentos com `nohup`.** Uma queda de SSH mata o processo; os scripts retomam de onde pararam (marcadores `.done`).

---

## 7. Trazer os resultados de volta

Os resultados do Thor ficam em `eval_cache/platforms/<placa>/` (só caches pequenos e logs). Para levar ao Orin ou ao paper:

```bash
git add eval_cache/platforms eval_cache/thor
git commit -m "Thor: smoke test e experimentos"
git push
```

---

## 8. Arquivos deste diretório

| arquivo | função |
|---|---|
| `config.sh` | configurações compartilhadas (edite só as 3 primeiras) |
| `00_preflight.sh` | diagnóstico, não muda nada |
| `01_sync_from_orin.sh` | copia modelos e dados do Orin por pacote |
| `02_build_edgellm.sh` | Edge-LLM v0.9.0 + patch + build |
| `03_python_env.sh` | `~/workzone/venv` |
| `04_build_engines.sh` | engines 2B, C3E e YOLO nesta placa |
| `05_smoke_test.sh` / `.py` | teste de fumaça, regressão vs Orin, latência |
| `06_run_experiments.sh` | experimentos do paper; `compare` gera as tabelas |
| `compare_platforms.py` | Orin × esta placa, só a partir dos caches |
| `edgellm_patch/` | patch do Edge-LLM (base v0.9.0, commit em `BASE_COMMIT.txt`) |
| `requirements-thor.txt` | dependências Python além do torch |
