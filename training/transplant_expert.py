"""Transplanta o expert de acao PRE-TREINADO do professor 10B pro nosso 2B.

Motivo (a descoberta que reorienta o projeto):
    alpamayo_r1/models/alpamayo_r1.py linha 94 faz
        self.expert = AutoModel.from_config(expert_config)
    ou seja, o expert nasce com pesos ALEATORIOS. Nossos 4 experimentos anteriores
    (GT direto, destilacao de trajetoria, KD de campo de velocidade, cotrain do VLM)
    mudavam o SINAL DE TREINO mas sempre partiram de um expert de 1.77B aleatorio,
    treinado em 3644 clipes -- ~500k parametros por exemplo. Nenhum alvo ensina a
    variedade de acoes nessas condicoes. Dai o "leque" de direcoes que vemos nas
    figuras: o expert nunca aprendeu como e' uma trajetoria de direcao plausivel.

    O expert do professor tem a MESMA arquitetura por camada (verificado: 24/24
    tensores com shapes identicas) mas foi pre-treinado pela NVIDIA no dataset
    inteiro. Reaproveita-lo e' o principio do VITA-VLA (arxiv 2510.09607):
    reutilizar um decodificador de acao pre-treinado em vez de treinar do zero.

Mapeamento de profundidade:
    professor = 36 camadas (espelha o VLM de 36); nosso = 28 (espelha o VLM de 28).
    O expert atende ao KV-cache do VLM CAMADA A CAMADA, entao a contagem e' amarrada
    ao backbone e nao da pra simplesmente copiar 36 em 28. Usamos amostragem
    uniformemente espacada (0->0, ..., 27->35), que preserva o "arco" de computacao
    do inicio ao fim, em vez de truncar e jogar fora as camadas finais.

Ressalva honesta: o expert do professor aprendeu a ler as features do VLM DELE
(4096-dim). As nossas sao 2048-dim. As shapes do KV-cache batem (kv_heads=8,
head_dim=128 nos dois), mas a semantica nao -- por isso ainda e' preciso fine-tune.
A aposta e' que partir de um prior de acao REAL bata partir de ruido.
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
# VLM congelado neste checkpoint => os pesos do VLM sao os do workzone-2b-stage7-1-hf
SRC = f"{BASE}/checkpoints/sft_action_2b_kd_velocity/checkpoint-684"
TEACHER = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
DST = f"{BASE}/checkpoints/expert_transplant_init"

os.makedirs(DST, exist_ok=True)


def load_sd(d):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for k, shard in idx.items():
        with safe_open(f"{d}/{shard}", framework="pt") as f:
            sd[k] = f.get_tensor(k)
    return sd


print("[1] Carregando pesos...")
ours = load_sd(SRC)
teach = load_sd(TEACHER)

n_ours = max(int(k.split(".")[2]) for k in ours if k.startswith("expert.layers.")) + 1
n_teach = max(int(k.split(".")[2]) for k in teach if k.startswith("expert.layers.")) + 1
print(f"    nosso expert: {n_ours} camadas | professor: {n_teach} camadas")

# camada i do aluno <- camada uniformemente espacada do professor
mapping = {i: round(i * (n_teach - 1) / (n_ours - 1)) for i in range(n_ours)}
print(f"[2] Mapeamento de profundidade: {mapping[0]}, {mapping[1]}, ..., "
      f"{mapping[n_ours - 2]}, {mapping[n_ours - 1]}")

new_sd = dict(ours)      # mantem o VLM (workzone 2B) intacto
n_layer, n_other = 0, 0

for si, ti in mapping.items():
    pref_s, pref_t = f"expert.layers.{si}.", f"expert.layers.{ti}."
    for k in [k for k in teach if k.startswith(pref_t)]:
        tgt = k.replace(pref_t, pref_s)
        if tgt in new_sd and new_sd[tgt].shape == teach[k].shape:
            new_sd[tgt] = teach[k].clone()
            n_layer += 1

# modulos fora das camadas: mesma shape nos dois, copia 1:1
for k in teach:
    if k.startswith(("expert.norm.", "action_in_proj.", "action_out_proj.", "diffusion.")):
        if k in new_sd and new_sd[k].shape == teach[k].shape:
            new_sd[k] = teach[k].clone()
            n_other += 1

print(f"[3] Transplantados: {n_layer} tensores de camada + {n_other} de modulos "
      f"(action_in_proj / action_out_proj / norm)")

# sanidade: o VLM NAO pode ter mudado
vlm_keys = [k for k in ours if k.startswith("vlm.")]
assert all(torch.equal(new_sd[k], ours[k]) for k in vlm_keys[:50]), "VLM foi alterado!"
print(f"[4] OK: {len(vlm_keys)} tensores do VLM preservados (nosso workzone 2B intacto)")

print("[5] Salvando...")
save_file({k: v.contiguous() for k, v in new_sd.items()},
          f"{DST}/model.safetensors", metadata={"format": "pt"})
for f in ("config.json", "generation_config.json"):
    if os.path.exists(f"{SRC}/{f}"):
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")

print(f"\n=== checkpoint com expert PRE-TREINADO em {DST} ===")
