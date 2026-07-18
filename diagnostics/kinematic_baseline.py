"""Baseline CINEMATICO (zero visao): quanto da' pra pontuar so extrapolando o ego?

O experimento de controle que faltou. Sete abordagens (GT-direto, destilacao,
KD de velocidade, cotrain, transplante, ResNet+DETR, escala de dados) convergiram
todas em minADE 4.57-5.12 no chunk 3126, contra 1.26 do professor. A pesquisa
externa mostrou que a propria NVIDIA reporta ganhos de dados continuando ate 2M
exemplos de treino — nos temos 3.6k (550x menos).

Hipotese do "plato cinematico": com tao pouco dado, TODO modelo aprende apenas a
extrapolar o historico do ego (seguir em frente na velocidade atual, com leve
correcao), e a leitura VISUAL da estrada nunca emerge. Se for verdade, um
baseline sem imagem nenhuma — 6 trajetorias de acao constante integradas pelo
mesmo action space — deve pontuar ~4.5-5.5, IGUAL aos nossos modelos treinados.

Leitura do resultado:
  - baseline ~= 4.6-5.1  => nossos 7 modelos nao aprenderam NADA visual; o plato
    e' o teto da cinematica pura. O gap 4.6 -> 1.26 e' TODO capacidade visual de
    trajetoria, que so emerge com ordens de magnitude mais dados.
  - baseline >> 5.5      => os modelos aprenderam algo visual sim, e o plato tem
    outra causa (a investigar).

Evidencias previas consistentes com a hipotese: minADE@0.5s otimo (0.05m, puro
kinematics) vs @5s pessimo (3.0m, exige ler a estrada); e o "leque" de direcoes
nas figuras BEV — o modelo nao sabe pra onde a via vai.

So CPU (nao ha modelo!); as 6 acoes constantes sao integradas pelo MESMO
UnicycleAccelCurvatureActionSpace dos modelos, entao a comparacao e' exata.
"""
import json
import os
import sys

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla")
sys.path.insert(0, f"{BASE}/alpamayo-recipes/recipes")
os.environ.setdefault("HF_HOME", f"{BASE}/.hf_cache")

import torch
from alpamayo.data.pai import PAIDataset
from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace

ACT_CFG = {k: v for k, v in json.load(open(
    f"{BASE}/checkpoints/sft_action_2b_kd_velocity/checkpoint-684/config.json"
))["action_space_cfg"].items() if k != "_target_"}
space = UnicycleAccelCurvatureActionSpace(**ACT_CFG)

# 6 "modos" fixos em espaco NORMALIZADO de acao (mesma normalizacao dos modelos):
# (accel, curvatura). (0,0) = acelera/curva na MEDIA do dataset ~= segue reto na
# velocidade atual. Os outros: curvas suaves p/ cada lado e leve frenagem.
CANDIDATES = [
    (0.0, 0.0),
    (0.0, +0.5),   # 0.5 sigma de curvatura ~= raio ~77m
    (0.0, -0.5),
    (0.0, +1.0),   # 1.0 sigma ~= raio ~38m
    (0.0, -1.0),
    (-0.5, 0.0),   # leve frenagem
]

ds = PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=[3126],
                num_history_steps=16, num_future_steps=64, time_step=0.1,
                use_default_keyframe=True)
print(f"[setup] {len(ds)} clipes de validacao (chunk 3126) | {len(CANDIDATES)} modos fixos\n")

tot, per_t = 0.0, {5: 0.0, 10: 0.0, 30: 0.0, 50: 0.0}
n = 0
for i in range(len(ds)):
    s = ds[i]
    if s is None:
        continue
    hx, hr = s["ego_history_xyz"][0], s["ego_history_rot"][0]      # [16,3], [16,3,3]
    gt = s["ego_future_xyz"][0]                                     # [64,3]

    best, best_traj = 9e9, None
    for a, k in CANDIDATES:
        act = torch.tensor([[a, k]]).repeat(64, 1)[None]            # [1,64,2]
        xyz, _ = space.action_to_traj(act, hx[None], hr[None])      # [1,64,3]
        ade = (xyz[0, :, :2] - gt[:, :2]).norm(dim=-1).mean().item()
        if ade < best:
            best, best_traj = ade, xyz[0]

    tot += best
    n += 1
    for t in per_t:   # indices 5,10,30,50 ~= 0.5s,1.0s,3.0s,5.0s (10Hz, ate' idx t)
        per_t[t] += (best_traj[:t, :2] - gt[:t, :2]).norm(dim=-1).mean().item()
    if n % 20 == 0:
        print(f"  [{n}] minADE parcial: {tot/n:.4f}m", flush=True)

print(f"\n=== BASELINE CINEMATICO (sem visao, 6 modos fixos) ===")
print(f"minADE_6 @6.4s : {tot/n:.4f}m   (n={n})")
for t, lbl in [(5, "0.5s"), (10, "1.0s"), (30, "3.0s"), (50, "5.0s")]:
    print(f"minADE @{lbl}   : {per_t[t]/n:.4f}m")
print("\nreferencias: nossos modelos treinados 4.57-5.12 | professor 10B 1.262")
