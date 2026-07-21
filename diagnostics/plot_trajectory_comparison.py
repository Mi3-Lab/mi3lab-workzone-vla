"""Plot final comparando as 3 trajetorias obtidas pra cena boston/workzone:
1. Cosmos3-Edge inverse_dynamics -- movimento PASSADO recuperado do video REAL
2. Cosmos3-Edge policy -- planejamento FUTURO a partir de 1 imagem, sem historico
3. Alpamayo 10B -- planejamento FUTURO, historico FABRICADO (reta a 10m/s)

IMPORTANTE: unidades do Cosmos3-Edge nao sao confirmadas como metros (docs nao
especificam). O Alpamayo esta confirmadamente em metros reais (x=frente,
y=lateral). Por isso o eixo do Cosmos3-Edge fica em "unidades do modelo",
nao em metros, pra nao sugerir uma equivalencia que nao foi verificada.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELIVER = sys.argv[1]

inv = np.array(json.load(open(f"{DELIVER}/predicted_action_60x9.json")))[:, :3]
pol = np.array(json.load(open(f"{DELIVER}/cosmos3edge_policy_action_60x9.json")))[:, :3]
alp = json.load(open(f"{DELIVER}/alpamayo_boston_trajectory.json"))
alp_xyz = np.array(alp["teacher_10b_xyz"])  # [K,T,3] metros, x=frente,y=lateral

cum_inv = np.vstack([[0, 0, 0], np.cumsum(inv, axis=0)])
cum_pol = np.vstack([[0, 0, 0], np.cumsum(pol, axis=0)])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(cum_inv[:, 0], cum_inv[:, 2], "o-", ms=3, color="tab:green",
       label="Cosmos3-Edge inverse_dynamics\n(passado, video REAL)")
ax.plot(cum_pol[:, 0], cum_pol[:, 2], "o-", ms=3, color="tab:purple",
       label="Cosmos3-Edge policy\n(futuro, sem historico)")
ax.scatter([0], [0], color="k", s=80, zorder=5, label="origem (t=0)")
ax.set_xlabel("eixo 1 acumulado (unidades do modelo)")
ax.set_ylabel("eixo 3 acumulado (unidades do modelo)")
ax.set_title("Cosmos3-Edge (unidades NAO confirmadas como metros)")
ax.axis("equal")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="best")

ax = axes[1]
for j, p in enumerate(alp_xyz):
    ax.plot(p[:, 1], p[:, 0], color="tab:orange", lw=1.6, alpha=.85,
           label="Alpamayo 10B (5 amostras)\nhistorico FABRICADO 10m/s" if j == 0 else None)
ax.plot(0, 0, marker="s", ms=9, color="k")
ax.set_xlabel("lateral (m)   <- esquerda | direita ->")
ax.set_ylabel("frente (m)")
ax.set_title("Alpamayo 10B (metros reais, x=frente/y=lateral confirmado)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="best")
ax.set_aspect("equal", adjustable="datalim")

fig.suptitle("Cena boston (workzone) -- 3 trajetorias, 3 tarefas diferentes, "
             "NAO comparaveis geometricamente 1:1 (ver ressalvas no relatorio)",
             fontsize=11, color="#b00000", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{DELIVER}/trajectory_comparison_final.png", dpi=130)
print(f"salvo em {DELIVER}/trajectory_comparison_final.png")
