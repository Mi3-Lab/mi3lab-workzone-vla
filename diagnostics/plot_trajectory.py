"""Plota a trajetoria prevista (60x9: 3D translacao + 6D rotacao) em vista de
cima (top-down), acumulando os deltas por passo. So' leitura de um JSON local,
sem GPU -- roda dentro do job pra respeitar a regra de nada no no de login.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

action_path, out_path = sys.argv[1], sys.argv[2]
action = np.array(json.load(open(action_path)))  # [60, 9]
translation = action[:, :3]  # tx, ty, tz por passo

cumulative = np.cumsum(translation, axis=0)
cumulative = np.vstack([[0, 0, 0], cumulative])  # incluir o ponto de partida

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(cumulative[:, 0], cumulative[:, 2], "o-", markersize=3, color="tab:blue")
ax.scatter([cumulative[0, 0]], [cumulative[0, 2]], color="green", s=80, label="inicio", zorder=5)
ax.scatter([cumulative[-1, 0]], [cumulative[-1, 2]], color="red", s=80, label="fim (6s)", zorder=5)
ax.set_xlabel("tx acumulado (lateral)")
ax.set_ylabel("tz acumulado (longitudinal / 'pra frente')")
ax.set_title("Trajetoria prevista (vista de cima)\ncumsum dos deltas por passo -- NAO validada contra GPS real")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
steps = np.arange(len(translation))
ax.plot(steps, translation[:, 0], label="tx (lateral)")
ax.plot(steps, translation[:, 1], label="ty")
ax.plot(steps, translation[:, 2], label="tz (longitudinal)")
ax.set_xlabel("passo (10 FPS, 6s total)")
ax.set_ylabel("delta por passo")
ax.set_title("Componentes de translacao por passo")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"salvo em {out_path}")
