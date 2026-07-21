"""Renderiza o painel de trajetoria como VIDEO animado (nao imagem estatica):
desenha progressivamente as trajetorias previstas (Cosmos3-Edge x2 modos +
Alpamayo, 5 amostras cada) sincronizado com o tempo real do video principal.

Anima do frame 0 ate ANIM_SECONDS (duracao real dos dados de acao, ~6-6.4s),
depois congela no resultado final ate o fim do video (TOTAL_SECONDS).
Eixos fixos desde o primeiro frame (calculados do dado COMPLETO) pra nao
pular/re-escalar durante a animacao.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELIVER = sys.argv[1]
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
TOTAL_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
ANIM_SECONDS = 6.4  # horizonte real dos dados de acao (60 passos @ 10Hz = 6.0s pro
                    # Cosmos3-Edge, 64 passos @ 10Hz = 6.4s pro Alpamayo)

FRAMES_DIR = f"{DELIVER}/_traj_anim_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

inv5 = np.array(json.load(open(f"{DELIVER}/cosmos3edge_inverse_dynamics_5samples.json")))[:, :, :3]  # [5,60,3]
pol5 = np.array(json.load(open(f"{DELIVER}/cosmos3edge_policy_5samples.json")))[:, :, :3]  # [5,60,3]
alp = json.load(open(f"{DELIVER}/alpamayo_boston_trajectory.json"))
alp_xyz = np.array(alp["teacher_10b_xyz"])  # [5,64,3] metros

n_cosmos_steps = inv5.shape[1]   # 60
n_alp_steps = alp_xyz.shape[1]   # 64

# cumsum pra cosmos (deltas por passo); alpamayo ja' vem em posicao absoluta
cum_inv = np.concatenate([np.zeros((5, 1, 3)), np.cumsum(inv5, axis=1)], axis=1)  # [5,61,3]
cum_pol = np.concatenate([np.zeros((5, 1, 3)), np.cumsum(pol5, axis=1)], axis=1)  # [5,61,3]

# limites fixos calculados do dado COMPLETO (eixo nao pula durante a animacao)
all_x_top = np.concatenate([cum_inv[:, :, 0].ravel(), cum_pol[:, :, 0].ravel()])
all_y_top = np.concatenate([cum_inv[:, :, 2].ravel(), cum_pol[:, :, 2].ravel()])
xlim_top = (all_x_top.min() - 1, all_x_top.max() + 1)
ylim_top = (all_y_top.min() - 1, all_y_top.max() + 1)

all_x_bot = alp_xyz[:, :, 1].ravel()
all_y_bot = alp_xyz[:, :, 0].ravel()
xlim_bot = (all_x_bot.min() - 2, all_x_bot.max() + 2)
ylim_bot = (all_y_bot.min() - 2, all_y_bot.max() + 2)

n_total_frames = int(TOTAL_SECONDS * FPS)
n_anim_frames = int(ANIM_SECONDS * FPS)

print(f"renderizando {n_total_frames} frames ({FPS}fps, {TOTAL_SECONDS}s total, "
     f"animando ate {ANIM_SECONDS}s)...", flush=True)

for f in range(n_total_frames):
    t = f / FPS
    progress = min(t / ANIM_SECONDS, 1.0)
    k_cosmos = max(1, int(progress * n_cosmos_steps))
    k_alp = max(1, int(progress * n_alp_steps))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(4.4, 8), dpi=100)
    fig.patch.set_facecolor((0.06, 0.06, 0.06))

    for j in range(5):
        ax_top.plot(cum_inv[j, :k_cosmos + 1, 0], cum_inv[j, :k_cosmos + 1, 2],
                    "-", lw=1.6, color="#3fd67a", alpha=.85,
                    label="inverse_dynamics\n(video real)" if j == 0 else None)
        ax_top.plot(cum_pol[j, :k_cosmos + 1, 0], cum_pol[j, :k_cosmos + 1, 2],
                    "-", lw=1.6, color="#c77dff", alpha=.85,
                    label="policy\n(futuro, sem historico)" if j == 0 else None)
    ax_top.scatter([0], [0], color="white", s=35, zorder=5)
    ax_top.set_xlim(*xlim_top)
    ax_top.set_ylim(*ylim_top)
    ax_top.set_title("Cosmos3-Edge (5 amostras cada modo)\nunidades do modelo",
                     fontsize=10, color="white")
    ax_top.grid(alpha=.25, color="gray")
    ax_top.legend(fontsize=7, loc="upper left", labelcolor="white", facecolor=(0.15, 0.15, 0.15), framealpha=.6)
    ax_top.set_facecolor((0.06, 0.06, 0.06))
    ax_top.tick_params(labelsize=7, colors="white")
    for spine in ax_top.spines.values():
        spine.set_color("gray")

    for j in range(5):
        ax_bot.plot(alp_xyz[j, :k_alp, 1], alp_xyz[j, :k_alp, 0],
                   "-", lw=1.6, color="#ffb347", alpha=.85)
    ax_bot.scatter([0], [0], color="white", s=35, zorder=5)
    ax_bot.set_xlim(*xlim_bot)
    ax_bot.set_ylim(*ylim_bot)
    ax_bot.set_title("Alpamayo 10B (5 amostras)\nmetros, historico FABRICADO 10m/s",
                     fontsize=10, color="white")
    ax_bot.grid(alpha=.25, color="gray")
    ax_bot.set_facecolor((0.06, 0.06, 0.06))
    ax_bot.tick_params(labelsize=7, colors="white")
    for spine in ax_bot.spines.values():
        spine.set_color("gray")

    fig.suptitle(f"t={t:.1f}s" + ("" if progress < 1.0 else "  (trajetoria completa)"),
                fontsize=9, color="#aaaaaa")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{FRAMES_DIR}/frame_{f:04d}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    if f % 30 == 0:
        print(f"  frame {f}/{n_total_frames}", flush=True)

print(f"OK: {n_total_frames} frames em {FRAMES_DIR}/")
