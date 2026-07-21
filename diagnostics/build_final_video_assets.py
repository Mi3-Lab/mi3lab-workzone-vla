"""Gera os assets pro video final composto (boston.mp4 inteiro + legendas com
as 5 respostas do reasoner + painel de trajetoria com as 5 amostras de cada
fonte): 5 imagens de legenda (uma por pergunta/resposta) + 1 painel de
trajetoria (PNG com canal alfa, pra overlay via ffmpeg).

Nao usa drawtext do ffmpeg com texto do modelo diretamente no comando de
shell (risco de caracteres especiais/injecao) -- todo texto vira imagem aqui,
o ffmpeg so' faz overlay de PNGs com filtros numericos (enable=between(t,a,b)).
"""
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DELIVER = sys.argv[1]
VIDEO_W, VIDEO_H = 722, 406  # resolucao real do boston.mp4

# ---------- 1. legendas (5 QA, uma imagem por pergunta) ----------
qa = json.load(open(f"{DELIVER}/cosmos3edge_qa_fresh.json"))
order = ["gate_test", "ego_test", "active_test", "sign_test", "desc_test"]
labels = {"gate_test": "GATE", "ego_test": "EGO", "active_test": "ACTIVE",
         "sign_test": "SIGN", "desc_test": "DESC"}

try:
    font_q = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    font_a = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    font_q = font_a = ImageFont.load_default()

for i, key in enumerate(order):
    prompt = qa[key]["prompt"]
    answer = qa[key]["answer"]
    # remove o raciocinio "</think>" quando presente, mostra so' a resposta final
    # (mais legivel numa legenda curta -- o raciocinio completo fica nos JSONs)
    answer_clean = answer.split("</think>")[-1].strip() or answer.strip()
    answer_short = textwrap.shorten(answer_clean.replace("\n", " "), width=140, placeholder="...")

    bar_h = 90
    img = Image.new("RGBA", (VIDEO_W, bar_h), (0, 0, 0, 190))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), f"[{labels[key]}] {prompt}", font=font_q, fill=(255, 210, 80, 255))
    for j, line in enumerate(textwrap.wrap(answer_short, width=78)):
        draw.text((10, 34 + j * 18), line, font=font_a, fill=(255, 255, 255, 255))
    img.save(f"{DELIVER}/caption_{i}.png")
    print(f"legenda {i} ({labels[key]}) salva: {answer_short[:60]}...")

# ---------- 2. painel de trajetoria (5 amostras por fonte) ----------
inv5 = np.array(json.load(open(f"{DELIVER}/cosmos3edge_inverse_dynamics_5samples.json")))[:, :, :3]  # [5,60,3]
pol5 = np.array(json.load(open(f"{DELIVER}/cosmos3edge_policy_5samples.json")))[:, :, :3]  # [5,60,3]
alp = json.load(open(f"{DELIVER}/alpamayo_boston_trajectory.json"))
alp_xyz = np.array(alp["teacher_10b_xyz"])  # [5,T,3] metros

fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=130)

ax = axes[0]
for j in range(5):
    cum_inv = np.vstack([[0, 0, 0], np.cumsum(inv5[j], axis=0)])
    cum_pol = np.vstack([[0, 0, 0], np.cumsum(pol5[j], axis=0)])
    ax.plot(cum_inv[:, 0], cum_inv[:, 2], "-", lw=1.3, color="tab:green", alpha=.7,
           label="inverse_dynamics (video real)" if j == 0 else None)
    ax.plot(cum_pol[:, 0], cum_pol[:, 2], "-", lw=1.3, color="tab:purple", alpha=.7,
           label="policy (futuro, sem historico)" if j == 0 else None)
ax.scatter([0], [0], color="k", s=40, zorder=5)
ax.set_title("Cosmos3-Edge (5 amostras cada)\nunidades do modelo", fontsize=9)
ax.axis("equal")
ax.grid(alpha=.3)
ax.legend(fontsize=6.5, loc="best")
ax.tick_params(labelsize=7)

ax = axes[1]
for j in range(5):
    ax.plot(alp_xyz[j, :, 1], alp_xyz[j, :, 0], "-", lw=1.3, color="tab:orange", alpha=.75)
ax.plot(0, 0, marker="s", ms=7, color="k")
ax.set_title("Alpamayo 10B (5 amostras)\nmetros, historico FABRICADO 10m/s", fontsize=9)
ax.grid(alpha=.3)
ax.tick_params(labelsize=7)
ax.set_aspect("equal", adjustable="datalim")

fig.patch.set_alpha(0.85)
fig.tight_layout()
fig.savefig(f"{DELIVER}/traj_panel.png", dpi=130, transparent=False,
           facecolor=(0.06, 0.06, 0.06))
plt.close(fig)
print("painel de trajetoria salvo")
