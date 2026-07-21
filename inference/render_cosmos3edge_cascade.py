"""Renderiza o boston.mp4 com o painel da CascadeStateMachine (MESMO codigo
usado no pipeline de producao do 2B, cascade_state_machine.py) alimentado
pelas respostas do Cosmos3-Edge coletadas em 25 frames reais (cadencia 0.6s).

Diferenca do render_cascade_videos.py original: aqui as respostas vem
frescas (25 chamadas reais ao Cosmos3-Edge, uma por frame amostrado), nao de
um batch pre-computado do 2B. O texto do modelo inclui o raciocinio
("Got it, let's...") antes da resposta final -- extraimos so' o trecho apos
"</think>" quando presente (senao usamos o texto inteiro como fallback).
"""
import json
import re
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/data/wesleyferreiramaia/wokzone-alpamayo/mi3lab-workzone-vla/inference")
from cascade_state_machine import CascadeStateMachine, WZState, classify_ego, has_corroboration

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
DELIVER = f"{BASE}/outputs/cosmos3edge_trajectory_test"
VIDEO = f"{BASE}/videos/boston.mp4"
CADENCE_S = 0.6
OUT_PATH = f"{DELIVER}/boston_CASCADE_cosmos3edge.mp4"


def final_segment(text):
    """Extrai o trecho apos </think> (resposta final); se nao houver, usa o
    texto inteiro -- mesma logica usada nas legendas do video de trajetoria."""
    if "</think>" in text:
        seg = text.split("</think>")[-1].strip()
        if seg:
            return seg
    return text.strip()


def parse_gate_cosmos(text):
    """GATE do Cosmos3-Edge: raciocina em prosa livre, nao no template do 2B
    ('N temporary signs detected'). Olha o SEGMENTO FINAL por yes/no."""
    seg = final_segment(text).lower()
    if seg.startswith("yes") or seg == "yes" or "\nyes" in seg or seg.endswith("yes"):
        return True
    if seg.startswith("no") or seg == "no" or "\nno" in seg or seg.endswith("no"):
        return False
    # fallback: procura yes/no isolado nas ultimas 2 frases
    tail = seg[-60:]
    if re.search(r"\byes\b", tail):
        return True
    if re.search(r"\bno\b", tail):
        return False
    return None


# ---------------------------------------------------------------- dados
raw = json.load(open(f"{DELIVER}/cosmos3edge_cascade_frames_raw.json"))
n_samples = len(raw)
print(f"[dados] {n_samples} amostras carregadas (cadencia {CADENCE_S}s)")

STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40, 40, 40),  "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80, 60, 0),   "fg": (255, 200, 50),  "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80, 10, 10),  "fg": (255, 80, 80),   "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10, 60, 10),  "fg": (80, 230, 80),   "label": "EXITING"},
}
PANEL_W = 680

try:
    F = "/usr/share/fonts/truetype/dejavu/DejaVuSans-%s.ttf"
    font_status = ImageFont.truetype(F % "Bold", 30)
    font_mid    = ImageFont.truetype(F % "Bold", 17)
    font_desc   = ImageFont.truetype(F % "Regular", 16)
    font_bar    = ImageFont.truetype(F % "Regular", 14)
except Exception:
    font_status = font_mid = font_desc = font_bar = ImageFont.load_default()

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(f"{DELIVER}/_cascade_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W + PANEL_W, H))

sm = CascadeStateMachine()
cur_state = sm.state
cur_gate = False
cur_corr = False
cur_desc = "Analyzing..."
cur_ego_raw = ""
transitions = []

print(f"[render] {total_fr} frames do video, {n_samples} amostras a cada {CADENCE_S}s")
fi = 0
si = 0
next_sample_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    sec = fi / fps

    if si < n_samples and fi >= next_sample_frame:
        idx = str(si + 1)
        qa = raw.get(idx, {})
        gate = parse_gate_cosmos(qa.get("GATE", ""))
        ego_final = final_segment(qa.get("EGO", ""))
        desc_final = final_segment(qa.get("DESC", ""))
        sign_final = final_segment(qa.get("SIGN", ""))

        prev = sm.state
        cur_state = sm.update(gate, ego_final, desc_final, sign_final)
        cur_gate = bool(gate)
        cur_corr = has_corroboration(desc_final, sign_final)
        cur_desc = desc_final
        cur_ego_raw = ego_final
        if cur_state != prev:
            transitions.append((sec, prev.value, cur_state.value))
            print(f"  t={sec:5.1f}s [TRANSICAO] {prev.value.upper()} -> {cur_state.value.upper()}")
        else:
            print(f"  t={sec:5.1f}s gate={gate} estado={cur_state.value} ego='{ego_final[:40]}'")

        si += 1
        next_sample_frame = int(round(si * CADENCE_S * fps))

    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw = ImageDraw.Draw(panel)
    style = STATE_STYLE[cur_state]
    draw.rectangle([0, 0, PANEL_W, 58], fill=style["bg"])
    draw.text((14, 12), style["label"], font=font_status, fill=style["fg"])

    y = 66
    gate_n = sum(1 for x in sm.gate_window if x)
    draw.text((14, y), f"GATE  (janela {gate_n}/{len(sm.gate_window) or 1}): "
                       f"{'YES' if cur_gate else 'no'}",
              font=font_mid, fill=(120, 220, 120) if cur_gate else (150, 150, 150))
    y += 26
    draw.text((14, y), f"CORROBORACAO (DESC/SIGN): {'sim' if cur_corr else 'nao'}",
              font=font_mid, fill=(120, 220, 120) if cur_corr else (150, 150, 150))
    y += 34

    draw.text((14, y), "EGO (Cosmos3-Edge, resposta final):", font=font_bar, fill=(150, 200, 255))
    y += 20
    for line in textwrap.wrap(cur_ego_raw, width=52)[:3]:
        draw.text((14, y), line, font=font_desc, fill=(200, 220, 255))
        y += 21
    y += 10

    draw.text((14, y), "WHAT THE MODEL SEES (DESC):", font=font_bar, fill=(255, 200, 80))
    y += 20
    for line in textwrap.wrap(cur_desc, width=52)[:7]:
        draw.text((14, y), line, font=font_desc, fill=(235, 235, 210))
        y += 21

    draw.rectangle([0, H - 44, PANEL_W, H - 22], fill=(20, 30, 20))
    draw.text((14, H - 40), "Cosmos3-Edge (de fabrica, sem fine-tuning) + CascadeStateMachine",
              font=font_bar, fill=(120, 230, 160))
    progress = fi / max(total_fr, 1)
    draw.rectangle([0, H - 22, PANEL_W, H], fill=(20, 20, 20))
    draw.rectangle([0, H - 22, int(PANEL_W * progress), H], fill=(60, 60, 60))
    draw.text((6, H - 20), f"{sec:.1f}s / {total_fr/fps:.1f}s  (amostra {si}/{n_samples})",
              font=font_bar, fill=(210, 210, 210))

    combined = np.concatenate([frame, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)], axis=1)
    writer.write(combined)
    fi += 1

cap.release()
writer.release()

print()
if not transitions:
    print("ATENCAO: nenhuma transicao de estado ao longo do video")
else:
    print(f"{len(transitions)} transicoes de estado detectadas")
print(f"video bruto salvo -> {DELIVER}/_cascade_raw.mp4 (precisa reencodar p/ h264+faststart)")
