"""Renderiza os 5 videos de benchmark (4 de obra + comma2k19 sem obra) com o
modelo stage7_1 (INT4/TensorRT) + CascadeStateMachine (design evidence-first).

Substitui o painel do filtro Bayesiano puro: mostra o estado da cascade, o
status do gate (janela k-de-n), corroboracao, e o texto da DESC.
Saida: outputs/v17_cascade/*.mp4
"""
import os, sys, json, textwrap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla/inference")
from cascade_state_machine import (CascadeStateMachine, WZState, parse_gate,
                                   has_corroboration)

E6 = f"{BASE}/models/workzone-2b-stage6-engines"
E7 = f"{BASE}/models/workzone-2b-stage7-1-engines"
OUT_DIR = f"{BASE}/outputs/v17_cascade"
os.makedirs(OUT_DIR, exist_ok=True)


def clean(t):
    return t.replace(".DATA", "").strip()


import re
_CITY_RE = re.compile(
    r"\s*Scene:\s*\w+\s+environment\s+in\s+[^,]+,\s*[^,]*conditions,\s*[^.]*lighting\.\s*",
    re.IGNORECASE)


def strip_city(t):
    return _CITY_RE.sub(" ", t).strip()


# ---------------------------------------------------------------- dados
gate_out = json.load(open(f"{E7}/output_gate_batch_s71.json"))
gate_meta = json.load(open(f"{E6}/input_gate_batch_meta.json"))
gate = {(m["source"], m["clip_id"], m["frame_idx"]): parse_gate(r["output_text"])
        for r, m in zip(gate_out["responses"], gate_meta)}


def load_answers(out_f, meta_f):
    out = json.load(open(out_f))
    meta = json.load(open(meta_f))
    d = {}
    for r, m in zip(out["responses"], meta):
        d.setdefault((m["city"], m["frame_idx"]), {})[m["question"]] = clean(r["output_text"])
    return d


wz_answers = load_answers(f"{E7}/output_video_batch_s71.json",
                          f"{E6}/input_video_batch_meta.json")
nc_answers = load_answers(f"{E7}/output_negcontrol_video_batch_s71.json",
                          f"{E6}/input_negcontrol_video_batch_meta.json")

VIDEOS = [
    # (clip_id no batch, source do gate, arquivo de video, answers dict)
    ("seattle_unseen", "workzone_videos", f"{BASE}/videos/seattle_unseen.mp4", wz_answers),
    ("boston", "workzone_videos", f"{BASE}/videos/boston.mp4", wz_answers),
    ("denver", "workzone_videos", f"{BASE}/videos/denver.mp4", wz_answers),
    ("chicago", "workzone_videos", f"{BASE}/videos/chicago.mp4", wz_answers),
    ("comma2k19_highway_night", "comma2k19",
     f"{BASE}/videos_negcontrol/comma2k19_highway_night.mp4", nc_answers),
]

STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40, 40, 40),  "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80, 60, 0),   "fg": (255, 200, 50),  "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80, 10, 10),  "fg": (255, 80, 80),   "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10, 60, 10),  "fg": (80, 230, 80),   "label": "EXITING"},
}
PANEL_W = 680

try:
    F = "/usr/share/fonts/liberation-mono/LiberationMono-%s.ttf"
    font_status = ImageFont.truetype(F % "Bold", 30)
    font_mid    = ImageFont.truetype(F % "Bold", 17)
    font_desc   = ImageFont.truetype(F % "Regular", 17)
    font_bar    = ImageFont.truetype(F % "Regular", 14)
except Exception:
    font_status = font_mid = font_desc = font_bar = ImageFont.load_default()

for clip_id, gate_src, video_path, answers in VIDEOS:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = f"{OUT_DIR}/cascade_{clip_id}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PANEL_W, H))

    sample_idxs = sorted(fi for (c, fi) in answers if c == clip_id)
    sm = CascadeStateMachine()
    cur_state = sm.state
    cur_gate = False
    cur_corr = False
    cur_text = "Analyzing..."
    transitions = []

    print(f"[{clip_id}] {total_fr} frames, {len(sample_idxs)} amostras")
    si = 0
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sec = fi / fps

        if si < len(sample_idxs) and fi >= sample_idxs[si]:
            qa = answers[(clip_id, sample_idxs[si])]
            g = gate.get((gate_src, clip_id, sample_idxs[si]))
            prev = sm.state
            cur_state = sm.update(g, qa.get("EGO", ""), qa.get("DESC", ""), qa.get("SIGN", ""))
            cur_gate = bool(g)
            cur_corr = has_corroboration(qa.get("DESC", ""), qa.get("SIGN", ""))
            if "DESC" in qa:
                cur_text = strip_city(qa["DESC"])
            if cur_state != prev:
                transitions.append((sec, prev.value, cur_state.value))
                print(f"  t={sec:5.1f}s [TRANSITION] {prev.value.upper()} -> {cur_state.value.upper()}")
            si += 1

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

        draw.text((14, y), "WHAT THE MODEL SEES:", font=font_bar, fill=(255, 200, 80))
        y += 20
        for line in textwrap.wrap(cur_text, width=42)[:8]:
            draw.text((14, y), line, font=font_desc, fill=(235, 235, 210))
            y += 22

        draw.rectangle([0, H - 44, PANEL_W, H - 22], fill=(20, 30, 20))
        tag = "CONTROLE NEGATIVO (sem obra)" if gate_src == "comma2k19" else "video de obra"
        draw.text((14, H - 40), f"stage7_1 INT4 + cascade -- {tag}",
                  font=font_bar, fill=(120, 230, 160))
        progress = fi / max(total_fr, 1)
        draw.rectangle([0, H - 22, PANEL_W, H], fill=(20, 20, 20))
        draw.rectangle([0, H - 22, int(PANEL_W * progress), H], fill=(60, 60, 60))
        draw.text((6, H - 20), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_bar,
                  fill=(210, 210, 210))

        combined = np.concatenate([frame, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)], axis=1)
        writer.write(combined)
        fi += 1

    cap.release()
    writer.release()
    if not transitions and gate_src != "comma2k19":
        print(f"  ATENCAO: nenhuma transicao em video de obra!")
    print(f"  salvo -> {out_path}\n")

print("=== RENDER CASCADE CONCLUIDO ===")
