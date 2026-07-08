"""Renderiza o video anotado do controle negativo (comma2k19, rodovia SEM
obra, confirmado visualmente limpo) com o MESMO painel/filtro Bayesiano do
render_trt_video.py -- pra visualizar a trajetoria real do estado EGO.
"""
import os, json, textwrap, re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import Enum

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"


class WZState(Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    INSIDE = "inside"
    EXITING = "exiting"


STATES = [WZState.OUTSIDE, WZState.APPROACHING, WZState.INSIDE, WZState.EXITING]
STATE_NAMES = ["OUTSIDE", "APPROACHING", "INSIDE", "EXITING"]
EGO_SHORT = ["OUTSIDE", "APPROACHING", "INSIDE"]
PEAK = 0.80


def peaked_dist(idx, n):
    p = np.full(n, (1.0 - PEAK) / max(n - 1, 1))
    p[idx] = PEAK
    return p


EGO_EMIT = np.array([
    [0.85, 0.12, 0.03], [0.10, 0.75, 0.15], [0.05, 0.55, 0.40], [0.50, 0.25, 0.25],
])
TRAFFIC_EMIT = np.array([
    [0.10, 0.03, 0.02, 0.85], [0.50, 0.14, 0.13, 0.23],
    [0.45, 0.14, 0.11, 0.30], [0.30, 0.06, 0.05, 0.59],
])
ACTIVE_EMIT = np.array([
    [0.10, 0.90], [0.45, 0.55], [0.50, 0.50], [0.25, 0.75],
])
SIGN_EMIT = np.array([
    [0.05, 0.95], [0.45, 0.55], [0.30, 0.70], [0.10, 0.90],
])
SIGN_WZ_KEYWORDS = [
    "ROAD WORK", "WORK ZONE", "DETOUR", "LANE CLOSED", "ROAD CLOSED",
    "SIDEWALK CLOSED", "UTILITY WORK", "SHOULDER WORK", "FLAGGER",
    "BE PREPARED TO STOP", "LANE ENDS", "LANE SHIFT",
]
TRANS = np.array([
    [0.92, 0.08, 0.00, 0.00], [0.02, 0.88, 0.10, 0.00],
    [0.00, 0.00, 0.98, 0.02], [0.06, 0.00, 0.06, 0.88],
])
PRIOR = np.array([0.55, 0.20, 0.20, 0.05])


def classify_answer(answer, labels):
    a = answer.lower()
    for i, lab in enumerate(labels):
        if lab.lower() in a:
            return i
    return None


def parse_sign(sign_txt):
    up = sign_txt.upper()
    if any(k in up for k in SIGN_WZ_KEYWORDS):
        clean = sign_txt.replace("Signs read:", "").strip().rstrip(".")
        return peaked_dist(0, 2), clean
    return np.ones(2) / 2, None


_CITY_CLAUSE_RE = re.compile(
    r"\s*Scene:\s*\w+\s+environment\s+in\s+[^,]+,\s*[^,]*conditions,\s*[^.]*lighting\.\s*",
    re.IGNORECASE,
)


def strip_city_clause(text):
    return _CITY_CLAUSE_RE.sub(" ", text).strip()


def clean_gen(text):
    return text.replace(".DATA", "").strip()


class BayesFilter:
    def __init__(self):
        self.belief = PRIOR.copy()

    def update(self, emission):
        b = (self.belief @ TRANS) * emission
        self.belief = b / b.sum()
        return self.belief


class DisplayPolicy:
    COMMIT = 0.60
    REFUTE = 0.15
    NEXT_HOP = {
        (WZState.OUTSIDE, WZState.APPROACHING): WZState.APPROACHING,
        (WZState.OUTSIDE, WZState.INSIDE): WZState.APPROACHING,
        (WZState.OUTSIDE, WZState.EXITING): WZState.APPROACHING,
        (WZState.APPROACHING, WZState.OUTSIDE): WZState.OUTSIDE,
        (WZState.APPROACHING, WZState.INSIDE): WZState.INSIDE,
        (WZState.APPROACHING, WZState.EXITING): WZState.INSIDE,
        (WZState.INSIDE, WZState.EXITING): WZState.EXITING,
        (WZState.INSIDE, WZState.OUTSIDE): WZState.EXITING,
        (WZState.INSIDE, WZState.APPROACHING): WZState.EXITING,
        (WZState.EXITING, WZState.OUTSIDE): WZState.OUTSIDE,
        (WZState.EXITING, WZState.APPROACHING): WZState.OUTSIDE,
        (WZState.EXITING, WZState.INSIDE): WZState.INSIDE,
    }

    def __init__(self):
        self.state = WZState.OUTSIDE

    def update(self, belief):
        target = STATES[int(belief.argmax())]
        if target == self.state:
            return self.state
        current_belief = belief[STATES.index(self.state)]
        refuted = current_belief < self.REFUTE
        if belief.max() < self.COMMIT and not refuted:
            return self.state
        self.state = self.NEXT_HOP[(self.state, target)]
        return self.state


STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40,  40,  40), "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80,  60,   0), "fg": (255, 200,  50), "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80,  10,  10), "fg": (255,  80,  80), "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10,  60,  10), "fg": (80,  230,  80), "label": "EXITING"},
}
BAR_COLORS = [(180, 180, 180), (255, 200, 50), (255, 80, 80), (80, 230, 80)]
PANEL_W = 680

try:
    font_status = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 30)
    font_sign   = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 19)
    font_desc   = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 17)
    font_bar    = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 14)
    font_lat    = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 15)
except Exception:
    font_status = font_sign = font_desc = font_bar = font_lat = ImageFont.load_default()

with open(f"{ENG}/output_negcontrol_video_batch.json") as f:
    out = json.load(f)
with open(f"{ENG}/input_negcontrol_video_batch_meta.json") as f:
    meta = json.load(f)
assert len(out["responses"]) == len(meta)

LAT_PREFILL_MS = 27.14
LAT_VISION_MS = 21.68
LAT_PER_TOKEN_MS = 2.73
LAT_TOK = {"EGO": 10, "TRAFFIC": 12, "ACTIVE": 10, "SIGN": 25, "DESC": 80}


def est_latency_ms(question_names):
    total = LAT_VISION_MS
    for q in question_names:
        total += LAT_PREFILL_MS + LAT_TOK[q] * LAT_PER_TOKEN_MS
    return total


by_frame = {}
for resp, m in zip(out["responses"], meta):
    by_frame.setdefault(m["frame_idx"], {})[m["question"]] = clean_gen(resp["output_text"])

os.makedirs(f"{BASE}/outputs/negcontrol", exist_ok=True)

LAT_CYCLE_MS = est_latency_ms(["EGO", "TRAFFIC", "ACTIVE", "SIGN"])
LAT_DESC_MS = LAT_VISION_MS + LAT_PREFILL_MS + LAT_TOK["DESC"] * LAT_PER_TOKEN_MS

video_in = f"{BASE}/videos_negcontrol/comma2k19_highway_night.mp4"
video_out = f"{BASE}/outputs/negcontrol/comma2k19_highway_night_ANNOTATED.mp4"
cap = cv2.VideoCapture(video_in)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PANEL_W, H))

sample_idxs = sorted(by_frame.keys())

bf = BayesFilter()
policy = DisplayPolicy()
belief = PRIOR.copy()
current_state = policy.state
current_text = "Analyzing..."
sign_read = None
sign_last_sec = -99.0
last_lat_ms = LAT_CYCLE_MS

print(f"comma2k19_highway_night: {total_fr} frames, {len(sample_idxs)} amostras")
si = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    sec = frame_idx / fps

    if si < len(sample_idxs) and frame_idx >= sample_idxs[si]:
        ans = by_frame[sample_idxs[si]]
        ego_txt = ans.get("EGO", "")
        ego_i = classify_answer(ego_txt, EGO_SHORT)
        p_ego = peaked_dist(ego_i, 3) if ego_i is not None else np.ones(3) / 3

        traffic_txt = ans.get("TRAFFIC", "")
        tl = traffic_txt.lower()
        tr_i = None
        if "partially blocked" in tl or "partial" in tl: tr_i = 0
        elif "shift" in tl: tr_i = 1
        elif "fully blocked" in tl or "fully" in tl: tr_i = 2
        elif "clear" in tl or "no lane" in tl: tr_i = 3
        p_traffic = peaked_dist(tr_i, 4) if tr_i is not None else np.ones(4) / 4

        active_txt = ans.get("ACTIVE", "")
        al = active_txt.lower()
        ac_i = 0 if "active" in al else (1 if "passive" in al else None)
        p_active = peaked_dist(ac_i, 2) if ac_i is not None else np.ones(2) / 2

        sign_txt = ans.get("SIGN", "")
        p_sign, sr = parse_sign(sign_txt)
        if sr is not None:
            sign_read, sign_last_sec = sr, sec

        emission = ((EGO_EMIT @ p_ego) * (TRAFFIC_EMIT @ p_traffic)
                    * (ACTIVE_EMIT @ p_active) * (SIGN_EMIT @ p_sign))
        emission = emission / emission.sum()
        belief = bf.update(emission)
        prev_state = current_state
        current_state = policy.update(belief)
        if current_state != prev_state:
            print(f"  t={sec:5.1f}s [TRANSITION] {prev_state.value.upper()} -> {current_state.value.upper()}")

        if "DESC" in ans:
            current_text = strip_city_clause(ans["DESC"])
            last_lat_ms = LAT_CYCLE_MS + LAT_DESC_MS
        else:
            last_lat_ms = LAT_CYCLE_MS

        si += 1

    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw = ImageDraw.Draw(panel)

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 0, PANEL_W, 58], fill=style["bg"])
    draw.text((14, 12), style["label"], font=font_status, fill=style["fg"])
    draw.text((PANEL_W - 130, 20), f"P={belief.max():.2f}", font=font_sign, fill=style["fg"])

    y = 62
    if sign_read is not None and (sec - sign_last_sec) < 4.0:
        draw.rectangle([0, y, PANEL_W, y + 32], fill=(70, 55, 0))
        draw.text((14, y + 6), f"SIGN: {sign_read[:38]}", font=font_sign, fill=(255, 220, 60))
        y += 38
    else:
        y += 6

    bar_max = PANEL_W - 200
    for i, (name, p) in enumerate(zip(STATE_NAMES, belief)):
        draw.text((14, y), f"{name:<11s}", font=font_bar, fill=(200, 200, 200))
        draw.rectangle([125, y + 2, 125 + int(bar_max * p), y + 13], fill=BAR_COLORS[i])
        draw.text((132 + int(bar_max * p), y), f"{p:.2f}", font=font_bar, fill=(150, 150, 150))
        y += 19
    y += 10

    draw.text((14, y), "WHAT THE MODEL SEES:", font=font_bar, fill=(255, 200, 80))
    y += 20
    for line in textwrap.wrap(current_text, width=42)[:8]:
        draw.text((14, y), line, font=font_desc, fill=(235, 235, 210))
        y += 22
        if y > H - 70:
            break

    draw.rectangle([0, H - 44, PANEL_W, H - 22], fill=(20, 30, 20))
    draw.text((14, H - 40), f"TensorRT INT4 (A100): ~{last_lat_ms:.0f} ms/ciclo -- CONTROLE NEGATIVO (sem obra)",
              font=font_lat, fill=(230, 120, 120))

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 22, PANEL_W, H], fill=(20, 20, 20))
    draw.rectangle([0, H - 22, int(PANEL_W * progress), H], fill=(60, 60, 60))
    draw.text((6, H - 20), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_bar, fill=(210, 210, 210))

    combined = np.concatenate([frame, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)], axis=1)
    writer.write(combined)
    frame_idx += 1

cap.release()
writer.release()
print(f"salvo -> {video_out}")
