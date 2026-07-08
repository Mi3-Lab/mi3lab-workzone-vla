"""Aplica o MESMO filtro Bayesiano/DisplayPolicy do render_trt_video.py nas
respostas do video de controle negativo (comma2k19, rodovia SEM obra,
confirmado visualmente limpo) -- pra responder: no pipeline real, com
suavizacao temporal ligada, o estado EGO fica OUTSIDE o tempo todo?
"""
import json
import re
import numpy as np
from enum import Enum

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"


class WZState(Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    INSIDE = "inside"
    EXITING = "exiting"


STATES = [WZState.OUTSIDE, WZState.APPROACHING, WZState.INSIDE, WZState.EXITING]
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
        return peaked_dist(0, 2), sign_txt
    return np.ones(2) / 2, None


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


out = json.load(open(f"{ENG}/output_negcontrol_video_batch.json"))
meta = json.load(open(f"{ENG}/input_negcontrol_video_batch_meta.json"))
assert len(out["responses"]) == len(meta)

by_frame = {}
for resp, m in zip(out["responses"], meta):
    by_frame.setdefault(m["frame_idx"], {"sec": m["sec"]})[m["question"]] = clean_gen(resp["output_text"])

frame_idxs = sorted(by_frame.keys())
print(f"Video: comma2k19_highway_night.mp4 (rodovia noturna, SEM obra, confirmado visualmente)")
print(f"{len(frame_idxs)} amostras ao longo de {by_frame[frame_idxs[-1]]['sec']:.1f}s\n")

bf = BayesFilter()
policy = DisplayPolicy()
state_time = {s: 0.0 for s in STATES}
transitions = []
raw_ego_dist = {"OUTSIDE": 0, "APPROACHING": 0, "INSIDE": 0, "OTHER": 0}
prev_sec = 0.0

WZ_OBJECT_WORDS = re.compile(
    r"\bcone|\bbarrier|\bbarricade|TTC sign|\bworker|\bflagger|excavat|"
    r"\bdrum\b|tubular marker|work vehicle|construction (?:crew|equipment|site)",
    re.I)

audit_rows = []  # registro completo, frame a frame, pra auditoria manual

for fi in frame_idxs:
    ans = by_frame[fi]
    sec = ans["sec"]
    dt = sec - prev_sec
    prev_sec = sec

    ego_txt = ans.get("EGO", "")
    ego_i = classify_answer(ego_txt, EGO_SHORT)
    if ego_i is None:
        raw_ego_dist["OTHER"] += 1
    else:
        raw_ego_dist[EGO_SHORT[ego_i]] += 1
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

    emission = ((EGO_EMIT @ p_ego) * (TRAFFIC_EMIT @ p_traffic)
                * (ACTIVE_EMIT @ p_active) * (SIGN_EMIT @ p_sign))
    emission = emission / emission.sum()
    belief = bf.update(emission)
    prev_state = policy.state
    current_state = policy.update(belief)
    state_time[current_state] += dt
    if current_state != prev_state:
        transitions.append((sec, prev_state.value.upper(), current_state.value.upper(), ego_txt[:80]))

    desc_txt = ans.get("DESC", "")
    hallucinated_object = bool(WZ_OBJECT_WORDS.search(desc_txt))
    audit_rows.append({
        "sec": sec,
        "ego_raw": ego_txt,
        "traffic_raw": traffic_txt,
        "active_raw": active_txt,
        "sign_raw": sign_txt,
        "desc_raw": desc_txt,
        "desc_hallucinates_object": hallucinated_object,
        "belief": {s.value: round(float(belief[i]), 3) for i, s in enumerate(STATES)},
        "displayed_state": current_state.value.upper(),
    })

total_t = by_frame[frame_idxs[-1]]["sec"]
print("=== Distribuicao RAW por-frame (antes do filtro Bayesiano) ===")
n = len(frame_idxs)
for k, v in raw_ego_dist.items():
    print(f"  {k}: {v}/{n} ({100*v/n:.0f}%)")

print(f"\n=== Estado FINAL exibido (com filtro Bayesiano/DisplayPolicy, {n} amostras / {total_t:.1f}s) ===")
for s in STATES:
    pct = 100 * state_time[s] / total_t if total_t else 0
    print(f"  {s.value.upper():12s}: {state_time[s]:5.1f}s ({pct:.0f}%)")

print(f"\n=== Transicoes de estado ===")
if not transitions:
    print("  NENHUMA -- estado ficou OUTSIDE o video inteiro.")
else:
    for sec, frm, to, ego_txt in transitions:
        print(f"  t={sec:5.1f}s  {frm} -> {to}   (EGO cru: {ego_txt!r})")

# ---------------------------------------------------------------------------
# Registro completo frame-a-frame, pra separar 3 hipoteses:
#   (1) overfitting/vies de classificacao -- EGO diz APPROACHING/INSIDE sem a
#       DESC citar nenhum objeto especifico de obra (viés no cabecalho de
#       classificacao, nao alucinacao de percepcao)
#   (2) alucinacao de percepcao -- DESC inventa objetos especificos (cone,
#       barreira, trabalhador) que nao existem na cena
#   (3) filtro Bayesiano mal calibrado -- o estado EXIBIDO diverge do que o
#       sinal cru por-frame (raw_ego_dist) sozinho sugeriria
# ---------------------------------------------------------------------------
audit_path = f"{BASE}/mi3lab-workzone-vla/diagnostics/negcontrol_video_audit.json"
with open(audit_path, "w") as f:
    json.dump(audit_rows, f, indent=1)

n_approaching_or_inside_raw = sum(
    1 for r in audit_rows if r["ego_raw"].upper().startswith(("APPROACHING", "INSIDE")))
n_with_object_hallucination = sum(1 for r in audit_rows if r["desc_hallucinates_object"])
n_approaching_raw_no_object = sum(
    1 for r in audit_rows
    if r["ego_raw"].upper().startswith(("APPROACHING", "INSIDE")) and not r["desc_hallucinates_object"])
n_displayed_wz_but_raw_outside = sum(
    1 for r in audit_rows
    if r["displayed_state"] != "OUTSIDE" and r["ego_raw"].upper().startswith("OUTSIDE"))

print(f"\n=== Diagnostico: overfitting vs alucinacao de percepcao vs filtro Bayesiano ===")
print(f"Registro completo (todas as {len(audit_rows)} amostras, texto cru de EGO/TRAFFIC/ACTIVE/SIGN/DESC "
      f"+ belief do filtro + estado exibido) salvo em:\n  {audit_path}\n")
print(f"  EGO cru = APPROACHING/INSIDE (nao OUTSIDE): {n_approaching_or_inside_raw}/{len(audit_rows)} "
      f"({100*n_approaching_or_inside_raw/len(audit_rows):.0f}%)")
print(f"  ...dessas, SEM nenhum objeto especifico de obra citado na DESC : "
      f"{n_approaching_raw_no_object}/{max(n_approaching_or_inside_raw,1)} "
      f"({100*n_approaching_raw_no_object/max(n_approaching_or_inside_raw,1):.0f}%) "
      f"-> aponta pra VIES DE CLASSIFICACAO (overfitting), nao alucinacao de percepcao")
print(f"  DESC cita objeto especifico de obra (cone/barreira/trabalhador/etc): "
      f"{n_with_object_hallucination}/{len(audit_rows)} ({100*n_with_object_hallucination/len(audit_rows):.0f}%) "
      f"-> se >0, e' alucinacao de percepcao genuina, nao so vies de classificacao")
print(f"  Estado EXIBIDO != OUTSIDE mas EGO cru daquele frame ERA 'OUTSIDE': "
      f"{n_displayed_wz_but_raw_outside}/{len(audit_rows)} "
      f"-> se > 0, o FILTRO BAYESIANO esta mantendo um estado que o sinal daquele "
      f"instante especifico ja nao sustenta (inercia do TRANS/COMMIT/REFUTE)")
