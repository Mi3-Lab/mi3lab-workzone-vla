"""Compara o design ANTIGO (filtro Bayes v16, entrada guiada so pelo EGO)
contra o NOVO (CascadeStateMachine: gate de existencia + corroboracao +
histerese) nos 3 benchmarks:

  A. BDD100k (249 imagens sem obra)    -> taxa de falso-positivo por frame
  B. comma2k19 (video continuo s/ obra) -> % do tempo fora de OUTSIDE
  C. 4 videos de obra (positivos)       -> transicoes/tempo em estado ativo
                                           (nao pode regredir: obra tem que
                                           continuar sendo detectada)

Reusa as respostas de estado ja computadas (batches anteriores) + as
respostas novas do gate (output_gate_batch.json).
"""
import json
import sys
import os

sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../inference")
from cascade_state_machine import (CascadeStateMachine, WZState, parse_gate,
                                   has_corroboration, classify_ego)

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"


def clean_gen(t):
    return t.replace(".DATA", "").strip()


# ---------------------------------------------------------------- carga
gate_out = json.load(open(f"{ENG}/output_gate_batch.json"))
gate_meta = json.load(open(f"{ENG}/input_gate_batch_meta.json"))
assert len(gate_out["responses"]) == len(gate_meta)

gate = {}  # (source, clip_id, frame_idx) -> True/False/None
for resp, m in zip(gate_out["responses"], gate_meta):
    gate[(m["source"], m["clip_id"], m["frame_idx"])] = parse_gate(resp["output_text"])


def load_answers(output_file, meta_file, key_fields):
    out = json.load(open(f"{ENG}/{output_file}"))
    meta = json.load(open(f"{ENG}/{meta_file}"))
    by_frame = {}
    for resp, m in zip(out["responses"], meta):
        key = tuple(m[k] for k in key_fields)
        by_frame.setdefault(key, {})[m["question"]] = clean_gen(resp["output_text"])
    return by_frame


# ================================================================ A. BDD100k
print("=" * 72)
print("A. BDD100k -- 249 imagens de direcao generica SEM obra (frames isolados)")
print("=" * 72)
bdd_answers = load_answers("output_negcontrol_batch.json",
                           "input_negcontrol_batch_meta.json",
                           ("source", "clip_id", "frame_idx"))
bdd_keys = [k for k in bdd_answers if k[0] == "bdd100k"]

old_fp = 0   # design antigo: EGO != OUTSIDE => alarme
new_fp = 0   # design novo (por-frame): gate Yes E corroboracao no mesmo frame
gate_yes = 0
for key in bdd_keys:
    qa = bdd_answers[key]
    g = gate.get(("bdd100k", key[1], 0))
    if not qa.get("EGO", "").upper().startswith("OUTSIDE"):
        old_fp += 1
    if g:
        gate_yes += 1
        if has_corroboration(qa.get("DESC", ""), qa.get("SIGN", "")):
            new_fp += 1

n = len(bdd_keys)
print(f"  Design ANTIGO (EGO cru != OUTSIDE)          : {old_fp}/{n} ({100*old_fp/n:.0f}%)")
print(f"  Gate sozinho responde Yes                    : {gate_yes}/{n} ({100*gate_yes/n:.0f}%)")
print(f"  Design NOVO (gate Yes + corroboracao)        : {new_fp}/{n} ({100*new_fp/n:.0f}%)")

# ================================================================ B. comma2k19
print()
print("=" * 72)
print("B. comma2k19 -- video continuo SEM obra (80 amostras / 47.4s)")
print("=" * 72)
nc_answers = load_answers("output_negcontrol_video_batch.json",
                          "input_negcontrol_video_batch_meta.json",
                          ("city", "frame_idx"))
nc_meta = json.load(open(f"{ENG}/input_negcontrol_video_batch_meta.json"))
sec_by_idx = {m["frame_idx"]: m["sec"] for m in nc_meta}
nc_idxs = sorted({k[1] for k in nc_answers})

sm = CascadeStateMachine()
state_time = {s: 0.0 for s in WZState}
transitions = []
prev_sec = 0.0
for fi in nc_idxs:
    qa = nc_answers[("comma2k19_highway_night", fi)]
    g = gate.get(("comma2k19", "comma2k19_highway_night", fi))
    sec = sec_by_idx[fi]
    dt = sec - prev_sec
    prev_sec = sec
    prev = sm.state
    st = sm.update(g, qa.get("EGO", ""), qa.get("DESC", ""), qa.get("SIGN", ""))
    state_time[st] += dt
    if st != prev:
        transitions.append((sec, prev.value, st.value))

total_t = sec_by_idx[nc_idxs[-1]]
outside_pct = 100 * state_time[WZState.OUTSIDE] / total_t
print(f"  Design ANTIGO (medido antes): 99% do tempo em APPROACHING, transicao em t=1.2s")
print(f"  Design NOVO: tempo em OUTSIDE = {state_time[WZState.OUTSIDE]:.1f}s/{total_t:.1f}s ({outside_pct:.0f}%)")
if transitions:
    for sec, a, b in transitions:
        print(f"    t={sec:5.1f}s  {a.upper()} -> {b.upper()}")
else:
    print(f"    nenhuma transicao -- OUTSIDE o video inteiro")

# ================================================================ C. positivos
print()
print("=" * 72)
print("C. 4 videos de OBRA (positivos) -- deteccao nao pode regredir")
print("=" * 72)
wz_answers = load_answers("output_video_batch_lmheadfix.json",
                          "input_video_batch_meta.json",
                          ("city", "frame_idx"))
wz_meta = json.load(open(f"{ENG}/input_video_batch_meta.json"))
sec_by = {}
for m in wz_meta:
    sec_by[(m["city"], m["frame_idx"])] = m["sec"]

# Referencia (design antigo, validado visualmente nos videos renderizados):
OLD_TRANSITIONS = {
    "seattle_unseen": [(1.2, "APPROACHING"), (9.6, "INSIDE")],
    "boston": [(6.0, "APPROACHING"), (6.6, "INSIDE")],
    "denver": [(0.6, "APPROACHING"), (4.2, "INSIDE")],
    "chicago": [(0.6, "APPROACHING"), (1.2, "INSIDE")],
}

for city in ["seattle_unseen", "boston", "denver", "chicago"]:
    idxs = sorted(fi for (c, fi) in wz_answers if c == city)
    sm = CascadeStateMachine()
    transitions = []
    state_time = {s: 0.0 for s in WZState}
    prev_sec = 0.0
    for fi in idxs:
        qa = wz_answers[(city, fi)]
        g = gate.get(("workzone_videos", city, fi))
        sec = sec_by[(city, fi)]
        dt = sec - prev_sec
        prev_sec = sec
        prev = sm.state
        st = sm.update(g, qa.get("EGO", ""), qa.get("DESC", ""), qa.get("SIGN", ""))
        state_time[st] += dt
        if st != prev:
            transitions.append((sec, st.value.upper()))
    total_t = sec_by[(city, idxs[-1])]
    active_t = total_t - state_time[WZState.OUTSIDE]
    old_str = ", ".join(f"t={t}s->{s}" for t, s in OLD_TRANSITIONS[city])
    new_str = ", ".join(f"t={t}s->{s}" for t, s in transitions) or "NENHUMA (REGRESSAO!)"
    print(f"  [{city}] ({total_t:.0f}s)")
    print(f"    antigo: {old_str}")
    print(f"    novo  : {new_str}")
    print(f"    tempo em estado ativo (novo): {active_t:.1f}s ({100*active_t/total_t:.0f}%)")
