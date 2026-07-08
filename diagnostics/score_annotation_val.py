"""Valida o pipeline contra a anotacao humana (workzone_annotations_full.json).

27 snippets ROADWork limpos (nunca em treino), 30s cada, ~51 amostras/snippet.
Compara 3 designs de fusao usando AS MESMAS respostas do modelo stage7_1 INT4:
  A. cascade (gate + corroboracao + sign latch + histerese)   <- atual
  B. bayes-v16 (filtro Bayesiano puro, design antigo)
  C. ego-cru (sem fusao nenhuma, so a resposta EGO por frame) <- baseline

Metricas:
  - acuracia por amostra (4 estados) e binaria (OUTSIDE vs ativo)
  - matriz de confusao
  - erro de timing das transicoes OUTSIDE->ativo (mediana de |dt|)
"""
import json, sys, os, collections
import numpy as np

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla/inference")
from cascade_state_machine import (CascadeStateMachine, WZState, parse_gate,
                                   classify_ego, peaked_dist, EGO_EMIT, TRANS,
                                   PRIOR, STATES)

ENG = f"{BASE}/models/workzone-2b-stage7-1-engines"
ANN = json.load(open("/home/wesleyferreiramaia/data/wokzone-alpamayo/workzone_annotations_full.json"))

out = json.load(open(f"{ENG}/output_annval_batch.json"))
meta = json.load(open(f"{ENG}/input_annval_batch_meta.json"))
assert len(out["responses"]) == len(meta)

by_clip = {}
for r, m in zip(out["responses"], meta):
    by_clip.setdefault(m["clip"], {}).setdefault(m["frame_idx"], {})[m["question"]] = \
        r["output_text"].replace(".DATA", "").strip()


def human_state(ann, frame_idx):
    for state, ranges in ann.items():
        for a, b in ranges:
            if a <= frame_idx <= b:
                return state
    return None


# ---------- design B: filtro Bayesiano v16 (EGO+TRAFFIC+ACTIVE+SIGN) ----------
TRAFFIC_EMIT = np.array([[0.10,0.03,0.02,0.85],[0.50,0.14,0.13,0.23],
                         [0.45,0.14,0.11,0.30],[0.30,0.06,0.05,0.59]])
ACTIVE_EMIT = np.array([[0.10,0.90],[0.45,0.55],[0.50,0.50],[0.25,0.75]])
SIGN_EMIT = np.array([[0.05,0.95],[0.45,0.55],[0.30,0.70],[0.10,0.90]])
SIGN_KW = ["ROAD WORK","WORK ZONE","DETOUR","LANE CLOSED","ROAD CLOSED",
           "SIDEWALK CLOSED","UTILITY WORK","SHOULDER WORK","FLAGGER",
           "BE PREPARED TO STOP","LANE ENDS","LANE SHIFT"]


class BayesV16:
    COMMIT, REFUTE = 0.60, 0.15
    NEXT = {("outside","approaching"):"approaching",("outside","inside"):"approaching",
            ("outside","exiting"):"approaching",("approaching","outside"):"outside",
            ("approaching","inside"):"inside",("approaching","exiting"):"inside",
            ("inside","exiting"):"exiting",("inside","outside"):"exiting",
            ("inside","approaching"):"exiting",("exiting","outside"):"outside",
            ("exiting","approaching"):"outside",("exiting","inside"):"inside"}

    def __init__(self):
        self.belief = PRIOR.copy()
        self.state = "outside"

    def update(self, qa):
        ego_i = classify_ego(qa.get("EGO",""))
        p_ego = peaked_dist(ego_i,3) if ego_i is not None else np.ones(3)/3
        tl = qa.get("TRAFFIC","").lower()
        tr = 0 if ("partial" in tl) else 1 if "shift" in tl else 2 if "fully" in tl else 3 if ("clear" in tl or "no lane" in tl) else None
        p_tr = peaked_dist(tr,4) if tr is not None else np.ones(4)/4
        al = qa.get("ACTIVE","").lower()
        ac = 0 if "active" in al else 1 if "passive" in al else None
        p_ac = peaked_dist(ac,2) if ac is not None else np.ones(2)/2
        su = qa.get("SIGN","").upper()
        p_sg = peaked_dist(0,2) if any(k in su for k in SIGN_KW) else np.ones(2)/2
        em = (EGO_EMIT@p_ego)*(TRAFFIC_EMIT@p_tr)*(ACTIVE_EMIT@p_ac)*(SIGN_EMIT@p_sg)
        em = em/em.sum()
        b = (self.belief@TRANS)*em
        self.belief = b/b.sum()
        names = ["outside","approaching","inside","exiting"]
        target = names[int(self.belief.argmax())]
        if target != self.state:
            cur = self.belief[names.index(self.state)]
            if self.belief.max() >= self.COMMIT or cur < self.REFUTE:
                self.state = self.NEXT[(self.state, target)]
        return self.state


def ego_raw_state(qa):
    e = qa.get("EGO","").upper()
    for s in ("OUTSIDE","APPROACHING","INSIDE"):
        if e.startswith(s):
            return s.lower()
    return "outside"


def run_design(name, clip, samples):
    """Retorna lista de (frame_idx, estado_predito)."""
    preds = []
    if name == "cascade":
        sm = CascadeStateMachine()
        for fi in samples:
            qa = by_clip[clip][fi]
            g = parse_gate(qa.get("GATE",""))
            st = sm.update(g, qa.get("EGO",""), qa.get("DESC",""), qa.get("SIGN",""))
            preds.append((fi, st.value))
    elif name == "bayes_v16":
        bf = BayesV16()
        for fi in samples:
            preds.append((fi, bf.update(by_clip[clip][fi])))
    else:  # ego cru
        for fi in samples:
            preds.append((fi, ego_raw_state(by_clip[clip][fi])))
    return preds


DESIGNS = ["cascade", "bayes_v16", "ego_cru"]
acc = {d: [0,0] for d in DESIGNS}
acc_bin = {d: [0,0] for d in DESIGNS}
conf = {d: collections.Counter() for d in DESIGNS}
trans_err = {d: [] for d in DESIGNS}

for clip, frames in sorted(by_clip.items()):
    ann = ANN[clip]
    samples = sorted(frames)
    fps = 30.0
    # primeira transicao humana OUTSIDE -> ativo (se existir)
    h_first_active = None
    for state in ("approaching","inside"):
        for a,b in ann.get(state,[]):
            if h_first_active is None or a < h_first_active:
                h_first_active = a
    for d in DESIGNS:
        preds = run_design(d, clip, samples)
        p_first_active = None
        for fi, p in preds:
            h = human_state(ann, fi)
            if h is None:
                continue
            acc[d][1] += 1
            acc[d][0] += int(p == h)
            hb = "outside" if h == "outside" else "active"
            pb = "outside" if p == "outside" else "active"
            acc_bin[d][1] += 1
            acc_bin[d][0] += int(pb == hb)
            conf[d][(h,p)] += 1
            if p != "outside" and p_first_active is None:
                p_first_active = fi
        if h_first_active is not None:
            if p_first_active is not None:
                trans_err[d].append((p_first_active - h_first_active)/fps)
            else:
                trans_err[d].append(float("inf"))  # nunca detectou

print(f"{len(by_clip)} snippets | {acc['cascade'][1]} amostras com ground truth\n")
print(f"{'design':<12} {'acc 4-estados':>14} {'acc binaria':>12} {'timing 1a deteccao (mediana dt, s)':>36} {'zonas perdidas':>15}")
for d in DESIGNS:
    a = 100*acc[d][0]/max(acc[d][1],1)
    ab = 100*acc_bin[d][0]/max(acc_bin[d][1],1)
    errs = trans_err[d]
    finite = [e for e in errs if e != float("inf")]
    missed = sum(1 for e in errs if e == float("inf"))
    med = np.median(np.abs(finite)) if finite else float("nan")
    late = np.median(finite) if finite else float("nan")
    print(f"{d:<12} {a:>13.1f}% {ab:>11.1f}% {med:>18.1f} (vies {late:+.1f}) {missed:>10}/{len(errs)}")

print("\nmatriz de confusao (cascade) — linhas=humano, colunas=predito:")
names = ["outside","approaching","inside","exiting"]
print(f"{'':>12}" + "".join(f"{n:>12}" for n in names))
for h in names:
    row = "".join(f"{conf['cascade'][(h,p)]:>12}" for p in names)
    print(f"{h:>12}{row}")
