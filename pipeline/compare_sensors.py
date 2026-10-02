#!/usr/bin/env python3
"""Response bias and discrimination of every evidence source on California,
side by side, on the seconds all of them have.

Sources (whichever records exist):
  detector        eval_cache/ood_stream, column det (score > 0 = responds)
  C3E gate        eval_cache/ood_stream, column gate
  Qwen-Drive gate eval_cache/qwendrive_ood_gate (or _stream), column gate

Reports, per source:
  yes-rate on seconds with no work in view, on visible-but-irrelevant work, on
  ego-relevant work; AUC for perception (visible vs none) and for ego-relevance
  (relevant vs visible-only), overall and per condition; and the window-hit
  proxy at the 39 annotated signs next to its rate on work-free 7 s windows.

Labels: annotation/california_labels.json if present, else the draft (flagged).
"""
import json
import os

import numpy as np

H = os.path.expanduser
ANN = H("~/jetson-deploy/annotation")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def load_qd():
    for d in ("qwendrive_ood_stream", "qwendrive_ood_gate"):
        if os.path.isdir(H(f"~/eval_cache/{d}")):
            return d
    return None


def main():
    lp = os.path.join(ANN, "california_labels.json")
    draft = not os.path.exists(lp)
    L = json.load(open(os.path.join(ANN, "california_draft.json") if draft else lp))
    qd = load_qd()
    print(f"labels: {'DRAFT (provisional)' if draft else 'verified'}   Qwen-Drive record: {qd or 'none yet'}\n")
    rows, sign_hits = [], {}
    for key, rec in L.items():
        if key.startswith("_"):
            continue
        z = np.load(H(f"~/eval_cache/ood_stream/{key}.npz"), allow_pickle=True)
        o = z["obs"]; fps = float(z["fps"]); t = o[:, 0].astype(int)
        q = None
        if qd and os.path.exists(H(f"~/eval_cache/{qd}/{key}.npz")):
            qo = np.load(H(f"~/eval_cache/{qd}/{key}.npz"), allow_pickle=True)["obs"]
            qmap = {int(r[0]): r[1] for r in qo}
            q = np.array([qmap.get(int(x), np.nan) for x in t])
        fr = (t * fps).astype(int)
        inside = lambda iv: np.array([any(a <= f <= b for a, b in iv) for f in fr])  # noqa: E731
        vis, act = inside(rec["visible"]), ~inside(rec["outside"])
        cond = str(z["condition"])
        for i in range(len(t)):
            rows.append(dict(key=key, t=t[i], cond=cond, vis=vis[i], act=act[i],
                             det=o[i, 1], c3e=o[i, 2] > .5,
                             qd=(np.nan if q is None else q[i])))
        sign_hits[key] = (t, np.array(z["approaches"], int))

    have_qd = [r for r in rows if not np.isnan(r["qd"])]
    sources = [("detector", "det"), ("C3E gate", "c3e")] + ([("Qwen-Drive gate", "qd")] if have_qd else [])
    base = have_qd if have_qd else rows
    print(f"seconds compared: {len(base)} (only seconds every listed source has)\n")
    get = lambda rs, k: np.array([float(r[k]) for r in rs])  # noqa: E731
    groups = [("no work visible", lambda r: not r["vis"] and not r["act"]),
              ("visible, not ego-relevant", lambda r: r["vis"] and not r["act"]),
              ("ego-relevant", lambda r: r["act"])]
    print(f"{'yes-rate':28s}" + "".join(f"{n:>18s}" for n, _ in sources))
    for gname, f in groups:
        sub = [r for r in base if f(r)]
        print(f"{gname + f' ({len(sub)} s)':28s}" + "".join(f"{(get(sub, k) > 0.5 if k != 'det' else get(sub, k) > 0).mean():18.1%}" for _, k in sources))

    print(f"\n{'AUC (0.5 = chance)':28s}" + "".join(f"{n:>18s}" for n, _ in sources))
    # same definitions as the paper's AUC table: visible work vs no work in view;
    # the detector contributes its continuous score, the VLMs a yes/no
    tasks = [("perception, all", None, lambda r: r["vis"], lambda r: not r["vis"] and not r["act"])]
    for c in ("day", "sunset", "night", "night-rain", "night-fog"):
        tasks.append((f"  {c}", c, lambda r: r["vis"], lambda r: not r["vis"] and not r["act"]))
    tasks.append(("ego-relevance", None, lambda r: r["act"], lambda r: r["vis"] and not r["act"]))
    for name, c, pf, nf in tasks:
        sub = [r for r in base if c is None or r["cond"] == c]
        pos = [r for r in sub if pf(r)]; neg = [r for r in sub if nf(r)]
        print(f"{name + f' ({len(pos)}/{len(neg)})':28s}" +
              "".join(f"{auc(get(pos, k), get(neg, k)):18.2f}" for _, k in sources))

    # window-hit proxy at signs vs on work-free windows (gate/response only, all sources alike)
    print(f"\n{'window hit (7 s, 1 Hz samples)':28s}" + "".join(f"{n:>18s}" for n, _ in sources))
    idx = {(r["key"], r["t"]): r for r in base}
    at_sign = {k: [] for _, k in sources}; clean = {k: [] for _, k in sources}
    for key, (t, signs) in sign_hits.items():
        for s0 in signs:
            w = [idx.get((key, x)) for x in range(s0, s0 + 7)]
            w = [r for r in w if r]
            if w:
                for _, k in sources:
                    at_sign[k].append(any(r[k] > (0 if k == 'det' else 0.5) for r in w))
        ts = sorted(x for (kk, x) in idx if kk == key)
        for a in ts:
            w = [idx.get((key, x)) for x in range(a, a + 7)]
            if all(w) and all(not r["vis"] and not r["act"] for r in w):
                for _, k in sources:
                    clean[k].append(any(r[k] > (0 if k == 'det' else 0.5) for r in w))
    print(f"{'at 39 annotated signs':28s}" + "".join(f"{np.mean(at_sign[k]):18.0%}" for _, k in sources))
    print(f"{'on work-free windows':28s}" + "".join(f"{np.mean(clean[k]):18.0%}" for _, k in sources))


if __name__ == "__main__":
    main()
