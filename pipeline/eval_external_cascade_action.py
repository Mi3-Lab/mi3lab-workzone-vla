#!/usr/bin/env python3
"""The world model's own cascade on the California footage, scored by ACTION.

eval_external_cosmos3.py scores the cascade by raw evidence (a gate activation
or a sign keyword anywhere in the window).  The fitted estimators are scored by
whether their state leaves OUTSIDE.  Those are different bars, and comparing a
system on one against a system on the other is not the paired comparison the
out-of-domain table claims.  This runs the deployed C3E cascade -- the real
CascadeStateMachine, with the thresholds calibrated for C3E -- on the same 6 s
windows and 1 s samples, and records both outcomes.

Run with the workzone venv.
"""
import os
import time

import cv2

import eval_external_cosmos3 as E
from cascade_state_machine import (CascadeStateMachine, WZState,
                                   has_corroboration, parse_gate)
from evaluate_c3e_spine import C3E_CSM
from live_cascade_demo import DESC_PROMPT, GATE_PROMPT, SIGN_PROMPT, PersistentEngine

SHM = "/dev/shm/cascade_ood.jpg"
WIN_S, STEP_S = 6.0, 1.0


def window(engine, cap, fps, t0):
    csm = CascadeStateMachine(**C3E_CSM)
    acted = evid = False
    t = 0.0
    while t <= WIN_S:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + t) * fps))
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(SHM, cv2.resize(frame, (E.W, E.H), interpolation=cv2.INTER_AREA))
        g = bool(parse_gate(engine.infer(SHM, GATE_PROMPT, 8)[0] or ""))
        s = has_corroboration("", (engine.infer(SHM, SIGN_PROMPT, 15)[0] or "").strip())
        c = has_corroboration(engine.infer(SHM, DESC_PROMPT, 40)[0] or "", "")
        evid = evid or g or s
        # identical evidence mapping to calibrate_cascade.replay, which is how
        # these thresholds were fitted
        st = csm.update_evidence(g or s, c or s, None, fast_entry=s or (g and c))
        acted = acted or st != WZState.OUTSIDE
        t += STEP_S
    return acted, evid


def main():
    engine = PersistentEngine(
        os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video"),
        os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"),
        E.ENG, E.VIS, temperature=0.0)
    by, tot, act_tot, ev_tot = {}, 0, 0, 0
    for key, tss in E.parse_txt():
        path = E.find_video(key)
        if not path:
            continue
        cond = E.condition(key)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t0 in tss:
            t_start = time.monotonic()
            a, e = window(engine, cap, fps, t0)
            c = by.setdefault(cond, [0, 0, 0])
            c[0] += a; c[1] += 1; c[2] += e
            tot += 1; act_tot += a; ev_tot += e
            print(f"{key} @{t0//60}:{t0%60:02d} [{cond}] -> action={'Y' if a else 'n'} "
                  f"evidence={'Y' if e else 'n'} ({time.monotonic()-t_start:.1f}s)", flush=True)
        cap.release()
    engine.close()
    print("\n=== C3E cascade, out-of-domain: evidence recovered vs. acted upon ===")
    for cond, (a, n, e) in sorted(by.items()):
        print(f"  {cond:12s} action {a}/{n} = {a/n:4.0%}   evidence {e}/{n} = {e/n:4.0%}")
    print(f"  {'OVERALL':12s} action {act_tot}/{tot} = {act_tot/tot:4.0%}   "
          f"evidence {ev_tot}/{tot} = {ev_tot/tot:4.0%}")


if __name__ == "__main__":
    main()
