#!/usr/bin/env python3
"""Three-way out-of-domain comparison on the self-recorded California footage.

All three systems are scored on the SAME 39 annotated work-zone sign approaches
(one timestamp per sign, 6 s window sampled at 1 s), so the numbers are paired:
  eval_external_ca.py       our fine-tuned 2B
  eval_external_yolo.py     the ported detector+CLIP
  eval_external_cosmos3.py  Cosmos3-Edge INT4, zero-shot

The footage is out of domain in every sense that matters: a different continent
of the country from ROADWork's two cities, a different camera, and conditions
the benchmark does not contain at all (night, rain, fog, sunset).

This runs each evaluator as a subprocess (they each own the GPU for the length
of their run -- running them concurrently would contend for it) and parses the
per-condition breakdown each prints, then emits one table.

Usage:
  python3 ood_california_table.py --run          # run all three, then tabulate
  python3 ood_california_table.py                # tabulate from saved logs
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.expanduser("~/eval_cache/ood_california")
VENV = os.path.expanduser("~/workzone/venv/bin/python")

SYSTEMS = [
    ("our 2B (fine-tuned)", "eval_external_ca.py", sys.executable),
    ("detector+CLIP", "eval_external_yolo.py", VENV),
    ("Cosmos3-Edge INT4 (zero-shot)", "eval_external_cosmos3.py", sys.executable),
]
CONDITIONS = ["day", "evening", "sunset", "night", "night-rain", "night-fog"]


def run_all():
    os.makedirs(LOGDIR, exist_ok=True)
    for name, script, interp in SYSTEMS:
        log = os.path.join(LOGDIR, script.replace(".py", ".log"))
        print(f"=== {name} -> {log}", flush=True)
        with open(log, "w") as fh:
            subprocess.run([interp, "-u", os.path.join(HERE, script)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)


def parse(log):
    """Reads the '  <cond>  d/n = p%' block every evaluator prints at the end."""
    if not os.path.exists(log):
        return None
    got = {}
    for line in open(log):
        m = re.match(r"\s+([A-Za-z\-]+)\s+(\d+)/(\d+)\s*=", line)
        if m:
            got[m.group(1).lower()] = (int(m.group(2)), int(m.group(3)))
    return got or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the three evaluators first")
    args = ap.parse_args()
    if args.run:
        run_all()

    rows = []
    for name, script, _ in SYSTEMS:
        got = parse(os.path.join(LOGDIR, script.replace(".py", ".log")))
        if got:
            rows.append((name, got))
    if not rows:
        raise SystemExit(f"nenhum log em {LOGDIR}; rode com --run")

    cols = [c for c in CONDITIONS if any(c in g for _, g in rows)]
    hdr = f"{'system':32s} {'overall':>10s} " + " ".join(f"{c:>11s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, got in rows:
        o = got.get("overall")
        cells = []
        for c in cols:
            if c in got:
                d, n = got[c]
                cells.append(f"{d}/{n} {d/n:5.0%}")
            else:
                cells.append(f"{'--':>11s}")
        ov = f"{o[0]}/{o[1]} {o[0]/o[1]:4.0%}" if o else "--"
        print(f"{name:32s} {ov:>10s} " + " ".join(f"{c:>11s}" for c in cells))
    print("\n(mesmos 39 sinais anotados para os tres sistemas; janela de 6 s,")
    print(" passo 1 s; deteccao = GATE afirmativo OU transcricao de placa com")
    print(" palavra-chave de obra dentro da janela)")


if __name__ == "__main__":
    main()
