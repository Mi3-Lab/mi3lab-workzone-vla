"""Compara o batch pos-fix do lm_head com o batch quebrado original.

Metricas por arquivo de saida (685 respostas):
  - %% que comeca com '.'/token-lixo (proxy do bug do 1o token)
  - %% de respostas DESC com loop de repeticao (mesma palavra 4+ vezes seguidas)
  - %% de respostas DESC com alucinacao off-domain (podcast/subscribe/internet...)
"""
import json
import re
import sys

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"

META = json.load(open(f"{ENG}/input_video_batch_meta.json"))

HALLU_PAT = re.compile(
    r"podcast|subscribe|listening|episode|internet|conversation between|"
    r"COLLECTION IN|REALTIME|individuals", re.I)


def rep_loop(text):
    words = re.findall(r"\w+", text.upper())
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    return False


def load_responses(path):
    d = json.load(open(path))
    if isinstance(d, dict) and "responses" in d:
        return [r["output_text"] for r in d["responses"]]
    out = []
    for r in d:
        out.append(r if isinstance(r, str) else r.get("output_text", r.get("text", json.dumps(r))))
    return out


def analyze(path, label):
    try:
        resp = load_responses(path)
    except FileNotFoundError:
        print(f"[{label}] arquivo nao existe: {path}")
        return
    n = len(resp)
    starts_dot = sum(1 for t in resp if t.lstrip().startswith("."))
    starts_junk = sum(1 for t in resp if t[:1] not in "" and not t.lstrip()[:1].isalnum())
    desc_idx = [i for i, m in enumerate(META) if m.get("question") == "DESC"]
    if not desc_idx:  # fallback: identifica DESC pelo tamanho da resposta
        desc_idx = [i for i, t in enumerate(resp) if len(t) > 150]
    desc = [resp[i] for i in desc_idx if i < n]
    reps = sum(1 for t in desc if rep_loop(t))
    hallu = sum(1 for t in desc if HALLU_PAT.search(t))
    print(f"[{label}] n={n}")
    print(f"  comeca com '.'          : {starts_dot}/{n} ({100*starts_dot/max(n,1):.0f}%)")
    print(f"  comeca com nao-alfanum  : {starts_junk}/{n} ({100*starts_junk/max(n,1):.0f}%)")
    print(f"  DESC n={len(desc)} | repeticao-loop: {reps} ({100*reps/max(len(desc),1):.0f}%) | "
          f"alucinacao off-domain: {hallu} ({100*hallu/max(len(desc),1):.0f}%)")
    for t in desc[:3]:
        print(f"    exemplo DESC: {t[:130]!r}")


if __name__ == "__main__":
    analyze(f"{ENG}/output_video_batch.json", "ORIGINAL (lm_head errado)")
    analyze(f"{ENG}/output_video_batch_fixed.json", "SUPRESSAO 52728 (revertida)")
    analyze(f"{ENG}/output_video_batch_lmheadfix.json", "LMHEAD FIX")
