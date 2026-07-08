"""Taxa de falso-positivo do modelo em imagens/frames SEM zona de obra
(controle negativo, nunca vistos em nenhum estagio de treino).

BDD100k: controle limpo (dataset sem curadoria de obra) -- metrica principal.
ROADWork (unused): mesmo dominio de camera/cidade, sem ground-truth
confirmado -- reportado a parte, como sinal exploratorio.
"""
import json
import re

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"

WZ_KEYWORDS = re.compile(
    r"\bcone|\bbarrier|\bbarricade|TTC sign|work zone|construction|"
    r"\bworker|\bflagger|road work|lane closed|detour|excavat", re.I)

out = json.load(open(f"{ENG}/output_negcontrol_batch.json"))
meta = json.load(open(f"{ENG}/input_negcontrol_batch_meta.json"))
assert len(out["responses"]) == len(meta)

by_source = {"bdd100k": [], "roadwork_unused": []}
for resp, m in zip(out["responses"], meta):
    by_source[m["source"]].append((m, resp["output_text"]))


def analyze(source, items):
    by_frame = {}
    for m, text in items:
        by_frame.setdefault((m["clip_id"], m["frame_idx"]), {})[m["question"]] = text

    n = len(by_frame)
    ego_dist = {"OUTSIDE": 0, "APPROACHING": 0, "INSIDE": 0, "OTHER": 0}
    active_workers = 0
    sign_wz_text = 0
    desc_wz_mention = 0
    flagged_frames = []

    for key, qa in by_frame.items():
        ego = qa.get("EGO", "")
        if ego.upper().startswith("OUTSIDE"):
            ego_dist["OUTSIDE"] += 1
        elif ego.upper().startswith("APPROACHING"):
            ego_dist["APPROACHING"] += 1
        elif ego.upper().startswith("INSIDE"):
            ego_dist["INSIDE"] += 1
        else:
            ego_dist["OTHER"] += 1

        active = qa.get("ACTIVE", "")
        if re.search(r"active.{0,15}(work|worker)", active, re.I) and "passive" not in active.lower():
            active_workers += 1

        sign = qa.get("SIGN", "")
        if WZ_KEYWORDS.search(sign):
            sign_wz_text += 1

        desc = qa.get("DESC", "")
        if WZ_KEYWORDS.search(desc):
            desc_wz_mention += 1

        is_false_positive = (not ego.upper().startswith("OUTSIDE")) or WZ_KEYWORDS.search(desc)
        if is_false_positive:
            flagged_frames.append((key, ego, active, sign, desc))

    print(f"\n=== {source} === (n={n} frames)")
    print(f"  EGO distribuicao: OUTSIDE={ego_dist['OUTSIDE']} ({100*ego_dist['OUTSIDE']/n:.0f}%) "
          f"APPROACHING={ego_dist['APPROACHING']} ({100*ego_dist['APPROACHING']/n:.0f}%) "
          f"INSIDE={ego_dist['INSIDE']} ({100*ego_dist['INSIDE']/n:.0f}%) "
          f"OTHER={ego_dist['OTHER']}")
    print(f"  ACTIVE diz 'active workers'  : {active_workers}/{n} ({100*active_workers/n:.0f}%)")
    print(f"  SIGN cita termo de obra      : {sign_wz_text}/{n} ({100*sign_wz_text/n:.0f}%)")
    print(f"  DESC menciona elemento de obra: {desc_wz_mention}/{n} ({100*desc_wz_mention/n:.0f}%)")
    print(f"  Frames flagged (EGO!=OUTSIDE ou DESC com termo de obra): {len(flagged_frames)}/{n} "
          f"({100*len(flagged_frames)/n:.0f}%)")
    print(f"\n  Primeiros 8 frames flagged (pra spot-check visual):")
    for key, ego, active, sign, desc in flagged_frames[:8]:
        print(f"    {key}: EGO={ego!r} | DESC={desc[:100]!r}")
    return flagged_frames


flagged_bdd = analyze("bdd100k", by_source["bdd100k"])
flagged_rw = analyze("roadwork_unused", by_source["roadwork_unused"])

print("\n=== paths dos frames flagged (BDD100k) p/ visualizacao ===")
for (clip_id, _), *_ in flagged_bdd[:10]:
    print(f"  {BASE}/data/roadwork/bdd100k_unseen/images/{clip_id}.jpg")

print("\n=== paths dos frames flagged (ROADWork unused) p/ visualizacao ===")
for (clip_id, frame_idx), *_ in flagged_rw[:10]:
    import glob
    matches = glob.glob(f"{ENG}/negcontrol_frames/{clip_id}_*.jpg")
    for m in matches:
        print(f"  {m}")
