"""Avaliacao BF16 do stage7 (hard negatives) nos 3 benchmarks.

Roda as 6 perguntas (GATE + 5 do v16) nos 466 frames:
  - 137 frames dos 4 videos de obra (positivos — nao pode regredir)
  - 80 frames do comma2k19 (video continuo sem obra)
  - 249 imagens BDD100k do benchmark (negativos por-frame)

Saida: JSONL por frame + resumo comparando com o stage6.
"""
import os, glob, json
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_DIR = f"{BASE}/models/workzone-2b-stage7-1-hf"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"
OUT_PATH = f"{BASE}/mi3lab-workzone-vla/diagnostics/stage7_1_bf16_answers.jsonl"

QUESTIONS = {
    "GATE": ("Are there any road work indicators in this scene "
             "(cones, barriers, temporary signs, workers, work vehicles)? "
             "Answer Yes or No."),
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"GATE": 16, "EGO": 32, "TRAFFIC": 16, "ACTIVE": 16, "SIGN": 25, "DESC": 80}

frames = []
for p in sorted(glob.glob(f"{ENG}/video_frames/*.jpg")):
    city = os.path.basename(p).rsplit("_", 1)[0]
    frames.append(("workzone:" + city, p))
for p in sorted(glob.glob(f"{ENG}/negcontrol_video_frames/*.jpg")):
    frames.append(("comma2k19", p))
for p in sorted(glob.glob(f"{BASE}/data/roadwork/bdd100k_unseen/images/*.jpg")):
    frames.append(("bdd100k", p))
print(f"{len(frames)} frames")

print("[LOAD] workzone-2b-stage7-1-hf (BF16)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL_DIR)

results = []
with open(OUT_PATH, "w") as fout:
    for i, (source, path) in enumerate(frames):
        image = Image.open(path).convert("RGB")
        row = {"source": source, "frame": os.path.basename(path)}
        for qname, q in QUESTIONS.items():
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": q},
            ]}]
            inputs = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=MAX_TOK[qname],
                                     do_sample=False,
                                     pad_token_id=proc.tokenizer.eos_token_id)
            n_in = inputs["input_ids"].shape[1]
            row[qname] = proc.tokenizer.decode(
                gen[0][n_in:], skip_special_tokens=True).strip()
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        results.append(row)
        if i % 40 == 0:
            print(f"[{i+1}/{len(frames)}] {source} GATE={row['GATE'][:40]!r} EGO={row['EGO'][:30]!r}")

# ------------------------------------------------------------- resumo
print("\n" + "=" * 72)
print("RESUMO STAGE 7.1 (BF16) — comparar com stage6")
print("=" * 72)


def gate_is_yes(t):
    return t.strip().lower().startswith("yes")


def ego_label(t):
    tu = t.upper()
    for lab in ("OUTSIDE", "APPROACHING", "INSIDE"):
        if lab in tu:
            return lab
    return "OTHER"


REF = {
    "workzone":  "stage6: GATE~100% | EGO quase todo APPROACHING/INSIDE (correto)",
    "comma2k19": "stage6: GATE 100% Yes (ERRADO) | EGO 100% APPROACHING (ERRADO)",
    "bdd100k":   "stage6: GATE 95% Yes (ERRADO) | EGO 34% nao-OUTSIDE (ERRADO)",
}
by_source = {}
for r in results:
    src = r["source"].split(":")[0] if r["source"].startswith("workzone") else r["source"]
    by_source.setdefault(src, []).append(r)

for src, rows in by_source.items():
    n = len(rows)
    yes = sum(1 for r in rows if gate_is_yes(r["GATE"]))
    egos = {}
    for r in rows:
        egos[ego_label(r["EGO"])] = egos.get(ego_label(r["EGO"]), 0) + 1
    print(f"\n[{src}] n={n}  ({REF.get(src,'')})")
    print(f"  GATE = Yes: {yes}/{n} ({100*yes/n:.0f}%)")
    print(f"  EGO: {egos}")

print(f"\nrespostas completas: {OUT_PATH}")
