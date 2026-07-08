"""Testa os Cosmos-Reason2 BASE (2B e 8B, sem nenhum fine-tuning) nos mesmos
217 frames do benchmark (137 obra + 80 comma2k19 sem obra) -- complemento ao
eval do Alpamayo-10B (que nao segue formato VQA por ser VLA especializado).

O 2B e' o ancestral direto do nosso workzone-2b; o 8B e' o backbone do
Alpamayo. Se eles discriminarem bem (Yes na obra, No no video limpo), o
vies "sempre sim" do nosso modelo foi INTRODUZIDO pelo nosso SFT sem
negativos, e o Stage 7 e' restauracao de capacidade, nao ensino do zero.
"""
import os, sys, glob, json
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})
from transformers import AutoModelForImageTextToText, AutoProcessor

ENG = f"{BASE}/models/workzone-2b-stage6-engines"

QUESTIONS = {
    "GATE": ("Are there any road work indicators in this scene "
             "(cones, barriers, temporary signs, workers, work vehicles)? "
             "Answer Yes or No."),
    "EGO": ("Regarding the road work zone, what is this vehicle's current "
            "position status: OUTSIDE, APPROACHING, or INSIDE?"),
}
MAX_TOK = {"GATE": 32, "EGO": 48}

frames = []
for p in sorted(glob.glob(f"{ENG}/video_frames/*.jpg")):
    city = os.path.basename(p).rsplit("_", 1)[0]
    frames.append(("workzone:" + city, p))
for p in sorted(glob.glob(f"{ENG}/negcontrol_video_frames/*.jpg")):
    frames.append(("comma2k19", p))
print(f"{len(frames)} frames")


def gate_is_yes(t):
    tl = t.strip().lower()
    return tl.startswith("yes") or ("yes" in tl[:40] and not tl.startswith("no"))


def ego_label(t):
    tu = t.upper()
    for lab in ("OUTSIDE", "APPROACHING", "INSIDE"):
        if lab in tu:
            return lab
    return "OTHER"


for model_name in ("Cosmos-Reason2-2B", "Cosmos-Reason2-8B"):
    model_dir = f"{BASE}/models/{model_name}"
    out_path = f"{BASE}/mi3lab-workzone-vla/diagnostics/base_{model_name.lower()}_workzone_answers.jsonl"
    print(f"\n[LOAD] {model_name} base...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    results = []
    with open(out_path, "w") as fout:
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
                    gen = model.generate(**inputs,
                                         max_new_tokens=MAX_TOK[qname],
                                         do_sample=False,
                                         pad_token_id=proc.tokenizer.eos_token_id)
                n_in = inputs["input_ids"].shape[1]
                row[qname] = proc.tokenizer.decode(
                    gen[0][n_in:], skip_special_tokens=True).strip()
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            results.append(row)
            if i % 40 == 0:
                print(f"  [{i+1}/{len(frames)}] {source} GATE={row['GATE'][:50]!r}")

    print(f"\n===== RESUMO {model_name} =====")
    by_source = {}
    for r in results:
        src = "workzone" if r["source"].startswith("workzone") else "comma2k19"
        by_source.setdefault(src, []).append(r)
    for src, rows in by_source.items():
        n = len(rows)
        yes = sum(1 for r in rows if gate_is_yes(r["GATE"]))
        egos = {}
        for r in rows:
            egos[ego_label(r["EGO"])] = egos.get(ego_label(r["EGO"]), 0) + 1
        tgt = "ALTO (obra real)" if src == "workzone" else "~0 (sem obra)"
        print(f"  [{src}] n={n} | GATE=Yes: {yes}/{n} ({100*yes/n:.0f}%) <- deveria ser {tgt}")
        print(f"    EGO: {egos}")

    del model
    torch.cuda.empty_cache()
