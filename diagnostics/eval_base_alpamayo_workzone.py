"""Testa o Alpamayo-1.5-10B PURO (sem nenhum fine-tuning nosso) nos mesmos
frames dos 5 videos de benchmark (4 de obra + comma2k19 sem obra), pra medir
se o modelo base ja sabe identificar zona de obra out-of-the-box.

Perguntas por frame:
  GATE -- existencia, neutra, binaria (mesma do redesign cascade)
  EGO  -- a mesma pergunta do v16 (com pressuposicao), pra comparacao direta
          com o nosso modelo fine-tunado

Saida: JSONL com todas as respostas + resumo agregado por fonte.
"""
import os, sys, glob, json
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import AutoProcessor
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

A1_CKPT = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VLM_PATH = f"{BASE}/models/Cosmos-Reason2-8B"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"
OUT_PATH = f"{BASE}/mi3lab-workzone-vla/diagnostics/base_alpamayo_workzone_answers.jsonl"

QUESTIONS = {
    "GATE": ("Are there any road work indicators in this scene "
             "(cones, barriers, temporary signs, workers, work vehicles)? "
             "Answer Yes or No."),
    "EGO": ("Regarding the road work zone, what is this vehicle's current "
            "position status: OUTSIDE, APPROACHING, or INSIDE?"),
}
MAX_TOK = {"GATE": 32, "EGO": 48}

print("[LOAD] Alpamayo-1.5-10B base (SEM fine-tuning)...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=A1_CKPT,
    vlm_name_or_path=VLM_PATH,
)
model = model.to(torch.bfloat16).cuda().eval()
processor = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)
print("[LOAD] pronto.")


def infer(image, question, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt",
                       padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    n_in = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(generated[0][n_in:],
                                      skip_special_tokens=True).strip()


# Mesmos frames dos benchmarks (ja extraidos em disco)
frames = []
for p in sorted(glob.glob(f"{ENG}/video_frames/*.jpg")):
    city = os.path.basename(p).rsplit("_", 1)[0]
    frames.append(("workzone:" + city, p))
for p in sorted(glob.glob(f"{ENG}/negcontrol_video_frames/*.jpg")):
    frames.append(("comma2k19", p))
print(f"{len(frames)} frames ({sum(1 for s,_ in frames if s.startswith('workzone'))} obra, "
      f"{sum(1 for s,_ in frames if s=='comma2k19')} sem obra)")

results = []
with open(OUT_PATH, "w") as fout:
    for i, (source, path) in enumerate(frames):
        image = Image.open(path).convert("RGB")
        row = {"source": source, "frame": os.path.basename(path)}
        for qname, q in QUESTIONS.items():
            row[qname] = infer(image, q, MAX_TOK[qname])
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        results.append(row)
        if i % 20 == 0:
            print(f"[{i+1}/{len(frames)}] {source} GATE={row['GATE'][:50]!r}")

# ------------------------------------------------------------- resumo
print("\n" + "=" * 70)
print("RESUMO -- Alpamayo-1.5-10B base (sem fine-tuning)")
print("=" * 70)


def gate_is_yes(t):
    tl = t.strip().lower()
    return tl.startswith("yes") or ("yes" in tl[:40] and not tl.startswith("no"))


def ego_label(t):
    tu = t.upper()
    for lab in ("OUTSIDE", "APPROACHING", "INSIDE"):
        if lab in tu:
            return lab
    return "OTHER"


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
    print(f"\n[{src}] n={n}")
    print(f"  GATE = Yes: {yes}/{n} ({100*yes/n:.0f}%)"
          + ("  <- deveria ser ALTO (obra real)" if src == "workzone"
             else "  <- deveria ser ~0 (sem obra)"))
    print(f"  EGO: {egos}")

print(f"\nrespostas completas: {OUT_PATH}")
