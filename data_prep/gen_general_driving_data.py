"""Self-distillation: Cosmos-Reason2-2B BASE gera Q&A de direção GERAL.

Motivação (usuário): work zone é UM long tail. O modelo final deve ser um VLA
de direção geral que TAMBÉM domina work zones — não um detector monomaníaco.
Teste 170550 provou: nosso 2B colapsou em templates ("OUTLUS NO. NO.") em
cenas normais, enquanto o Cosmos-2B base raciocina bem sobre elas.

Este script gera dados de retenção: o BASE responde perguntas gerais de
direção sobre frames diversos (BDD100k + frames dos vídeos ROADWork — muitos
fora da zona). As respostas dele viram alvo de treino do Stage 6, ancorando
o modelo ao comportamento geral original (anti catastrophic forgetting).
"""
import os, sys, glob, random, json
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import AutoProcessor, AutoModelForImageTextToText

COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"
OUT_DIR   = f"{BASE}/data/roadwork/lingoqa_general"
FRAME_DIR = os.path.join(OUT_DIR, "video_frames")
os.makedirs(FRAME_DIR, exist_ok=True)
random.seed(11)

QUESTIONS = [
    "Describe this driving scene.",
    "What should the ego vehicle do next, and why?",
    "Are there any hazards ahead? Explain briefly.",
    "Is it safe to change to the left lane right now? Explain.",
    "Is it safe to change to the right lane right now? Explain.",
    "What are the other road users around the ego vehicle doing?",
    "How do the current weather and lighting conditions affect driving?",
    "Should the ego vehicle adjust its speed? Why or why not?",
]
Q_PER_FRAME = 3   # perguntas sorteadas por frame

print("[LOAD] Cosmos-Reason2-2B base (professor de direção geral)...")
model = AutoModelForImageTextToText.from_pretrained(
    COSMOS_2B, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
proc = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


@torch.no_grad()
def generate(pil, question, max_new_tokens=110):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": question},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil], return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=proc.tokenizer.eos_token_id)
    n = inputs["input_ids"].shape[1]
    return proc.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


# ── Coletar frames ─────────────────────────────────────────────────────────────
# 1) BDD100k (direção normal, diversa)
bdd = sorted(glob.glob(f"{BASE}/data/roadwork/bdd100k/*.jpg"))
sources = [("bdd", p) for p in bdd]

# 2) Frames dos vídeos ROADWork (5 frames/segmento, muitos fora da zona;
#    cenas COM zona também valem — o base responde em linguagem de direção
#    geral, ensinando integração em vez de separação de domínios)
vids = sorted(glob.glob(f"{BASE}/data/roadwork/videos/*_snippet.mp4"))
random.shuffle(vids)
for v in vids[:160]:
    cap = cv2.VideoCapture(v)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        continue
    for fi in np.linspace(0, total - 1, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        name = f"{os.path.basename(v).replace('_snippet.mp4','')}_{fi:05d}.jpg"
        out_path = os.path.join(FRAME_DIR, name)
        if not os.path.exists(out_path):
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sources.append(("vid", out_path))
    cap.release()

random.shuffle(sources)
print(f"[FRAMES] total: {len(sources)}  (bdd={sum(1 for s,_ in sources if s=='bdd')}, "
      f"video={sum(1 for s,_ in sources if s=='vid')})")

# ── Gerar Q&A ──────────────────────────────────────────────────────────────────
rows = []
for i, (src, path) in enumerate(sources):
    try:
        pil = Image.open(path).convert("RGB")
    except Exception:
        continue
    rel = (os.path.join("bdd", os.path.basename(path)) if src == "bdd"
           else os.path.join("video_frames", os.path.basename(path)))
    for q in random.sample(QUESTIONS, Q_PER_FRAME):
        try:
            a = generate(pil, q)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(path)}: {str(e)[:60]}")
            continue
        if len(a) < 20:   # respostas degeneradas
            continue
        rows.append({"images": [rel], "question": q, "answer": a})
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(sources)} frames → {len(rows)} pares")
        pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, "train_partial.parquet"), index=False)

# symlink bdd para o data_root
bdd_link = os.path.join(OUT_DIR, "bdd")
if not os.path.exists(bdd_link):
    os.symlink(f"{BASE}/data/roadwork/bdd100k", bdd_link)

random.shuffle(rows)
n_val = max(100, len(rows) // 20)
pd.DataFrame(rows[n_val:]).to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
pd.DataFrame(rows[:n_val]).to_parquet(os.path.join(OUT_DIR, "val.parquet"), index=False)
print(f"\n[DONE] train={len(rows)-n_val}  val={n_val}  → {OUT_DIR}")
