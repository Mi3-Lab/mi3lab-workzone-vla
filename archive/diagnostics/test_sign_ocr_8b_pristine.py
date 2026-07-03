"""Comparativo: Cosmos-Reason2-8B PURO (sem fine-tune de direção) lê as placas?

O Alpamayo-10B (A1) teve o VLM fine-tunado pela NVIDIA para trajetória e
degradou o OCR (32% recall, saídas com artefatos). Este teste isola a variável:
se o 8B puro ler bem, o limite é o fine-tune (recuperável via dados); se ler
mal, o limite é a legibilidade das imagens.
"""
import os, sys, json, random
import torch
from PIL import Image
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import AutoProcessor, AutoModelForImageTextToText

VLM_PATH = f"{BASE}/models/Cosmos-Reason2-8B"

print("[LOAD] Cosmos-Reason2-8B puro...")
model = AutoModelForImageTextToText.from_pretrained(
    VLM_PATH, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
proc = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

Q_OCR = ("Read the text on the temporary traffic control signs in this scene. "
         "Report each sign's text exactly as written. If no sign text is legible, "
         "say 'none legible'.")


@torch.no_grad()
def generate(pil, question, max_new_tokens=80):
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


# Mesmas 15 imagens de val (seed 7) do teste do 10B
random.seed(7)
d = json.load(open(f"{BASE}/data/roadwork/scene/annotations/instances_val_gps_split_with_signs.json"))
imgs = {i["id"]: i for i in d["images"]}
gt_by_img = {}
for a in d["annotations"]:
    at = a.get("attributes", {})
    if isinstance(at, dict):
        t = (at.get("sign_text") or "").strip()
        if t and not at.get("text_occluded", False):
            gt_by_img.setdefault(a["image_id"], set()).add(t.upper())

sample_ids = random.sample(sorted(gt_by_img.keys()), 15)
print(f"\nPARTE 1 — mesmas 15 imagens VAL (comparável ao 10B)\n{'='*72}")
hits, total = 0, 0
for img_id in sample_ids:
    fn = imgs[img_id]["file_name"]
    path = f"{BASE}/data/roadwork/scene/images/{fn}"
    if not os.path.exists(path):
        continue
    pil = Image.open(path).convert("RGB")
    if max(pil.size) > 1920:
        r = 1920 / max(pil.size)
        pil = pil.resize((int(pil.size[0]*r), int(pil.size[1]*r)))
    pred = generate(pil, Q_OCR)
    gts = gt_by_img[img_id]
    hit = sum(1 for g in gts if g in pred.upper())
    hits += hit
    total += len(gts)
    print(f"\n  [{fn[:60]}]")
    print(f"    GT  : {sorted(gts)}")
    print(f"    PRED: {pred[:150]}")
    print(f"    match: {hit}/{len(gts)}")

print(f"\n  >>> RECALL 8B puro: {hits}/{total} ({100*hits/max(total,1):.0f}%)  [10B Alpamayo: 6/19 = 32%]")

print(f"\nPARTE 2 — frames dos vídeos de demo\n{'='*72}")
for city, frames in [("boston", [0, 30, 90]), ("seattle_unseen", [0, 120, 600])]:
    cap = cv2.VideoCapture(f"{BASE}/videos/{city}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{city.upper()}]")
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"  t={fi/fps:5.1f}s: {generate(pil, Q_OCR)[:130]}")
    cap.release()

# Parte 3: frame 1080p original do seattle (placas maiores)
print(f"\nPARTE 3 — seattle snippet ORIGINAL 1920x1080 (legibilidade máxima)\n{'='*72}")
src = f"{BASE}/data/roadwork/videos/seattle_11a6d1c0ae1944ec9bfe1ef71377be7a_000000_08700_snippet.mp4"
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
for fi in [0, 120, 300, 600]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(f"  t={fi/fps:5.1f}s: {generate(pil, Q_OCR)[:130]}")
cap.release()

print("\n=== DONE ===")
