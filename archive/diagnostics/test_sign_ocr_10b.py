"""Teste: o 10B BASE (Alpamayo-1.5-10B, Cosmos-Reason2-8B sem fine-tune ROADWork)
consegue ler o texto das placas corretamente?

Parte 1 — quantitativa: imagens do val split with_signs com transcrição humana
(ground truth). Compara predição vs GT.
Parte 2 — qualitativa: frames dos vídeos de demo (boston, seattle_unseen).
"""
import os, sys, glob, json, random
import torch
import numpy as np
from PIL import Image
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from transformers import AutoProcessor

A1_CKPT  = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VLM_PATH = f"{BASE}/models/Cosmos-Reason2-8B"

print("[LOAD] Carregando 10B BASE (sem fine-tune ROADWork)...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=A1_CKPT, vlm_name_or_path=VLM_PATH)
model = model.to(torch.bfloat16).cuda().eval()
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
        out = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=proc.tokenizer.eos_token_id,
            return_dict_in_generate=False, output_logits=False)
    n = inputs["input_ids"].shape[1]
    return proc.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


# ── Parte 1: val images com ground truth ─────────────────────────────────────
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
print(f"\n{'='*76}\nPARTE 1 — {len(sample_ids)} imagens de VAL com transcrição humana (GT)\n{'='*76}")
hits, total = 0, 0
for img_id in sample_ids:
    fn = imgs[img_id]["file_name"]
    path = f"{BASE}/data/roadwork/scene/images/{fn}"
    if not os.path.exists(path):
        continue
    pil = Image.open(path).convert("RGB")
    # reduzir imagens gigantes (4032px) para acelerar
    if max(pil.size) > 1920:
        r = 1920 / max(pil.size)
        pil = pil.resize((int(pil.size[0]*r), int(pil.size[1]*r)))
    pred = generate(pil, Q_OCR)
    gts = gt_by_img[img_id]
    pred_up = pred.upper()
    hit = sum(1 for g in gts if g in pred_up)
    hits += hit
    total += len(gts)
    print(f"\n  [{fn}]")
    print(f"    GT  : {sorted(gts)}")
    print(f"    PRED: {pred[:150]}")
    print(f"    match: {hit}/{len(gts)}")

print(f"\n  >>> RECALL de textos GT encontrados na predição: {hits}/{total} ({100*hits/max(total,1):.0f}%)")

# ── Parte 2: frames dos vídeos de demo ───────────────────────────────────────
print(f"\n{'='*76}\nPARTE 2 — frames dos vídeos de demo\n{'='*76}")
for city, frames in [("boston", [0, 30, 90]), ("seattle_unseen", [0, 120, 600])]:
    path = f"{BASE}/videos/{city}.mp4"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{city.upper()}]")
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pred = generate(pil, Q_OCR)
        print(f"  t={fi/fps:5.1f}s: {pred[:130]}")
    cap.release()

print("\n=== DONE ===")
