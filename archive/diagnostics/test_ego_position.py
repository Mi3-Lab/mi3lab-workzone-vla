"""Valida a pergunta EGO-CÊNTRICA de posição relativa à work zone.

Insight (usuário): ver workers != estar na zona. Câmera 4K vê workers a 100m
e o veículo ainda está APPROACHING. O estado é a posição do EGO em relação ao
início da zona, não o conteúdo da cena.

Testa se o 2B discrimina zero-shot: elementos "longe à frente" vs "ao lado do
veículo" vs "atrás" — o verdadeiro discriminador APPROACHING/INSIDE/EXITING.
"""
import os, sys, glob
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

from alpamayo1_5_sft.models.kd_model import build_student_model
from safetensors.torch import load_file
from transformers import AutoProcessor

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"
CKPT      = f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"

print("[LOAD] Carregando 2B...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()
proc = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


@torch.no_grad()
def candidate_probs(pil, question, candidates):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": question},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    base = proc(text=[text], images=[pil], return_tensors="pt")
    n = base["input_ids"].shape[1]
    scores = []
    for cand in candidates:
        cand_ids = proc.tokenizer.encode(cand, add_special_tokens=False)
        ids  = torch.cat([base["input_ids"],
                          torch.tensor([cand_ids], dtype=torch.long)], dim=1).cuda()
        attn = torch.ones_like(ids)
        kw = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in base:
                kw[k] = base[k].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(**kw)
        lp = torch.log_softmax(out.logits[0, n - 1:-1].float(), dim=-1)
        tok_lp = [lp[i, t].item() for i, t in enumerate(cand_ids)]
        scores.append(sum(tok_lp) / len(tok_lp))
    s = np.array(scores)
    p = np.exp(s - s.max())
    return p / p.sum()


# Variante 1: pergunta de posição relativa, respostas descritivas
Q1 = ("Look at the work zone elements such as cones, barriers, signs and workers. "
      "Where are the CLOSEST work zone elements relative to this vehicle?")
C1 = [
    "There are no work zone elements visible in this scene.",
    "They are far ahead of the vehicle; the vehicle has not reached the work zone yet.",
    "They are right beside the vehicle; the vehicle is passing through the work zone now.",
    "They are behind the vehicle; the vehicle has passed the work zone.",
]
L1 = ["none", "far ahead", "beside now", "behind"]

# Variante 2: pergunta sobre a linha de cones mais próxima (mais geométrica)
Q2 = ("This is a dashcam image. Consider the nearest traffic cone or barrier. "
      "Is it in the lower half of the image (very close to the vehicle) or in the "
      "upper half / near the horizon (still far away)? Or is none visible?")
C2 = [
    "No cone or barrier is visible.",
    "It is near the horizon, still far away from the vehicle.",
    "It is in the lower half of the image, very close to the vehicle.",
]
L2 = ["none", "far", "close"]

N_FRAMES = 12
for city in ["boston", "denver", "chicago"]:
    path = f"{BASE}/videos/{city}.mp4"
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    idxs  = np.linspace(0, total - 1, N_FRAMES).astype(int)

    print(f"\n{'='*72}\n[{city.upper()}]")
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p1 = candidate_probs(pil, Q1, C1)
        p2 = candidate_probs(pil, Q2, C2)
        d1 = "  ".join(f"{l}={v:.2f}" for l, v in zip(L1, p1))
        d2 = "  ".join(f"{l}={v:.2f}" for l, v in zip(L2, p2))
        print(f"  t={fi/fps:5.1f}s  Q1[{L1[p1.argmax()]:>10s}]: {d1}")
        print(f"           Q2[{L2[p2.argmax()]:>10s}]: {d2}")
    cap.release()

print("\n=== DONE ===")
