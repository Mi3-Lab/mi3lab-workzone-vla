"""Diagnóstico: por que o sensor ego do v10 dá valores idênticos p/ frames diferentes.

Hipótese: EGO_CANDS são frases longas (~40 tokens) com boilerplate previsível
após a primeira palavra. Média de logprob por token dilui o sinal que depende
da imagem (concentrado no primeiro token) com dezenas de tokens quase
determinísticos e iguais entre candidatos. Testa full-sentence vs short-label.
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
CKPT      = f"{BASE}/checkpoints/sft_stage5_egostate/checkpoint-1700"

print("[LOAD] Carregando 2B Stage-5...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
if not sd:
    sd = load_file(os.path.join(CKPT, "model.safetensors"), device="cpu")
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()
proc = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

Q_EGO = (
    "Regarding the road work zone, what is this vehicle's current position status: "
    "OUTSIDE, APPROACHING, or INSIDE?"
)
FULL_CANDS = [
    "OUTSIDE. No work zone elements near the vehicle. "
    "The road ahead is clear of construction activity.",
    "APPROACHING. Work zone elements are visible ahead but the vehicle has not "
    "reached them yet. Prepare to slow down and follow work zone guidance.",
    "INSIDE. Work zone elements are immediately beside the vehicle; it is "
    "currently passing through the work zone. Maintain reduced speed and "
    "maximum caution.",
]
SHORT_CANDS = ["OUTSIDE.", "APPROACHING.", "INSIDE."]
LABELS = ["OUTSIDE", "APPROACHING", "INSIDE"]


@torch.no_grad()
def candidate_scores(pil, question, candidates):
    """Retorna (probs_avg_por_token, probs_soma_total, n_tokens_por_cand)."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": question},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    base = proc(text=[text], images=[pil], return_tensors="pt")
    n = base["input_ids"].shape[1]
    avg_scores, sum_scores, ntoks = [], [], []
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
        avg_scores.append(sum(tok_lp) / len(tok_lp))
        sum_scores.append(sum(tok_lp))
        ntoks.append(len(cand_ids))
    def softmax(s):
        s = np.array(s); p = np.exp(s - s.max()); return p / p.sum()
    return softmax(avg_scores), softmax(sum_scores), ntoks


N_FRAMES = 8
for city in ["boston", "denver", "chicago"]:
    path = f"{BASE}/videos/{city}.mp4"
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    idxs  = np.linspace(0, total - 1, N_FRAMES).astype(int)

    print(f"\n{'='*76}\n[{city.upper()}]")
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        p_full_avg, p_full_sum, nt_full = candidate_scores(pil, Q_EGO, FULL_CANDS)
        p_short, _, nt_short = candidate_scores(pil, Q_EGO, SHORT_CANDS)

        fa = "  ".join(f"{l[:3]}={v:.2f}" for l, v in zip(LABELS, p_full_avg))
        fs = "  ".join(f"{l[:3]}={v:.2f}" for l, v in zip(LABELS, p_full_sum))
        sh = "  ".join(f"{l[:3]}={v:.2f}" for l, v in zip(LABELS, p_short))
        print(f"  t={fi/fps:5.1f}s")
        print(f"    full-avg (n_tok={nt_full}) : {fa}")
        print(f"    full-sum                  : {fs}")
        print(f"    short    (n_tok={nt_short}): {sh}")
    cap.release()

print("\n=== DONE ===")
