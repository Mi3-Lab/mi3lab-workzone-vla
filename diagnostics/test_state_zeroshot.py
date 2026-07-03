"""Teste de viabilidade: estado de work zone direto do VLA (sem regras).

Em vez de 3 perguntas Q&A + regex + state machine com regras:
  1. UMA pergunta direta: qual o estado do veículo em relação à work zone?
  2. Score dos 4 candidatos via logprob (teacher forcing) — distribuição calibrada
  3. Filtro Bayesiano (HMM) no tempo — física vira matriz de transição,
     não REMAP hack; suavização vira inferência, não contadores.

Se o 2B zero-shot já separa os estados, plugamos direto no v9.
Se não separar, os pares (frame, estado) auto-derivados do ROADWork viram
dado de fine-tune no Stage 2 (mesma pergunta, resposta = estado).
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

STATES = ["OUTSIDE", "APPROACHING", "INSIDE", "EXITING"]
QUESTION = (
    "This is a dashcam view from a driving vehicle. Regarding the road work zone, "
    "what is the vehicle's current status? Answer with exactly one word: "
    "OUTSIDE (no work zone nearby), APPROACHING (work zone visible ahead), "
    "INSIDE (currently passing through the work zone), or EXITING (leaving the work zone)."
)

# Variante 2: vocabulário próximo do fine-tune ROADWork (frases do train.parquet)
QUESTION_B = "Is the vehicle before, inside, or past the work zone in this scene?"
STATES_B   = ["before the work zone", "inside the work zone", "past the work zone"]


@torch.no_grad()
def candidate_logprobs(pil, question, candidates):
    """Logprob médio por token de cada resposta candidata (teacher forcing)."""
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
        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in base:
                fwd_kwargs[k] = base[k].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(**fwd_kwargs)
        logits = out.logits[0, n - 1:-1].float()          # prevê os tokens do candidato
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = [lp[i, t].item() for i, t in enumerate(cand_ids)]
        scores.append(sum(tok_lp) / len(tok_lp))          # normalizado por comprimento
    scores = np.array(scores)
    probs = np.exp(scores - scores.max())
    return probs / probs.sum()


# ── Filtro Bayesiano (HMM forward) ─────────────────────────────────────────────
# Física nas transições (não REMAP): INSIDE nunca vira APPROACHING.
TRANS = np.array([
    # OUT   APP   INS   EXI      ← para
    [0.90, 0.10, 0.00, 0.00],  # de OUTSIDE
    [0.05, 0.70, 0.25, 0.00],  # de APPROACHING
    [0.00, 0.00, 0.92, 0.08],  # de INSIDE
    [0.20, 0.05, 0.15, 0.60],  # de EXITING (pode re-entrar em outra zona)
])

def hmm_filter(emissions):
    belief = np.array([0.85, 0.05, 0.05, 0.05])  # prior: começa fora
    out = []
    for e in emissions:
        belief = belief @ TRANS
        belief = belief * e
        belief = belief / belief.sum()
        out.append(belief.copy())
    return out


# ── Rodar nos 3 vídeos ─────────────────────────────────────────────────────────
N_FRAMES = 12
for city in ["boston", "denver", "chicago"]:
    path = f"{BASE}/videos/{city}.mp4"
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    idxs  = np.linspace(0, total - 1, N_FRAMES).astype(int)

    print(f"\n{'='*70}\n[{city.upper()}]  {total} frames @ {fps:.0f}fps")
    emissions = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p = candidate_logprobs(pil, QUESTION, STATES)
        emissions.append(p)
        dist = "  ".join(f"{s[:3]}={v:.2f}" for s, v in zip(STATES, p))
        print(f"  t={fi/fps:5.1f}s  raw: {dist}  →  {STATES[p.argmax()]}")
    cap.release()

    print(f"  --- HMM filtrado ---")
    for fi, belief in zip(idxs, hmm_filter(emissions)):
        dist = "  ".join(f"{s[:3]}={v:.2f}" for s, v in zip(STATES, belief))
        print(f"  t={fi/fps:5.1f}s  flt: {dist}  →  {STATES[belief.argmax()]}")

    # Variante B (vocabulário mais simples) só no primeiro e último frame
    print(f"  --- Variante B (before/inside/past) ---")
    for fi in [idxs[0], idxs[len(idxs)//2], idxs[-1]]:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read(); cap.release()
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p = candidate_logprobs(pil, QUESTION_B, STATES_B)
        dist = "  ".join(f"'{s}'={v:.2f}" for s, v in zip(STATES_B, p))
        print(f"  t={fi/fps:5.1f}s  {dist}")

print("\n=== DONE ===")
