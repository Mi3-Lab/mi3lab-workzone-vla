"""Controle: .generate() de verdade com o checkpoint Stage-5, para separar

(a) colapso de treino (modelo sempre gera a mesma resposta, ignora a imagem)
de
(b) bug de inferência (candidate_probs via forward manual não bate com o
    preprocessing usado no treino, então SÓ o scoring está quebrado).

Se a geração livre variar corretamente com a cena, o problema é (b) — bug no
meu candidate_probs, não no fine-tune.
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
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
model = model.to(torch.bfloat16).cuda().eval()
proc = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

Q_EGO = (
    "Regarding the road work zone, what is this vehicle's current position status: "
    "OUTSIDE, APPROACHING, or INSIDE?"
)
Q_DESC = "Describe the work zone elements visible in this scene."


@torch.no_grad()
def generate(pil, question, max_new_tokens=60):
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


N_FRAMES = 6
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
        ego  = generate(pil, Q_EGO, max_new_tokens=50)
        desc = generate(pil, Q_DESC, max_new_tokens=50)
        print(f"  t={fi/fps:5.1f}s")
        print(f"    EGO : {ego[:100]}")
        print(f"    DESC: {desc[:100]}")
    cap.release()

print("\n=== DONE ===")
