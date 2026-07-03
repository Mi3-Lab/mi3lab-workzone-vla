"""Baixa (streaming HF) um clipe do PhysicalAI-AV e salva os tensores em /data.

Roda no login node (tem internet). O job de GPU depois carrega o .pt offline.
Curiosidade do usuário: como o Alpamayo-10B BASE gera raciocínio + trajetória
no formato nativo dele, ANTES de qualquer work zone.
"""
import os, sys
import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ.update({
    "HF_HOME": f"{BASE}/.hf_cache",
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
})

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"   # clipe do exemplo oficial
OUT = f"{BASE}/data/aiav_samples"
os.makedirs(OUT, exist_ok=True)

print(f"[FETCH] clip {CLIP_ID} (streaming HF)...")
data = load_physical_aiavdataset(CLIP_ID, t0_us=5_100_000)
for k, v in data.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")
torch.save(data, f"{OUT}/{CLIP_ID}.pt")
print(f"[SAVED] {OUT}/{CLIP_ID}.pt")
