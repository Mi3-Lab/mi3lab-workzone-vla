"""Mede o VAE encoder NATIVO em PyTorch, para comparar contra ONNX/TensorRT.

Pergunta que responde: todo o trabalho de ONNX/TensorRT vale a pena frente a
simplesmente rodar o modelo original? Para o transformer a resposta ja' e'
conhecida (nativo 155ms < ONNX 221ms, ou seja o ONNX PIORA; so' o TensorRT
com 102ms melhora). Falta o VAE.
"""
import time, torch
from pathlib import Path
from diffusers import AutoencoderKLWan

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
SNAP = BASE/".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
H, W, NF = 480, 832, 61

vae = AutoencoderKLWan.from_pretrained(SNAP, subfolder="vae", torch_dtype=torch.float16,
        local_files_only=True).to("cuda", torch.float16).eval()
video = torch.empty(1,3,NF,H,W, dtype=torch.float16, device="cuda").uniform_(-1,1)

with torch.inference_mode():
    for _ in range(2):                      # warmup
        vae.clear_cache(); vae._encode(video)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(5):
        torch.cuda.synchronize(); t0=time.perf_counter()
        vae.clear_cache(); out = vae._encode(video)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)

import statistics
print(f"VAE encoder NATIVO PyTorch FP16 @ {NF}x{H}x{W}")
print(f"  saida: {tuple(out.shape)}")
print(f"  media:  {statistics.mean(ts)*1000:.1f} ms")
print(f"  mediana:{statistics.median(ts)*1000:.1f} ms   min: {min(ts)*1000:.1f} ms")
print()
print("  ONNX Runtime (Codex):  8679 ms")
print("  TensorRT (medido hoje): 811 ms")
print(f"  => nativo vs TensorRT: {statistics.mean(ts)*1000/811:.2f}x")
