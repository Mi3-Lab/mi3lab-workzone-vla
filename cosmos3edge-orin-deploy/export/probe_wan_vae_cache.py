"""Sonda a estrutura do cache causal (`feat_cache`) do encoder do Wan VAE.

Motivacao: o export ONNX atual DESENROLOU as 16 iteracoes do loop de
streaming do `_encode` num grafo monolitico (495 Conv = 16 x ~31), o que
causa 44k nos, 125,8 GB de RAM no build e 686,8 ms de latencia por tick --
sendo que a cada tick do loop de 10Hz so' 1 frame e' novo (a janela de 61
desliza de 1 em 1).

O caminho certo e' exportar UMA iteracao com o cache causal como I/O
explicito do grafo. Antes de escrever esse export, esta sonda responde as
perguntas que determinam se e' viavel e qual o formato:

  1. Quantos slots de cache o encoder usa (`_enc_conv_num`)?
  2. Em regime permanente (chunk i>=1), quais slots sao TENSORES (viraveis
     em I/O do ONNX) e quais ficam None/"Rep" (strings, nao exportaveis)?
  3. Os shapes dos slots sao ESTAVEIS entre chunks? (se mudarem, um unico
     engine nao serve)
  4. A saida latente por chunk e' a mesma que a fatia correspondente da
     saida monolitica de 61 frames?

Nao exporta nada -- so' mede e reporta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
SNAPSHOT = BASE / ".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"

NUM_FRAMES = 61
H, W = 400, 720
DTYPE = torch.float16

print("[1] Carregando o VAE...", flush=True)
vae = (
    AutoencoderKLWan.from_pretrained(SNAPSHOT, subfolder="vae", torch_dtype=DTYPE, local_files_only=True)
    .to("cuda", dtype=DTYPE)
    .eval()
)
print(f"    patch_size={vae.config.patch_size}  use_tiling={vae.use_tiling}", flush=True)

vae.clear_cache()
n_slots = vae._enc_conv_num
print(f"[2] Slots de cache do encoder: {n_slots}", flush=True)

print(f"[3] Rodando o _encode instrumentado em {NUM_FRAMES}x{H}x{W}...", flush=True)
video = torch.zeros(1, 3, NUM_FRAMES, H, W, dtype=DTYPE, device="cuda")
video.uniform_(-1.0, 1.0)

from diffusers.models.autoencoders.autoencoder_kl_wan import patchify


def snapshot_cache(cache):
    """Descreve o estado atual do cache: tipo e shape de cada slot."""
    desc = []
    for i, e in enumerate(cache):
        if e is None:
            desc.append((i, "None", None))
        elif isinstance(e, str):
            desc.append((i, f'str:"{e}"', None))
        elif torch.is_tensor(e):
            desc.append((i, "tensor", tuple(e.shape)))
        else:
            desc.append((i, type(e).__name__, None))
    return desc


with torch.inference_mode():
    vae.clear_cache()
    x = patchify(video, patch_size=vae.config.patch_size) if vae.config.patch_size is not None else video
    print(f"    apos patchify: {tuple(x.shape)}", flush=True)

    iter_ = 1 + (x.shape[2] - 1) // 4
    print(f"    iteracoes do loop: {iter_}", flush=True)

    per_chunk_out = []
    cache_states = {}
    for i in range(iter_):
        vae._enc_conv_idx = [0]
        if i == 0:
            chunk = x[:, :, :1, :, :]
        else:
            chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
        out_i = vae.encoder(chunk, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx)
        per_chunk_out.append(out_i)
        if i in (0, 1, 2, 3, iter_ - 1):
            cache_states[i] = snapshot_cache(vae._enc_feat_map)
        if i <= 2:
            print(f"    chunk {i}: entrada {tuple(chunk.shape)} -> saida {tuple(out_i.shape)}", flush=True)

    out_loop = torch.cat(per_chunk_out, dim=2)
    enc_loop = vae.quant_conv(out_loop)
    print(f"    saida concatenada + quant_conv: {tuple(enc_loop.shape)}", flush=True)

print()
print("[4] Estado do cache DEPOIS de cada chunk:", flush=True)
for i, desc in sorted(cache_states.items()):
    kinds = {}
    for _, kind, _ in desc:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"    apos chunk {i:2d}: {kinds}", flush=True)

print()
print("[5] Estabilidade dos shapes entre chunk 1, 2, 3 e o ultimo:", flush=True)
ref = {i: s for i, k, s in cache_states[1] if k == "tensor"}
stable = True
for ci in (2, 3, iter_ - 1):
    cur = {i: s for i, k, s in cache_states[ci] if k == "tensor"}
    if set(cur.keys()) != set(ref.keys()):
        print(f"    chunk {ci}: CONJUNTO DE SLOTS DIFERENTE do chunk 1", flush=True)
        stable = False
        continue
    diffs = [(i, ref[i], cur[i]) for i in ref if ref[i] != cur[i]]
    if diffs:
        stable = False
        print(f"    chunk {ci}: {len(diffs)} slots com shape DIFERENTE. ex: {diffs[:3]}", flush=True)
    else:
        print(f"    chunk {ci}: todos os {len(cur)} slots com shape IDENTICO ao chunk 1  OK", flush=True)

print()
n_tensor = len(ref)
print(f"[6] Em regime permanente: {n_tensor} de {n_slots} slots sao tensores", flush=True)
tot_bytes = sum(int(torch.tensor(s).prod()) * 2 for s in ref.values())
print(f"    memoria total do cache: {tot_bytes/1e6:.1f} MB (fp16)", flush=True)
print("    shapes distintos:", flush=True)
from collections import Counter

for shp, cnt in Counter(ref.values()).most_common(10):
    print(f"      {shp}  x{cnt}", flush=True)

print()
print("[7] Conferindo contra o encode monolitico (o que o ONNX atual faz)...", flush=True)
with torch.inference_mode():
    vae.clear_cache()
    enc_mono = vae._encode(video)
print(f"    monolitico: {tuple(enc_mono.shape)}   loop: {tuple(enc_loop.shape)}", flush=True)
if enc_mono.shape == enc_loop.shape:
    d = (enc_mono.float() - enc_loop.float()).abs()
    print(f"    erro abs maximo: {d.max().item():.3e}   bit-exato: {torch.equal(enc_mono, enc_loop)}", flush=True)

print()
print("===== VEREDITO DE VIABILIDADE =====", flush=True)
if stable:
    print(f"    VIAVEL: {n_tensor} tensores de cache com shape estavel.", flush=True)
    print(f"    O grafo por chunk teria 1+{n_tensor} entradas e 1+{n_tensor} saidas.", flush=True)
else:
    print("    ATENCAO: shapes do cache NAO sao estaveis entre chunks --", flush=True)
    print("    um unico engine nao serve; investigar quais slots variam.", flush=True)
print("=== FIM ===", flush=True)
