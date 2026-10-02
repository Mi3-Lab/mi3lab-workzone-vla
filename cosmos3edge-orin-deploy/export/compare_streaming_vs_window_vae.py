"""Streaming (cache atravessa ticks) vs janela (reset por tick): os latentes
do VAE são os mesmos?

## Por que isso importa

O ganho de ~60x do encoder por chunk depende de MANTER o `feat_cache` entre
ticks. Mas o pipeline atual (`cosmos3edge_inverse_onnx_10hz.py`) faz o
oposto: a cada tick pega a janela de 61 frames e chama `vae.encode()`, que
começa com `clear_cache()` -- ou seja, RESETA o estado causal e trata o
primeiro frame da janela como início de sequência.

Os dois modos NÃO são equivalentes por construção:

  - JANELA:    latente k usa histórico truncado no início da janela
               (+ left-padding com o primeiro frame nos primeiros 6s)
  - STREAMING: latente k usa o histórico causal real desde o início do vídeo

Streaming é discutivelmente mais correto (é o modo natural de servir um
modelo causal, e elimina o artefato do padding), mas é uma MUDANÇA DE
COMPORTAMENTO. Este script mede o tamanho dessa diferença para que a decisão
seja informada, em vez de embutida numa "otimização".

Tudo roda em PyTorch/CUDA nos dois lados -- a pergunta é de semântica do
modelo, não de runtime, então não há ruído de backend contaminando.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
SNAPSHOT = BASE / ".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"

WINDOW = 61          # tamanho da janela causal do pipeline
LATENTS = 16         # 1 + (61-1)//4


def encode_window(vae, video, mean, inv_std):
    """Modo JANELA: o que o pipeline faz hoje -- reset do cache a cada chamada."""
    with torch.inference_mode():
        vae.clear_cache()
        enc = vae._encode(video)
        mu = enc[:, : enc.shape[1] // 2]
        return (mu - mean) * inv_std


class StreamingEncoder:
    """Modo STREAMING: cache persistente, encoda só o chunk novo."""

    def __init__(self, vae, mean, inv_std):
        self.vae = vae
        self.mean = mean
        self.inv_std = inv_std
        self.vae.clear_cache()
        self.started = False
        self.latents = []

    def push_chunk(self, rgb_chunk):
        """rgb_chunk: (1,3,1,H,W) no primeiro, (1,3,4,H,W) depois."""
        with torch.inference_mode():
            x = patchify(rgb_chunk, patch_size=self.vae.config.patch_size)
            self.vae._enc_conv_idx = [0]
            raw = self.vae.encoder(x, feat_cache=self.vae._enc_feat_map, feat_idx=self.vae._enc_conv_idx)
            raw = self.vae.quant_conv(raw)
            mu = raw[:, : raw.shape[1] // 2]
            lat = (mu - self.mean) * self.inv_std
            self.latents.append(lat)
            return lat

    def window_latents(self, n=LATENTS):
        """Os últimos n latentes -- o que o transformer consumiria."""
        return torch.cat(self.latents[-n:], dim=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--total-frames", type=int, default=241, help="duração do vídeo simulado")
    args = ap.parse_args()

    dtype = torch.float16
    H, W = args.height, args.width
    NT = args.total_frames

    print("[1] Carregando o VAE...", flush=True)
    vae = (
        AutoencoderKLWan.from_pretrained(SNAPSHOT, subfolder="vae", torch_dtype=dtype, local_files_only=True)
        .to("cuda", dtype=dtype)
        .eval()
    )
    mean = torch.tensor(vae.config.latents_mean, dtype=dtype, device="cuda").view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=dtype, device="cuda").view(1, -1, 1, 1, 1)

    print(f"[2] Vídeo simulado: {NT} frames de {H}x{W} (~{NT/10:.1f}s a 10Hz)", flush=True)
    torch.manual_seed(0)
    # vídeo com continuidade temporal (não ruído independente por frame), pra
    # não subestimar artificialmente o efeito do histórico causal
    base = torch.empty(1, 3, 1, H, W, dtype=dtype, device="cuda").uniform_(-1, 1)
    drift = torch.empty(1, 3, NT, H, W, dtype=dtype, device="cuda").uniform_(-0.15, 0.15)
    video = (base + drift.cumsum(dim=2) * 0.02).clamp(-1, 1).to(dtype)

    print("[3] STREAMING: cache atravessando todos os chunks...", flush=True)
    stream = StreamingEncoder(vae, mean, inv_std)
    stream.push_chunk(video[:, :, :1])
    n_chunks = 1 + (NT - 1) // 4
    for i in range(1, n_chunks):
        stream.push_chunk(video[:, :, 1 + 4 * (i - 1) : 1 + 4 * i])
    print(f"    {n_chunks} chunks -> {len(stream.latents)} latentes", flush=True)

    print("[4] JANELA: reset a cada tick (o que o pipeline faz hoje)...", flush=True)
    # compara em alguns instantes: logo após encher a janela, e mais tarde
    checkpoints = [WINDOW - 1, WINDOW + 39, NT - 1]
    checkpoints = [c for c in checkpoints if c < NT]

    print()
    print("    tick | latentes comparados | erro abs máx | erro rel p99 | corr")
    print("    -----+---------------------+--------------+--------------+------")
    for idx in checkpoints:
        window = video[:, :, idx - WINDOW + 1 : idx + 1]
        lat_win = encode_window(vae, window, mean, inv_std)

        n_avail = (idx // 4) + 1
        lat_str = torch.cat(stream.latents[max(0, n_avail - LATENTS) : n_avail], dim=2)
        if lat_str.shape[2] != lat_win.shape[2]:
            k = min(lat_str.shape[2], lat_win.shape[2])
            lat_str = lat_str[:, :, -k:]
            lat_win = lat_win[:, :, -k:]

        a = lat_win.float()
        b = lat_str.float()
        diff = (a - b).abs()
        rel = diff / a.abs().clamp_min(1e-6)
        corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        print(
            f"    {idx:4d} | {lat_win.shape[2]:19d} | {diff.max().item():12.4e} | "
            f"{torch.quantile(rel.flatten().float(), 0.99).item():12.4e} | {corr:.4f}",
            flush=True,
        )

    print()
    print("[5] Diferença por posição do latente na janela (tick final):", flush=True)
    idx = checkpoints[-1]
    window = video[:, :, idx - WINDOW + 1 : idx + 1]
    lat_win = encode_window(vae, window, mean, inv_std)
    n_avail = (idx // 4) + 1
    lat_str = torch.cat(stream.latents[max(0, n_avail - LATENTS) : n_avail], dim=2)
    k = min(lat_str.shape[2], lat_win.shape[2])
    print("      posição 0 = mais antigo da janela | posição -1 = mais recente", flush=True)
    for p in range(k):
        d = (lat_win[:, :, p].float() - lat_str[:, :, p].float()).abs().max().item()
        bar = "#" * min(40, int(d * 400))
        print(f"      lat {p:2d}: erro máx {d:.4e}  {bar}", flush=True)

    print()
    print("===== INTERPRETAÇÃO =====", flush=True)
    print("  Se o erro cair conforme a posição avança, o efeito do reset de", flush=True)
    print("  cache é local ao início da janela e some com histórico -- então", flush=True)
    print("  streaming é seguro para os latentes recentes, que é o que mais", flush=True)
    print("  importa para controle. Se for uniforme, a mudança afeta a janela", flush=True)
    print("  inteira e precisa de validação de comportamento, não só numérica.", flush=True)
    print("=== FIM ===", flush=True)


if __name__ == "__main__":
    main()
