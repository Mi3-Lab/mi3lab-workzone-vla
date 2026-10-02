"""Exporta os TRÊS grafos do encoder por chunk, para deploy ONNX/TensorRT puro.

## Contexto

Decisão de design: modo JANELA (opção A) -- o cache é resetado a cada tick,
preservando exatamente a semântica atual do pipeline (sem mudança de
comportamento). Ver `compare_streaming_vs_window_vae.py` para o porquê:
manter o cache entre ticks daria ~60x, mas altera os 8 latentes mais antigos
da janela em 1,2-2,6 (escala unitária), e o modo janela é o que casa com a
convenção de treino de VAEs de vídeo.

O ganho que a opção A entrega é o que destrava o Orin: o engine builda com
**3,74 GB** em vez de **125,8 GB**, porque o grafo deixa de ter as 16
iterações do loop desenroladas (533 nós / 32 Conv, contra 44.366 / 495).

## Por que três grafos

Resetando o cache a cada tick, todo tick passa pelos chunks "frios", que têm
shapes diferentes do regime permanente:

  chunk 0:  1 frame,  SEM cache de entrada        -> latente + 24 caches (T=1)
  chunk 1:  4 frames, cache com T=1               -> latente + 24 caches (T=2)
  chunk 2+: 4 frames, cache com T=2  (permanente) -> latente + 24 caches (T=2)

Exportando os três, o deploy não precisa carregar o VAE em PyTorch no Orin
só para os dois primeiros chunks -- fica ONNX/TensorRT do começo ao fim.

## Validação

Encadeia os 16 chunks usando SÓ os grafos ONNX e compara contra o
`_encode` monolítico do PyTorch. Falha se divergir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
SNAPSHOT = BASE / ".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"


class ChunkEncoder(nn.Module):
    """Uma iteração do loop causal com feat_cache como I/O explícito.

    `n_cache_in=0` gera a variante fria (chunk 0), que não recebe cache.
    """

    def __init__(self, vae, slot_ids, n_slots, dtype, n_cache_in: int) -> None:
        super().__init__()
        self.vae = vae
        self.slot_ids = slot_ids
        self.n_slots = n_slots
        self.n_cache_in = n_cache_in
        self.patch_size = vae.config.patch_size
        mean = torch.tensor(vae.config.latents_mean, dtype=dtype).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=dtype).view(1, -1, 1, 1, 1)
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_inv_std", inv_std)

    def forward(self, video_chunk: torch.Tensor, *caches: torch.Tensor):
        x = patchify(video_chunk, patch_size=self.patch_size) if self.patch_size is not None else video_chunk
        feat_cache = [None] * self.n_slots
        if self.n_cache_in:
            for slot, tensor in zip(self.slot_ids, caches):
                feat_cache[slot] = tensor
        feat_idx = [0]
        out = self.vae.encoder(x, feat_cache=feat_cache, feat_idx=feat_idx)
        out = self.vae.quant_conv(out)
        mu = out[:, : out.shape[1] // 2]
        normalized = (mu - self.latent_mean) * self.latent_inv_std
        return (normalized, *tuple(feat_cache[s] for s in self.slot_ids))


def export_one(wrapper, example_inputs, path, n_cache, tag):
    names_in = ["video_chunk_rgb"] + [f"cache_in_{i}" for i in range(n_cache)]
    names_out = ["normalized_latent"] + [f"cache_out_{i}" for i in range(len(wrapper.slot_ids))]
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(path),
        input_names=names_in,
        output_names=names_out,
        opset_version=20,
        do_constant_folding=True,
        external_data=False,
        dynamic_axes=None,  # estático de propósito: é o que o TensorRT precisa fundir
        dynamo=False,
    )
    import onnx
    from collections import Counter
    m = onnx.load(str(path), load_external_data=False)
    c = Counter(n.op_type for n in m.graph.node)
    print(
        f"    [{tag}] {len(m.graph.node)} nós, Conv={c.get('Conv',0)}, "
        f"Pad={c.get('Pad',0)}, If={c.get('If',0)} -> {path.name}",
        flush=True,
    )
    return names_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=BASE / "models/cosmos3edge-vae-encoder-chunk3")
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--num-frames", type=int, default=61)
    args = ap.parse_args()

    dtype = torch.float16
    H, W, NF = args.height, args.width, args.num_frames
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Carregando o VAE...", flush=True)
    vae = (
        AutoencoderKLWan.from_pretrained(SNAPSHOT, subfolder="vae", torch_dtype=dtype, local_files_only=True)
        .to("cuda", dtype=dtype)
        .eval()
    )
    vae.clear_cache()
    n_slots = vae._enc_conv_num
    mean = torch.tensor(vae.config.latents_mean, dtype=dtype, device="cuda").view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=dtype, device="cuda").view(1, -1, 1, 1, 1)

    torch.manual_seed(0)
    video = torch.empty(1, 3, NF, H, W, dtype=dtype, device="cuda").uniform_(-1, 1)

    print(f"[2] Referência monolítica ({NF}x{H}x{W})...", flush=True)
    with torch.inference_mode():
        vae.clear_cache()
        enc = vae._encode(video)
        mu = enc[:, : enc.shape[1] // 2]
        ref = ((mu - mean) * inv_std).float().cpu().numpy()
    print(f"    {ref.shape}", flush=True)

    print("[3] Capturando os estados de cache dos 3 regimes...", flush=True)
    with torch.inference_mode():
        vae.clear_cache()
        xp = patchify(video, patch_size=vae.config.patch_size)
        vae._enc_conv_idx = [0]
        vae.encoder(xp[:, :, :1], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx)
        slot_ids = [i for i, e in enumerate(vae._enc_feat_map) if torch.is_tensor(e)]
        cache_cold = [vae._enc_feat_map[i].clone() for i in slot_ids]   # T=1
        vae._enc_conv_idx = [0]
        vae.encoder(xp[:, :, 1:5], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx)
        cache_warm = [vae._enc_feat_map[i].clone() for i in slot_ids]   # T=2
    print(f"    {len(slot_ids)} slots | frio(T=1): {sum(c.numel()*2 for c in cache_cold)/1e6:.0f} MB"
          f" | quente(T=2): {sum(c.numel()*2 for c in cache_warm)/1e6:.0f} MB", flush=True)

    print("[4] Exportando os 3 grafos (shapes estáticos)...", flush=True)
    w_cold = ChunkEncoder(vae, slot_ids, n_slots, dtype, n_cache_in=0).cuda().eval()
    w_warm = ChunkEncoder(vae, slot_ids, n_slots, dtype, n_cache_in=len(slot_ids)).cuda().eval()

    p0 = args.output_dir / "encoder-chunk0-fp16.onnx"
    p1 = args.output_dir / "encoder-chunk1-fp16.onnx"
    p2 = args.output_dir / "encoder-chunkN-fp16.onnx"

    export_one(w_cold, (video[:, :, :1].contiguous(),), p0, 0, "chunk0  1 frame, sem cache")
    names_out = export_one(
        w_warm, (video[:, :, 1:5].contiguous(), *[c.contiguous() for c in cache_cold]), p1, len(slot_ids),
        "chunk1  4 frames, cache T=1",
    )
    export_one(
        w_warm, (video[:, :, 5:9].contiguous(), *[c.contiguous() for c in cache_warm]), p2, len(slot_ids),
        "chunkN  4 frames, cache T=2",
    )

    meta = {
        "mode": "window (opção A) — cache resetado a cada tick, sem mudança de comportamento",
        "graphs": {"chunk0": p0.name, "chunk1": p1.name, "chunkN": p2.name},
        "chunks_per_tick": 1 + (NF - 1) // 4,
        "cache_slots": len(slot_ids),
        "slot_ids": slot_ids,
        "resolution": [H, W],
        "window_frames": NF,
        "static_shapes": True,
    }
    (args.output_dir / "export_info.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("[5] VALIDAÇÃO: 16 chunks usando SÓ ONNX vs monolítico PyTorch...", flush=True)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s0 = ort.InferenceSession(str(p0), sess_options=so, providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(p1), sess_options=so, providers=["CPUExecutionProvider"])
    sN = ort.InferenceSession(str(p2), sess_options=so, providers=["CPUExecutionProvider"])

    res = s0.run(names_out, {"video_chunk_rgb": video[:, :, :1].cpu().numpy()})
    lats = [res[0].astype(np.float32)]
    cache = res[1:]

    n_chunks = 1 + (NF - 1) // 4
    for i in range(1, n_chunks):
        feeds = {"video_chunk_rgb": video[:, :, 1 + 4 * (i - 1) : 1 + 4 * i].cpu().numpy()}
        for k, arr in enumerate(cache):
            feeds[f"cache_in_{k}"] = arr
        res = (s1 if i == 1 else sN).run(names_out, feeds)
        lats.append(res[0].astype(np.float32))
        cache = res[1:]

    got = np.concatenate(lats, axis=2)
    print(f"    obtido {got.shape} | referência {ref.shape}", flush=True)
    diff = np.abs(got - ref)
    rel = diff / np.maximum(np.abs(ref), 1e-6)
    print(f"    erro abs máximo: {diff.max():.6e}", flush=True)
    print(f"    erro rel p99:    {np.percentile(rel, 99):.6e}", flush=True)
    print()
    if diff.max() < 1e-2:
        print("    VEREDITO: EQUIVALENTE -- pipeline ONNX puro reproduz o monolítico.", flush=True)
    else:
        print("    VEREDITO: DIVERGENTE -- não usar.", flush=True)
        raise SystemExit(2)
    print("=== FIM ===", flush=True)


if __name__ == "__main__":
    main()
