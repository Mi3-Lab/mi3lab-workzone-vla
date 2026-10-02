#!/usr/bin/env python3
"""Encoder do VAE por chunk -- substituto direto de `OrtWanEncoder`.

## Por que existe

O grafo monolítico do encoder (`encoder-av61-fp16.onnx`) traz as 16 iterações
do loop causal do Wan VAE DESENROLADAS num único grafo (495 Conv = 16 x ~31,
44.366 nós). Construir um engine TensorRT a partir dele exige **125,8 GB de
RAM do host** -- não cabe nos 64 GB do AGX Orin. Como engines do TensorRT são
travados ao hardware (têm que ser construídos no alvo), esse grafo
simplesmente **não tem como existir como engine no Orin**.

Aqui o encoder é chamado uma iteração por vez, com o `feat_cache` causal
passando como I/O explícito. Cada grafo tem ~533 nós / 32 Conv e constrói com
menos de 4 GB.

## Semântica preservada (modo JANELA)

O cache é **resetado a cada chamada de `encode()`**, exatamente como o
pipeline faz hoje (`vae.encode()` começa com `clear_cache()`). Isso mantém o
comportamento idêntico ao atual.

Manter o cache ENTRE ticks daria ~60x a mais, mas muda o que o modelo vê: os
8 latentes mais antigos da janela divergem de 1,2 a 2,6 (escala unitária),
porque no modo janela o latente 0 é calculado sem histórico e no streaming
com o histórico real. Além disso, VAEs de vídeo são treinados em clipes que
começam do zero, então streaming fica fora da distribuição de treino. Essa
troca exige validar a POLÍTICA (ações geradas), não só o VAE -- por isso não
é o padrão aqui. Ver `RELATORIO_COSMOS3EDGE_TENSORRT_FUSAO.md`.

## Custo medido (A100, engines FP16)

    chunk0 (1 frame, sem cache) : 19,1 ms
    chunk1 (4 frames, cache T=1): 40,6 ms
    chunkN (4 frames, cache T=2): 41,5 ms  x14
    ------------------------------------------
    total por tick              : 640,9 ms   (ONNX Runtime CUDA EP: 8.679 ms)

## Validação

A cadeia dos 16 chunks reproduz o `_encode` monolítico: erro abs máx
5,9e-03 / rel p99 3,5e-02 -- ruído de fp16, não divergência estrutural.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


def _make_session(path: Path) -> ort.InferenceSession:
    """Sessão ORT com TensorRT EP opcional (mesma convenção do pacote)."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    use_trt = os.environ.get("COSMOS3EDGE_ORT_TENSORRT", "0") == "1"
    providers: list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if use_trt:
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "COSMOS3EDGE_ORT_TENSORRT=1 mas este ONNX Runtime não tem TensorrtExecutionProvider"
            )
        cache_root = Path(os.environ.get("COSMOS3EDGE_TRT_CACHE", "./engines/ort-tensorrt")).resolve()
        cache_dir = cache_root / path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": True,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


class _DeterministicDistribution:
    """Mesma interface mínima que o pipeline espera de `latent_dist`."""

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value

    def sample(self, generator=None) -> torch.Tensor:  # noqa: ARG002
        return self._value


class ChunkedWanEncoder(nn.Module):
    """Substituto direto de `OrtWanEncoder`, com a mesma assinatura `encode()`.

    Espera três grafos no diretório (ou os engines correspondentes, via
    TensorRT EP):
        encoder-chunk0-fp16.onnx   1 frame,  sem cache      -> latente + cache(T=1)
        encoder-chunk1-fp16.onnx   4 frames, cache(T=1)     -> latente + cache(T=2)
        encoder-chunkN-fp16.onnx   4 frames, cache(T=2)     -> latente + cache(T=2)

    Os três regimes existem porque o cache é resetado a cada tick: todo tick
    passa pelos chunks "frios", cujos shapes de cache diferem do permanente
    (`CACHE_T=2`, mas o primeiro chunk só tem 1 frame de histórico).
    """

    def __init__(self, model_dir: Path, config) -> None:
        super().__init__()
        model_dir = Path(model_dir)
        self.sessions = {
            "chunk0": _make_session(model_dir / "encoder-chunk0-fp16.onnx"),
            "chunk1": _make_session(model_dir / "encoder-chunk1-fp16.onnx"),
            "chunkN": _make_session(model_dir / "encoder-chunkN-fp16.onnx"),
        }
        self.config = config
        self.dtype = torch.float16
        self.call_count = 0
        self.total_call_s = 0.0
        self.chunk_count = 0

        out0 = self.sessions["chunkN"].get_outputs()
        self.output_names = [o.name for o in out0]
        self.n_cache = len(self.output_names) - 1

        self.register_parameter(
            "_device_anchor", nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)
        )
        self.register_buffer("mean", torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(config.latents_std).view(1, -1, 1, 1, 1), persistent=False)

    def _run(self, tag: str, chunk: np.ndarray, cache: list | None):
        feeds = {"video_chunk_rgb": chunk}
        if cache is not None:
            for i, arr in enumerate(cache):
                feeds[f"cache_in_{i}"] = arr
        res = self.sessions[tag].run(self.output_names, feeds)
        self.chunk_count += 1
        return res[0], res[1:]

    def encode(self, video: torch.Tensor, return_dict: bool = True):
        """video: (1,3,T,H,W) em [-1,1]. Cache resetado aqui = semântica de janela."""
        video_np = video.half().cpu().numpy()
        num_frames = video_np.shape[2]
        n_chunks = 1 + (num_frames - 1) // 4

        started = time.perf_counter()

        lat, cache = self._run("chunk0", video_np[:, :, :1], None)
        lats = [lat]
        for i in range(1, n_chunks):
            tag = "chunk1" if i == 1 else "chunkN"
            piece = video_np[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            lat, cache = self._run(tag, piece, cache)
            lats.append(lat)

        normalized = torch.from_numpy(np.concatenate(lats, axis=2)).to(
            device=video.device, dtype=torch.float16
        )

        self.total_call_s += time.perf_counter() - started
        self.call_count += 1

        raw = normalized * self.std.to(normalized) + self.mean.to(normalized)
        result = SimpleNamespace(latent_dist=_DeterministicDistribution(raw))
        return result if return_dict else (result.latent_dist,)


def _self_test(model_dir: Path, height: int, width: int, frames: int) -> None:
    """Confere shapes e reporta a latência por chunk. Não substitui a
    validação numérica contra o grafo monolítico (essa está em
    `diagnostics/export_wan_vae_chunk_all.py`)."""
    cfg = SimpleNamespace(latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    enc = ChunkedWanEncoder(model_dir, cfg)
    video = torch.empty(1, 3, frames, height, width, dtype=torch.float16).uniform_(-1, 1)
    out = enc.encode(video)
    lat = out.latent_dist.mode()
    expected_t = 1 + (frames - 1) // 4
    print(f"latente: {tuple(lat.shape)}  (esperado T={expected_t})")
    assert lat.shape[2] == expected_t, f"T errado: {lat.shape[2]} != {expected_t}"
    print(f"chunks executados: {enc.chunk_count}")
    print(f"tempo total: {enc.total_call_s*1000:.1f} ms "
          f"({enc.total_call_s*1000/enc.chunk_count:.1f} ms/chunk)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Self-test do encoder por chunk")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--frames", type=int, default=61)
    a = ap.parse_args()
    _self_test(a.model_dir, a.height, a.width, a.frames)
