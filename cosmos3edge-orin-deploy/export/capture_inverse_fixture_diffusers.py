"""Captura um fixture do transformer inverse a partir do pipeline DIFFUSERS,
na resolução desejada — insumo do
`export_cosmos3edge_diffusers_inverse_transformer_onnx.py`.

## Por que existe

O grafo ONNX do transformer é exportado a partir de um *fixture*: uma chamada
real do denoiser, com todos os tensores e metadados de empacotamento. Como o
fixture fixa as formas, **o grafo fica congelado na resolução em que o fixture
foi capturado** (o pacote atual espera `position_ids` com 4969 tokens, de
480×832; em tier 256 seriam 1145).

O script de captura existente (`capture_cosmos3edge_denoiser_fixture.py`) usa
o caminho `cosmos_framework.inference`. Este aqui usa o `Cosmos3OmniPipeline`
do fork diffusers — o mesmo caminho que o runtime de deploy usa — e aceita
`--resolution-tier`, permitindo gerar fixtures para qualquer configuração.

## Como funciona

Envolve `pipe.transformer.forward` num interceptador, roda UMA chamada real do
pipeline em inverse dynamics, guarda os kwargs e a saída da primeira invocação
do denoiser, e aborta a inferência (não precisa completar os 30 passos).

Uso:
    python capture_inverse_fixture_diffusers.py \
        --resolution-tier 256 --output <fixture.pt> [--manifest timeline.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
SNAPSHOT = BASE / ".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"

# tensores que entram como I/O explícito do grafo (variam por passo);
# o resto do kwargs é metadado estático de empacotamento, gravado no fixture.
DYNAMIC_INPUTS = ("input_ids", "position_ids", "action_timesteps")


class _Captured(RuntimeError):
    """Aborta a inferência assim que o primeiro passo do denoiser é capturado."""


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous()
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=SNAPSHOT)
    ap.add_argument("--manifest", type=Path,
                    default=BASE / "outputs/cosmos3edge_boston_10hz/timeline.json")
    ap.add_argument("--resolution-tier", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--sample-idx", type=int, default=60)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition

    print(f"[1] Carregando o pipeline nativo (bf16)...", flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")

    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    idx = args.sample_idx
    images = []
    for s in samples[: idx + 1]:
        with Image.open(s["image"]) as im:
            images.append(im.convert("RGB").copy())
    window = images[max(0, idx - 60): idx + 1]
    window = [images[0]] * (61 - len(window)) + window
    print(f"[2] Janela de {len(window)} frames, tier={args.resolution_tier}", flush=True)

    captured: dict = {}
    original_forward = pipe.transformer.forward

    def intercept(*a, **kw):
        if not captured:
            out = original_forward(*a, **kw)
            captured["kwargs"] = _to_cpu(kw)
            captured["outputs"] = _to_cpu(out)
            raise _Captured
        return original_forward(*a, **kw)

    pipe.transformer.forward = intercept

    cond = CosmosActionCondition(
        mode="inverse_dynamics", chunk_size=args.chunk_size, domain_name="av",
        resolution_tier=args.resolution_tier, video=window, view_point="ego_view",
    )
    print(f"[3] Rodando uma chamada do denoiser para capturar...", flush=True)
    try:
        with torch.inference_mode():
            pipe(
                prompt="You are an autonomous vehicle planning system.",
                action=cond, fps=10.0, num_inference_steps=30,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(0),
                output_type="latent", enable_safety_check=False,
            )
    except _Captured:
        print("    capturado (inferência abortada de propósito)", flush=True)
    finally:
        pipe.transformer.forward = original_forward

    if not captured:
        raise SystemExit("o denoiser nunca foi chamado — nada capturado")

    k = captured["kwargs"]
    print(f"[4] Formas dos tensores dinâmicos:", flush=True)
    for name in DYNAMIC_INPUTS:
        v = k.get(name)
        if isinstance(v, torch.Tensor):
            print(f"      {name}: {tuple(v.shape)}", flush=True)
    for name in ("vision_tokens", "action_tokens"):
        v = k.get(name)
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            print(f"      {name}[0]: {tuple(v[0].shape)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, args.output)
    meta = {
        "profile": f"inverse_dynamics AV, tier {args.resolution_tier}, chunk {args.chunk_size}",
        "resolution_tier": args.resolution_tier,
        "chunk_size": args.chunk_size,
        "window_frames": len(window),
        "manifest": str(args.manifest),
        "sample_idx": idx,
        "position_ids_len": int(k["position_ids"].shape[1]) if isinstance(k.get("position_ids"), torch.Tensor) else None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[5] Salvo em {args.output}", flush=True)
    print(f"    {json.dumps(meta, indent=2)}", flush=True)
    print("=== FIM ===", flush=True)


if __name__ == "__main__":
    main()
