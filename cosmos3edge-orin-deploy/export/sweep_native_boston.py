"""Varredura resolução × passos de difusão no boston.mp4, em PyTorch NATIVO.

## Por que nativo e não ONNX

O pacote ONNX está CONGELADO na configuração 480×832: o transformer foi
exportado com contagem fixa de tokens (`position_ids` espera 4969; em
tier 256 seriam 1145) e recusa qualquer outra resolução. Testar a
configuração recomendada pela NVIDIA exigiria reexportar o transformer.
O modelo nativo não tem essa restrição -- responde a pergunta estratégica
sem nenhum trabalho de export. E já foi medido que o nativo é mais rápido
que o ONNX nos dois componentes (VAE 1002ms vs 8679ms; MoT 155ms vs 221ms
por passo).

## O critério correto de tempo real

O modelo usa ACTION CHUNKING: uma inferência produz um bloco de ações
executadas ao longo do tempo. A NVIDIA declara "32 actions per inference
on Jetson Thor - while achieving real-time control at 15 Hz" (32/15 =
2,13 s de controle por inferência).

Aqui: 60 ações a 10 Hz = **6,0 s de controle por inferência**. Portanto o
orçamento é 6,0 s, NÃO os 100 ms do período do tick.

Referência oficial da NVIDIA no model card: Policy DROID leva 6,32 s no
Jetson Thor T5000 e 8,63 s no T3000, com `image_size: 256`.

## O que mede

Para cada (resolution_tier, num_inference_steps): latência e a trajetória
resultante, comparada contra a configuração de referência, para ver onde
a qualidade começa a degradar.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path("/data/wesleyferreiramaia/wokzone-alpamayo")
BUDGET_S = 6.0  # 60 ações @ 10 Hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=BASE/".hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de")
    ap.add_argument("--manifest", type=Path,
                    default=BASE/"outputs/cosmos3edge_boston_10hz/timeline.json")
    ap.add_argument("--output", type=Path, default=BASE/"outputs/boston_native_sweep/sweep.json")
    ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--start-sample", type=int, default=60)
    args = ap.parse_args()

    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition

    print("[1] Carregando o pipeline NATIVO (bf16)...", flush=True)
    t0 = time.perf_counter()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    print(f"    carregado em {time.perf_counter()-t0:.1f}s", flush=True)

    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    images = []
    for s in samples[: args.start_sample + args.ticks]:
        with Image.open(s["image"]) as im:
            images.append(im.convert("RGB").copy())
    print(f"[2] {len(images)} frames carregados do boston.mp4", flush=True)

    configs = [(480, 30), (256, 30), (256, 16), (256, 8), (256, 4)]
    results = {}

    for tier, steps in configs:
        tag = f"tier{tier}_steps{steps}"
        print(f"\n[3] === {tag} ===", flush=True)
        lats, actions = [], []
        try:
            for i in range(args.ticks):
                idx = args.start_sample + i
                window = images[max(0, idx - 60): idx + 1]
                window = [images[0]] * (61 - len(window)) + window
                cond = CosmosActionCondition(
                    mode="inverse_dynamics", chunk_size=60, domain_name="av",
                    resolution_tier=tier, video=window, view_point="ego_view",
                )
                gen = torch.Generator(device="cuda").manual_seed(0)
                torch.cuda.synchronize(); t = time.perf_counter()
                with torch.inference_mode():
                    out = pipe(
                        prompt="You are an autonomous vehicle planning system.",
                        action=cond, fps=10.0, num_inference_steps=steps,
                        guidance_scale=1.0, generator=gen,
                        output_type="latent", enable_safety_check=False,
                    )
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                a = out.action[0].float().cpu().numpy()
                lats.append(dt); actions.append(a)
                print(f"    tick {idx}: {dt:.3f}s  acao {a.shape}", flush=True)
            results[tag] = {"tier": tier, "steps": steps,
                            "latencies": lats, "actions": [a.tolist() for a in actions]}
        except Exception as e:
            print(f"    FALHOU: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[tag] = {"tier": tier, "steps": steps, "error": str(e)[:300]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    print("\n" + "="*78)
    print(f"{'config':<18}{'latencia':>10}{'tempo real':>12}{'desloc Z':>11}{'vel':>11}{'vs ref':>12}")
    print("-"*78)
    ref = None
    for tier, steps in configs:
        r = results.get(f"tier{tier}_steps{steps}", {})
        if "error" in r:
            print(f"tier{tier} steps{steps:<4}  FALHOU: {r['error'][:44]}")
            continue
        lat = float(np.mean(r["latencies"]))
        A = np.array(r["actions"])
        xyz = np.cumsum(A[:, :, :3], axis=1)[:, -1, :]
        z = xyz[:, 2].mean(); vel = np.linalg.norm(xyz, axis=1).mean()/6.0*3.6
        if ref is None:
            ref = A; dev = "referencia"
        else:
            dev = f"{np.abs(A-ref).max():.3f}"
        ok = "SIM" if lat <= BUDGET_S else "nao"
        print(f"tier{tier} steps{steps:<4}{lat:>9.2f}s{ok:>12}{z:>10.1f}m{vel:>9.1f}km/h{dev:>12}")
    print("-"*78)
    print(f"orcamento (60 acoes @10Hz, action chunking): {BUDGET_S:.1f}s")
    print(f"referencia NVIDIA: Policy DROID = 6,32s no Thor T5000 / 8,63s no T3000")
    print("=== FIM ===")


if __name__ == "__main__":
    main()
