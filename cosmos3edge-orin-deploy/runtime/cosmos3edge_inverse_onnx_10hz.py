#!/usr/bin/env python3
"""Run a new Cosmos3-Edge AV inverse-dynamics inference at every video tick.

At 10 Hz each update consumes a causal 61-frame window.  Before 6 seconds the
missing history is left-padded with the first frame.  This is intentionally an
offline benchmark: measured latency is recorded and never presented as real
time when it exceeds the 100 ms cadence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from PIL import Image

from cosmos3edge_inverse_onnx import (
    OrtInverseTransformer,
    OrtWanEncoder,
    build_pipeline_shell,
    build_vae_encoder,
    load_pipeline_configs,
)


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vae-encoder", type=Path, required=True)
    parser.add_argument("--inverse-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--stop-sample", type=int)
    # NVIDIA usa image_size 256 para tarefas de policy; o pipeline vinha
    # fixo em 480 (=480x832 para o aspecto do boston, ~6,5x mais pixels).
    parser.add_argument("--resolution-tier", type=int, default=480)
    parser.add_argument("--chunk-size", type=int, default=60)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    hz = float(manifest["target_hz"])
    samples = manifest["samples"]
    if abs(hz - 10.0) > 1e-9:
        raise RuntimeError(f"this acceptance run requires exactly 10 Hz, got {hz}")
    if len(samples) != int(manifest["sample_count"]):
        raise RuntimeError("manifest sample count mismatch")

    images: list[Image.Image] = []
    for sample in samples:
        with Image.open(sample["image"]) as image:
            images.append(image.convert("RGB").copy())

    load_started = time.perf_counter()
    transformer_config, vae_config = load_pipeline_configs(args.checkpoint)
    pipe = build_pipeline_shell(
        args.checkpoint,
        OrtInverseTransformer(args.inverse_transformer, transformer_config),
        build_vae_encoder(args.vae_encoder, vae_config),
    )
    load_s = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()

    start = max(0, args.start_sample)
    stop = min(len(samples), args.stop_sample if args.stop_sample is not None else len(samples))
    if start >= stop:
        raise RuntimeError(f"empty range [{start}, {stop})")

    payload = {
        "schema": "cosmos3edge.inverse_onnx_timeline.v1",
        "backend": "onnxruntime_cuda_fp16",
        "mode": "inverse_dynamics",
        "meaning": "observed AV motion over a causal 61-frame window; not a future planner",
        "manifest": str(args.manifest.resolve()),
        "checkpoint_config": str(args.checkpoint.resolve()),
        "vae_encoder_onnx": str(args.vae_encoder.resolve()),
        "inverse_transformer_onnx": str(args.inverse_transformer.resolve()),
        "video": manifest["video"],
        "video_sha256": manifest["video_sha256"],
        "target_hz": hz,
        "cadence_ms": 1000.0 / hz,
        "window_frames": 61,
        "window_duration_s": 6.0,
        "steps": args.steps,
        "seed": args.seed,
        "sample_range": [start, stop],
        "load_s": load_s,
        "updates": [],
    }

    for sample_idx in range(start, stop):
        history = images[max(0, sample_idx - 60) : sample_idx + 1]
        left_padding = 61 - len(history)
        window = [images[0]] * left_padding + history
        if len(window) != 61:
            raise AssertionError(len(window))
        condition = CosmosActionCondition(
            mode="inverse_dynamics",
            chunk_size=args.chunk_size,
            domain_name="av",
            resolution_tier=args.resolution_tier,
            video=window,
            view_point="ego_view",
        )
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt="You are an autonomous vehicle planning system.",
                action=condition,
                fps=hz,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=generator,
                output_type="latent",
                enable_safety_check=False,
            )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - started
        action = result.action[0].float().cpu().numpy()
        if action.shape != (60, 9) or not np.isfinite(action).all():
            raise RuntimeError(f"sample {sample_idx}: invalid action {action.shape}")
        cumulative = np.vstack([np.zeros(3), np.cumsum(action[:, :3], axis=0)])
        update = {
            "sample_idx": sample_idx,
            "timestamp_s": float(samples[sample_idx]["timestamp_s"]),
            "source_frame_idx": int(samples[sample_idx]["source_frame_idx"]),
            "window_start_sample_idx": max(0, sample_idx - 60),
            "left_padding_frames": left_padding,
            "latency_s": latency_s,
            "deadline_s": 1.0 / hz,
            "meets_realtime_deadline": latency_s <= 1.0 / hz,
            "shape": [60, 9],
            "action_60x9": action.tolist(),
            "cumulative_translation_61x3": cumulative.tolist(),
        }
        payload["updates"].append(update)
        latencies = [item["latency_s"] for item in payload["updates"]]
        payload["progress"] = {
            "completed": len(payload["updates"]),
            "requested": stop - start,
            "last_sample_idx": sample_idx,
            "latency_mean_s": statistics.mean(latencies),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "realtime_deadlines_met": sum(item["meets_realtime_deadline"] for item in payload["updates"]),
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        }
        write_payload(args.output, payload)
        print(
            f"sample={sample_idx:04d}/{stop - 1:04d} t={samples[sample_idx]['timestamp_s']:.1f}s "
            f"latency={latency_s:.3f}s padding={left_padding}",
            flush=True,
        )

    payload["complete"] = len(payload["updates"]) == stop - start
    payload["completed_at_unix_s"] = time.time()
    write_payload(args.output, payload)


if __name__ == "__main__":
    main()
