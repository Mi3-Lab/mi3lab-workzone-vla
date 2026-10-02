#!/usr/bin/env python3
"""Predict a new 6-second AV policy trajectory at every video tick."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import CosmosActionCondition
from PIL import Image

from cosmos3edge_inverse_onnx import (
    OrtWanEncoder,
    build_pipeline_shell,
    build_vae_encoder,
    load_pipeline_configs,
)
from cosmos3edge_policy_onnx import OrtPolicyTransformer
from cosmos3edge_av_pose import integrate_av_relative_poses


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vae-encoder", type=Path, required=True)
    parser.add_argument("--policy-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "You are an autonomous vehicle planning system. Predict a safe future ego trajectory "
            "through the observed road-work scene."
        ),
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--stop-sample", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    hz = float(manifest["target_hz"])
    samples = manifest["samples"]
    if abs(hz - 10.0) > 1e-9 or len(samples) != int(manifest["sample_count"]):
        raise RuntimeError("policy timeline requires the exact 10 Hz manifest")
    images = []
    for sample in samples:
        with Image.open(sample["image"]) as image:
            images.append(image.convert("RGB").copy())

    load_started = time.perf_counter()
    transformer_config, vae_config = load_pipeline_configs(args.checkpoint)
    ort_transformer = OrtPolicyTransformer(args.policy_transformer, transformer_config)
    ort_encoder = build_vae_encoder(args.vae_encoder, vae_config)
    pipe = build_pipeline_shell(args.checkpoint, ort_transformer, ort_encoder)
    load_s = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()

    start = max(0, args.start_sample)
    stop = min(len(samples), args.stop_sample if args.stop_sample is not None else len(samples))
    if start >= stop:
        raise RuntimeError(f"empty range [{start}, {stop})")
    payload = {
        "schema": "cosmos3edge.policy_av_onnx_timeline_se3.v2",
        "backend": "onnxruntime_cuda_fp16",
        "mode": "policy",
        "domain": "av",
        "meaning": "experimental base-model AV future rollout; not a post-trained or control-validated AV policy",
        "manifest": str(args.manifest.resolve()),
        "checkpoint_config": str(args.checkpoint.resolve()),
        "vae_encoder_onnx": str(args.vae_encoder.resolve()),
        "policy_transformer_onnx": str(args.policy_transformer.resolve()),
        "video": manifest["video"],
        "video_sha256": manifest["video_sha256"],
        "target_hz": hz,
        "cadence_ms": 1000.0 / hz,
        "future_steps": 60,
        "future_horizon_s": 6.0,
        "action_convention": "9D pose: translation[0:3] + Zhou rot6d[3:9]; translation units are model-native",
        "trajectory_representation": "backward-framewise relative 9D poses integrated by SE(3) composition",
        "steps": args.steps,
        "seed": args.seed,
        "prompt": args.prompt,
        "sample_range": [start, stop],
        "load_s": load_s,
        "updates": [],
    }

    for sample_idx in range(start, stop):
        transformer_s_before = ort_transformer.total_call_s
        transformer_calls_before = ort_transformer.call_count
        encoder_s_before = ort_encoder.total_call_s
        condition = CosmosActionCondition(
            mode="policy", chunk_size=60, domain_name="av", resolution_tier=480,
            image=images[sample_idx], view_point="ego_view",
        )
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt, action=condition, fps=hz,
                num_inference_steps=args.steps, guidance_scale=1.0,
                generator=generator, use_system_prompt=False,
                output_type="latent", enable_safety_check=False,
            )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - started
        transformer_s = ort_transformer.total_call_s - transformer_s_before
        transformer_calls = ort_transformer.call_count - transformer_calls_before
        vae_encoder_s = ort_encoder.total_call_s - encoder_s_before
        action = result.action[0].float().cpu().numpy()
        if action.shape != (60, 9) or not np.isfinite(action).all():
            raise RuntimeError(f"sample {sample_idx}: invalid policy action {action.shape}")
        absolute_poses = integrate_av_relative_poses(action)
        payload["updates"].append({
            "sample_idx": sample_idx,
            "timestamp_s": float(samples[sample_idx]["timestamp_s"]),
            "source_frame_idx": int(samples[sample_idx]["source_frame_idx"]),
            "prediction_start_s": float(samples[sample_idx]["timestamp_s"]),
            "prediction_end_s": float(samples[sample_idx]["timestamp_s"]) + 6.0,
            "latency_s": latency_s,
            "deadline_s": 1.0 / hz,
            "meets_realtime_deadline": latency_s <= 1.0 / hz,
            "latency_breakdown_s": {
                "vae_encoder_onnx": vae_encoder_s,
                "policy_transformer_onnx": transformer_s,
                "policy_transformer_calls": transformer_calls,
                "host_prepost_scheduler": max(0.0, latency_s - vae_encoder_s - transformer_s),
            },
            "shape": [60, 9],
            "action_60x9": action.tolist(),
            "absolute_pose_translation_61x3": absolute_poses[:, :3, 3].tolist(),
            "pose_integration": "SE(3) backward_framewise: T[i+1] = T[i] @ delta_T[i]",
        })
        latencies = [item["latency_s"] for item in payload["updates"]]
        payload["progress"] = {
            "completed": len(latencies), "requested": stop - start,
            "last_sample_idx": sample_idx,
            "latency_mean_s": statistics.mean(latencies),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "realtime_deadlines_met": sum(item["meets_realtime_deadline"] for item in payload["updates"]),
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        }
        atomic_json(args.output, payload)
        print(f"sample={sample_idx:04d}/{stop - 1:04d} t={samples[sample_idx]['timestamp_s']:.1f}s latency={latency_s:.3f}s", flush=True)

    payload["complete"] = len(payload["updates"]) == stop - start
    payload["completed_at_unix_s"] = time.time()
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
