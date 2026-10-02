#!/usr/bin/env python3
"""Normalize TensorRT Edge-LLM batch output onto the 10 Hz timeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-s", type=float, required=True)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    raw = json.loads(args.raw_output.read_text())
    responses = sorted(raw["responses"], key=lambda item: int(item["request_idx"]))
    samples = manifest["samples"]
    if len(responses) != len(samples):
        raise RuntimeError(f"received {len(responses)} responses for {len(samples)} samples")

    updates = []
    for expected_idx, (sample, response) in enumerate(zip(samples, responses)):
        request_idx = int(response["request_idx"])
        if request_idx != expected_idx:
            raise RuntimeError(f"request order mismatch: {request_idx} != {expected_idx}")
        exact = response["output_text"]
        updates.append(
            {
                "sample_idx": expected_idx,
                "timestamp_s": float(sample["timestamp_s"]),
                "source_frame_idx": int(sample["source_frame_idx"]),
                "finish_reason": response.get("finish_reason"),
                "exact_text": exact,
                "display_text": exact.replace("<|im_end|>", "").strip(),
            }
        )

    profile = None
    if args.profile_log is not None:
        log = args.profile_log.read_text()
        def number(pattern: str) -> float:
            match = re.search(pattern, log)
            if not match:
                raise RuntimeError(f"profile metric not found: {pattern}")
            return float(match.group(1))

        prefill_ms = number(r"LLM Prefill - Total Runs: \d+, Total GPU Time: [0-9.]+ ms, Average: ([0-9.]+) ms")
        decode_ms_per_token = number(r"Average Time per Token: ([0-9.]+) ms\nLLM Generation")
        vision_ms = number(r"Vision Encoder - Total Runs: \d+, Total GPU Time: [0-9.]+ ms, Average: ([0-9.]+) ms")
        generated_tokens = int(number(r"Generated Tokens: (\d+)"))
        average_generated_tokens = generated_tokens / len(updates)
        estimated_gpu_ms = vision_ms + prefill_ms + average_generated_tokens * decode_ms_per_token
        profile = {
            "prefill_average_ms": prefill_ms,
            "decode_ms_per_token": decode_ms_per_token,
            "vision_encoder_average_ms": vision_ms,
            "generated_tokens_total": generated_tokens,
            "generated_tokens_average": average_generated_tokens,
            "estimated_average_gpu_pipeline_ms": estimated_gpu_ms,
            "estimated_average_meets_100ms_deadline": estimated_gpu_ms <= 100.0,
            "peak_gpu_memory_mb": number(r"Peak GPU Memory: ([0-9.]+) MB"),
            "profile_log": str(args.profile_log.resolve()),
        }

    payload = {
        "schema": "cosmos3edge.reasoner_timeline.v1",
        "backend": "TensorRT Edge-LLM",
        "precision": "reasoner W4A16 AWQ INT4 + visual FP16",
        "manifest": str(args.manifest.resolve()),
        "raw_output": str(args.raw_output.resolve()),
        "video": manifest["video"],
        "video_sha256": manifest["video_sha256"],
        "target_hz": manifest["target_hz"],
        "sample_count": len(updates),
        "wall_s_including_all_requests": args.wall_s,
        "average_wall_s_per_request": args.wall_s / len(updates),
        "per_request_latency_available": False,
        "gpu_profile": profile,
        "notes": [
            "Every exact model response is retained.",
            "The C++ summary exposes aggregate GPU-stage averages, not latency for each individual request.",
        ],
        "updates": updates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    text_output = args.text_output or args.output.with_name(args.output.stem + "_exact_texts.txt")
    text_output.write_text(
        "\n\n".join(
            f"[{item['sample_idx']:03d}] t={item['timestamp_s']:.1f}s frame={item['source_frame_idx']} "
            f"finish={item['finish_reason']}\n{item['exact_text']}"
            for item in updates
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()
