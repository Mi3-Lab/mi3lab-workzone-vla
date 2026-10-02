#!/usr/bin/env python3
"""Run the native Cosmos3-Edge reasoner on every 10 Hz timeline frame."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cosmos_framework.inference.inference as inference_module
from cosmos_framework.inference.common.args import ConfigFileType


def is_torchao_int4(value: object) -> bool:
    """Recognize the serialized TorchAO INT4 layouts used by this package."""
    type_name = f"{type(value).__module__}.{type(value).__name__}"
    return "Int4" in type_name or "AffineQuantizedTensor" in type_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("bf16", "int4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--native-vae-weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--native-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--int4-weights", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    inputs = sorted(args.input_dir.glob("t*.json"))
    expected = int(manifest["sample_count"])
    if len(inputs) != expected:
        raise RuntimeError(f"expected {expected} native inputs, found {len(inputs)}")
    if args.precision == "int4" and (args.int4_weights is None or not args.int4_weights.is_file()):
        raise RuntimeError("--int4-weights is required for INT4 and must exist")
    if not args.native_vae_weights.is_file():
        raise RuntimeError("--native-vae-weights must exist")

    load_info: dict = {}
    original_create = inference_module.OmniInference.create.__func__

    def patched_create(cls, setup_args, /):
        # Local HF checkpoint paths otherwise select the Transformers root
        # config.json.  The public registry name "Cosmos3-Edge" selects this
        # framework YAML; reproduce that behavior explicitly for a local path.
        setup_args.config_file = str(args.config_file.resolve())
        setup_args.config_file_type = ConfigFileType.from_path(setup_args.config_file)
        vae_override = f"model.config.tokenizer.vae_path={args.native_vae_weights.resolve()}"
        bucket_override = "model.config.tokenizer.bucket_name=''"
        setup_args.experiment_overrides = [
            *[
                value
                for value in setup_args.experiment_overrides
                if not value.startswith("model.config.tokenizer.vae_path=")
                and not value.startswith("model.config.tokenizer.bucket_name=")
            ],
            vae_override,
            bucket_override,
        ]
        pipe = original_create(cls, setup_args)
        if args.precision == "int4":
            # Importing torchao registers its quantized tensor subclasses for
            # torch.load. TorchAO 0.17 serializes Int4WeightOnlyConfig with the
            # concrete Int4TilePackedTo4dTensor type, not necessarily the older
            # generic AffineQuantizedTensor name.
            import torchao  # noqa: F401

            started = time.perf_counter()
            state = torch.load(args.int4_weights, map_location="cpu", weights_only=False)
            quantized_values = sum(is_torchao_int4(value) for value in state.values())
            if quantized_values == 0:
                raise RuntimeError("saved INT4 state contains no recognized TorchAO INT4 tensors")
            incompatible = pipe.model.net.language_model.load_state_dict(
                state, strict=False, assign=True
            )
            pipe.model.net.language_model.to("cuda")
            torch.cuda.synchronize()
            loaded_quantized_values = sum(
                is_torchao_int4(value)
                for value in pipe.model.net.language_model.state_dict().values()
            )
            if loaded_quantized_values == 0:
                raise RuntimeError("language model has no recognized TorchAO INT4 tensors after load")
            if loaded_quantized_values != quantized_values:
                raise RuntimeError(
                    "TorchAO INT4 tensor count changed during load: "
                    f"state={quantized_values}, model={loaded_quantized_values}"
                )
            load_info.update(
                int4_weight_file=str(args.int4_weights.resolve()),
                torchao_int4_state_values=quantized_values,
                torchao_int4_loaded_values=loaded_quantized_values,
                missing_keys=list(incompatible.missing_keys),
                unexpected_keys=list(incompatible.unexpected_keys),
                int4_state_load_s=time.perf_counter() - started,
            )
        torch.cuda.reset_peak_memory_stats()
        return pipe

    inference_module.OmniInference.create = classmethod(patched_create)
    sys.argv = [
        "inference.py",
        "--parallelism-preset=latency",
        "-i",
        *(str(path) for path in inputs),
        "-o",
        str(args.native_output_dir),
        "--checkpoint-path",
        str(args.checkpoint),
        "--seed=0",
        "--no-guardrails",
        "--keep-going",
    ]
    from cosmos_framework.scripts.inference import main as inference_main

    started = time.perf_counter()
    inference_main()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started

    console = (args.native_output_dir / "console.log").read_text()
    prefill = [float(value) for value in re.findall(r"prefill time: ([0-9.]+) sec", console)][-expected:]
    decode = [float(value) for value in re.findall(r"decode time: ([0-9.]+) sec", console)][-expected:]
    tokens = [
        int(value)
        for value in re.findall(r"decode time: [0-9.]+ sec, number of tokens: (\d+)", console)
    ][-expected:]
    if not (len(prefill) == len(decode) == len(tokens) == expected):
        raise RuntimeError(f"timing count mismatch: {len(prefill)}, {len(decode)}, {len(tokens)}")

    updates = []
    for idx, sample in enumerate(manifest["samples"]):
        sample_path = args.native_output_dir / f"t{idx:04d}" / "sample_outputs.json"
        result = json.loads(sample_path.read_text())
        exact = result["outputs"][0]["content"]["reasoner_text"]
        updates.append(
            {
                "sample_idx": idx,
                "timestamp_s": sample["timestamp_s"],
                "source_frame_idx": sample["source_frame_idx"],
                "prefill_s": prefill[idx],
                "decode_s": decode[idx],
                "generated_tokens": tokens[idx],
                "decode_ms_per_token": decode[idx] * 1000 / max(tokens[idx], 1),
                "exact_text": exact,
            }
        )

    payload = {
        "schema": "cosmos3edge.native_reasoner_timeline.v1",
        "precision": args.precision,
        "backend": "Cosmos Framework/PyTorch BF16"
        if args.precision == "bf16"
        else "Cosmos Framework/PyTorch + saved TorchAO INT4 weight-only language model",
        "manifest": str(args.manifest.resolve()),
        "video": manifest["video"],
        "video_sha256": manifest["video_sha256"],
        "target_hz": manifest["target_hz"],
        "sample_count": expected,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "wall_s_including_model_load": wall_s,
        "load": load_info,
        "prefill_mean_s": statistics.mean(prefill),
        "prefill_p95_s": float(np.percentile(prefill, 95)),
        "decode_weighted_ms_per_token": sum(decode) * 1000 / sum(tokens),
        "generated_tokens_total": sum(tokens),
        "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "updates": updates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output.with_name(args.output.stem + "_exact_texts.txt").write_text(
        "\n\n".join(
            f"[{item['sample_idx']:03d}] t={item['timestamp_s']:.1f}s frame={item['source_frame_idx']}\n"
            f"{item['exact_text']}"
            for item in updates
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()
