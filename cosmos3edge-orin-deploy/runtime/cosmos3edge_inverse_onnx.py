#!/usr/bin/env python3
"""End-to-end Cosmos3-Edge AV inverse dynamics using ONNX CUDA engines.

Diffusers remains the host orchestrator for media preprocessing, packing and
UniPC state. The two heavyweight calls (Wan encoder and 30 MoT steps) execute
through ONNX Runtime; the same interfaces can be backed by TensorRT on Orin.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import onnx
import onnxruntime as ort
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition, UniPCMultistepScheduler
from diffusers.utils import load_video
from torch import nn
from transformers import PreTrainedTokenizerFast


ONNX_TYPES = {
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.int64: onnx.TensorProto.INT64,
}


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    use_tensorrt = os.environ.get("COSMOS3EDGE_ORT_TENSORRT", "0") == "1"
    providers: list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if use_tensorrt:
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "COSMOS3EDGE_ORT_TENSORRT=1 but this ONNX Runtime build has no TensorrtExecutionProvider"
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
                    "trt_detailed_build_log": True,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=providers,
    )
    expected = "TensorrtExecutionProvider" if use_tensorrt else "CUDAExecutionProvider"
    if session.get_providers()[0] != expected:
        raise RuntimeError(f"{expected} not active for {path}: {session.get_providers()}")
    return session


def bind_input(binding: ort.IOBinding, name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.contiguous()
    binding.bind_input(name, "cuda", 0, ONNX_TYPES[value.dtype], tuple(value.shape), value.data_ptr())
    return value


def load_pipeline_configs(checkpoint: Path):
    """Read configs only; no BF16 transformer or VAE weights are loaded."""
    transformer_config = SimpleNamespace(
        **json.loads((checkpoint / "transformer" / "config.json").read_text())
    )
    vae_config = SimpleNamespace(**json.loads((checkpoint / "vae" / "config.json").read_text()))
    return transformer_config, vae_config


def build_pipeline_shell(checkpoint: Path, transformer: nn.Module, vae: nn.Module) -> Cosmos3OmniPipeline:
    """Construct the host pipeline directly around ONNX-backed modules."""
    model_index = json.loads((checkpoint / "model_index.json").read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint / "text_tokenizer", local_files_only=True)
    scheduler = UniPCMultistepScheduler.from_pretrained(checkpoint / "scheduler", local_files_only=True)
    return Cosmos3OmniPipeline(
        transformer=transformer,
        text_tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        sound_tokenizer=None,
        safety_checker=None,
        enable_safety_checker=False,
        default_use_system_prompt=bool(model_index.get("default_use_system_prompt", False)),
        use_native_flow_schedule=bool(model_index.get("use_native_flow_schedule", True)),
    )


class _DeterministicDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class OrtWanEncoder(nn.Module):
    def __init__(self, model: Path, config) -> None:
        super().__init__()
        self.session = make_session(model)
        self.config = config
        self.dtype = torch.float16
        self.call_count = 0
        self.total_call_s = 0.0
        self.register_parameter("_device_anchor", nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))
        self.register_buffer("mean", torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(config.latents_std).view(1, -1, 1, 1, 1), persistent=False)

    def encode(self, video: torch.Tensor, return_dict: bool = True):
        video = video.half().cuda().contiguous()
        out_shape = (video.shape[0], 48, (video.shape[2] - 1) // 4 + 1, video.shape[3] // 16, video.shape[4] // 16)
        normalized = torch.empty(out_shape, dtype=torch.float16, device="cuda")
        binding = self.session.io_binding()
        keep = bind_input(binding, "video_rgb_minus1_to1", video)
        binding.bind_output(
            "normalized_latents", "cuda", 0, onnx.TensorProto.FLOAT16, out_shape, normalized.data_ptr()
        )
        started = time.perf_counter()
        self.session.run_with_iobinding(binding)
        self.total_call_s += time.perf_counter() - started
        self.call_count += 1
        del keep
        raw = normalized * self.std.to(normalized) + self.mean.to(normalized)
        result = SimpleNamespace(latent_dist=_DeterministicDistribution(raw))
        return result if return_dict else (result.latent_dist,)


class OrtInverseTransformer(nn.Module):
    def __init__(self, model: Path, config) -> None:
        super().__init__()
        self.session = make_session(model)
        self.config = config
        self.dtype = torch.float16
        self.action_dim = config.action_dim
        self.register_parameter("_device_anchor", nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))

    def forward(self, *, input_ids, position_ids, vision_tokens, action_tokens, action_timesteps, **kwargs):
        values = {
            "input_ids": input_ids.to(device="cuda", dtype=torch.int64),
            "position_ids": position_ids.to(device="cuda", dtype=torch.float32),
            "vision_latents": vision_tokens[0].to(device="cuda", dtype=torch.float16),
            "action_latents": action_tokens[0].to(device="cuda", dtype=torch.float16),
            "action_timesteps": action_timesteps.to(device="cuda", dtype=torch.int64),
        }
        output = torch.empty((60, 64), dtype=torch.float16, device="cuda")
        binding = self.session.io_binding()
        keepalive = [bind_input(binding, name, value) for name, value in values.items()]
        binding.bind_output(
            "action_velocity", "cuda", 0, onnx.TensorProto.FLOAT16, tuple(output.shape), output.data_ptr()
        )
        self.session.run_with_iobinding(binding)
        del keepalive
        vision_velocity = [torch.zeros_like(vision_tokens[0])]
        result = (vision_velocity, None, [output.to(action_tokens[0].dtype)])
        if kwargs.get("return_dict", True):
            return SimpleNamespace(sample=result[0], sound=None, action=result[2])
        return result


def build_vae_encoder(path: Path, vae_config):
    """Escolhe entre o encoder monolítico e o por chunk.

    Se `path` for um DIRETÓRIO contendo `encoder-chunk0-fp16.onnx`, usa o
    `ChunkedWanEncoder`; caso contrário mantém o `OrtWanEncoder` monolítico.
    Também aceita forçar via `COSMOS3EDGE_VAE_CHUNKED=1|0`.

    Por que o modo por chunk existe: o grafo monolítico traz as 16 iterações
    do loop causal do VAE desenroladas (495 Conv, 44.366 nós) e exige
    ~125,8 GB de RAM do host para construir um engine TensorRT -- não cabe
    nos 64 GB do AGX Orin, e engines são travados ao hardware (têm que ser
    construídos no alvo). Os grafos por chunk constroem com <4 GB.
    A semântica de janela é idêntica (o cache é resetado a cada `encode()`).
    """
    forced = os.environ.get("COSMOS3EDGE_VAE_CHUNKED")
    is_chunk_dir = path.is_dir() and (path / "encoder-chunk0-fp16.onnx").exists()
    use_chunked = is_chunk_dir if forced is None else forced == "1"

    if not use_chunked:
        return OrtWanEncoder(path, vae_config)

    if not is_chunk_dir:
        raise RuntimeError(
            f"COSMOS3EDGE_VAE_CHUNKED=1 mas {path} não tem encoder-chunk0-fp16.onnx"
        )
    from cosmos3edge_vae_chunked import ChunkedWanEncoder

    print(f"[vae] modo por chunk (engines constroem com <4GB): {path}", flush=True)
    return ChunkedWanEncoder(path, vae_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vae-encoder", type=Path, required=True)
    parser.add_argument("--inverse-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_start = time.perf_counter()
    transformer_config, vae_config = load_pipeline_configs(args.checkpoint)
    pipe = build_pipeline_shell(
        args.checkpoint,
        OrtInverseTransformer(args.inverse_transformer, transformer_config),
        build_vae_encoder(args.vae_encoder, vae_config),
    )
    load_s = time.perf_counter() - load_start

    frames = load_video(str(args.video))
    condition = CosmosActionCondition(
        mode="inverse_dynamics",
        chunk_size=60,
        domain_name="av",
        resolution_tier=480,
        video=frames,
        view_point="ego_view",
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt="You are an autonomous vehicle planning system.",
            action=condition,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=generator,
            output_type="latent",
            enable_safety_check=False,
        )
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - start
    action = result.action[0].float()
    payload = {
        "backend": "onnxruntime_cuda_fp16",
        "video": str(args.video),
        "shape": list(action.shape),
        "seed": args.seed,
        "steps": args.steps,
        "load_s": load_s,
        "inference_s": inference_s,
        "all_finite": bool(torch.isfinite(action).all()),
        "action_60x9": action.tolist(),
    }
    if args.reference:
        ref_payload = json.loads(args.reference.read_text())
        ref = torch.tensor(ref_payload["action_60x9"], dtype=torch.float32)
        error = (action - ref).abs()
        payload.update(
            reference=str(args.reference),
            reference_mae=float(error.mean()),
            reference_max_abs_error=float(error.max()),
            reference_cosine_similarity=float(
                torch.nn.functional.cosine_similarity(action.flatten(), ref.flatten(), dim=0)
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "action_60x9"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
