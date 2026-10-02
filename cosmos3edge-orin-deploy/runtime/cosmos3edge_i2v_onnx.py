#!/usr/bin/env python3
"""Official 121-frame Cosmos3-Edge image-to-video through ONNX Runtime CUDA."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image
from torch import nn

from cosmos3edge_forward_onnx import OrtWanCodec
from cosmos3edge_inverse_onnx import bind_input, build_pipeline_shell, load_pipeline_configs, make_session


class OrtI2VTransformer(nn.Module):
    def __init__(self, conditional: Path, unconditional: Path, config) -> None:
        super().__init__()
        sessions = [make_session(conditional), make_session(unconditional)]
        self.sessions = {}
        for session in sessions:
            shapes = {item.name: tuple(item.shape) for item in session.get_inputs()}
            key = (shapes["input_ids"][0], shapes["position_ids"][-1])
            if key in self.sessions:
                raise RuntimeError(f"conditional and unconditional profiles have the same dispatch key: {key}")
            self.sessions[key] = session
        self.config = config
        self.dtype = torch.float16
        self.action_dim = config.action_dim
        self.register_parameter("_device_anchor", nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))

    def release(self) -> None:
        self.sessions.clear()

    def forward(self, *, input_ids, position_ids, vision_tokens, vision_timesteps, **kwargs):
        key = (input_ids.shape[0], position_ids.shape[-1])
        if key not in self.sessions:
            raise RuntimeError(f"no I2V ONNX profile for text/packed lengths {key}; available={list(self.sessions)}")
        session = self.sessions[key]
        values = {
            "input_ids": input_ids.long().cuda(),
            "position_ids": position_ids.float().cuda(),
            "vision_latents": vision_tokens[0].half().cuda(),
            "vision_timesteps": vision_timesteps.long().cuda(),
        }
        output = torch.empty_like(values["vision_latents"])
        io = session.io_binding()
        keep = [bind_input(io, name, value) for name, value in values.items()]
        io.bind_output("vision_velocity", "cuda", 0, onnx.TensorProto.FLOAT16, tuple(output.shape), output.data_ptr())
        session.run_with_iobinding(io)
        del keep
        result = ([output.to(vision_tokens[0].dtype)], None, None)
        return result if not kwargs.get("return_dict", True) else SimpleNamespace(sample=result[0], sound=None, action=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ["checkpoint", "image", "prompt", "vae-encoder", "vae-decoder", "transformer-cond", "transformer-uncond", "output-video", "output-json"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    transformer_config, vae_config = load_pipeline_configs(args.checkpoint)
    transformer = OrtI2VTransformer(args.transformer_cond, args.transformer_uncond, transformer_config)
    pipe = build_pipeline_shell(
        args.checkpoint,
        transformer,
        OrtWanCodec(
            args.vae_encoder,
            args.vae_decoder,
            vae_config,
            before_decode=transformer.release,
            release_encoder_after_encode=True,
        ),
    )
    load_s = time.perf_counter() - started
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt.read_text(),
            image=load_image(str(args.image)),
            num_frames=121,
            height=480,
            width=832,
            fps=24,
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(0),
            output_type="pil",
            enable_safety_check=False,
        )
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - started
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.video, str(args.output_video), fps=24, macro_block_size=1)
    payload = {
        "backend": "onnxruntime_cuda_fp16",
        "mode": "image2video",
        "frames": len(result.video),
        "height": 480,
        "width": 832,
        "fps": 24,
        "steps": 50,
        "guidance_scale": 5.0,
        "load_s": load_s,
        "inference_s": inference_s,
        "output_video": str(args.output_video),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
