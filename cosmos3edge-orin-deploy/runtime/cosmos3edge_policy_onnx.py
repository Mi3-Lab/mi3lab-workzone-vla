#!/usr/bin/env python3
"""End-to-end joint UMI policy with ONNX Runtime CUDA modules."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.utils import export_to_video, load_image
from torch import nn

from cosmos3edge_forward_onnx import OrtWanCodec
from cosmos3edge_inverse_onnx import bind_input, build_pipeline_shell, load_pipeline_configs, make_session


class OrtPolicyTransformer(nn.Module):
    def __init__(self, model: Path, config) -> None:
        super().__init__()
        self.session = make_session(model)
        self.config = config
        self.dtype = torch.float16
        self.action_dim = config.action_dim
        self.call_count = 0
        self.total_call_s = 0.0
        self.register_parameter("_device_anchor", nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))

    def forward(
        self,
        *,
        input_ids,
        position_ids,
        vision_tokens,
        action_tokens,
        vision_timesteps,
        action_timesteps,
        **kwargs,
    ):
        values = {
            "input_ids": input_ids.long().cuda(),
            "position_ids": position_ids.float().cuda(),
            "vision_latents": vision_tokens[0].half().cuda(),
            "action_latents": action_tokens[0].half().cuda(),
            "vision_timesteps": vision_timesteps.long().cuda(),
            "action_timesteps": action_timesteps.long().cuda(),
        }
        vision = torch.empty_like(values["vision_latents"])
        action = torch.empty_like(values["action_latents"])
        io = self.session.io_binding()
        keep = [bind_input(io, name, value) for name, value in values.items()]
        io.bind_output("vision_velocity", "cuda", 0, onnx.TensorProto.FLOAT16, tuple(vision.shape), vision.data_ptr())
        io.bind_output("action_velocity", "cuda", 0, onnx.TensorProto.FLOAT16, tuple(action.shape), action.data_ptr())
        started = time.perf_counter()
        self.session.run_with_iobinding(io)
        self.total_call_s += time.perf_counter() - started
        self.call_count += 1
        del keep
        result = ([vision.to(vision_tokens[0].dtype)], None, [action.to(action_tokens[0].dtype)])
        return result if not kwargs.get("return_dict", True) else SimpleNamespace(sample=result[0], sound=None, action=result[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ["checkpoint", "image", "actions-spec", "vae-encoder", "vae-decoder", "policy-transformer", "output-video", "output-json"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.actions_spec.read_text())
    started = time.perf_counter()
    transformer_config, vae_config = load_pipeline_configs(args.checkpoint)
    pipe = build_pipeline_shell(
        args.checkpoint,
        OrtPolicyTransformer(args.policy_transformer, transformer_config),
        OrtWanCodec(args.vae_encoder, args.vae_decoder, vae_config),
    )
    load_s = time.perf_counter() - started
    condition = CosmosActionCondition(
        mode="policy",
        chunk_size=spec["action_chunk_size"],
        domain_name=spec["domain_name"],
        resolution_tier=spec["image_size"],
        image=load_image(str(args.image)),
        view_point=spec["view_point"],
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=spec["prompt"],
            action=condition,
            fps=spec["fps"],
            num_inference_steps=30,
            guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(0),
            use_system_prompt=False,
            output_type="pil",
            enable_safety_check=False,
        )
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - started
    actions = result.action[0].float()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.video, str(args.output_video), fps=spec["fps"], macro_block_size=1)
    payload = {
        "backend": "onnxruntime_cuda_fp16",
        "mode": "policy",
        "domain": spec["domain_name"],
        "frames": len(result.video),
        "action_shape": list(actions.shape),
        "actions": actions.tolist(),
        "load_s": load_s,
        "inference_s": inference_s,
        "output_video": str(args.output_video),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "actions"}, indent=2))


if __name__ == "__main__":
    main()
