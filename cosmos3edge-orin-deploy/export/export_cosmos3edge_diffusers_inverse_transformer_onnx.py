#!/usr/bin/env python3
"""Export the current official Diffusers Cosmos3-Edge inverse denoising step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import Cosmos3OmniTransformer
from torch import nn
from torch.nn import functional as F
from diffusers.models.transformers.transformer_cosmos3 import _rotate_half


class ExportableCosmos3AttnProcessor:
    """Cosmos attention with explicit KV-head expansion for ONNX export."""

    @staticmethod
    def attend(query, key, value, is_causal):
        groups = query.shape[2] // key.shape[2]
        key = key.unsqueeze(3).expand(-1, -1, -1, groups, -1).flatten(2, 3)
        value = value.unsqueeze(3).expand(-1, -1, -1, groups, -1).flatten(2, 3)
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=is_causal,
        ).transpose(1, 2)

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_und, k_und = attn.norm_q(q_und), attn.norm_k(k_und)
        k_und_for_gen = attn.k_norm_und_for_gen(k_und) if attn.k_norm_und_for_gen is not None else k_und
        q_gen, k_gen = attn.norm_added_q(q_gen), attn.norm_added_k(k_gen)
        cos_und, sin_und, cos_gen, sin_gen = [item.unsqueeze(1) for item in rotary_emb]
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen
        causal = self.attend(q_und.unsqueeze(0), k_und.unsqueeze(0), v_und.unsqueeze(0), True)
        all_k = torch.cat([k_und_for_gen, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full = self.attend(q_gen.unsqueeze(0), all_k.unsqueeze(0), all_v.unsqueeze(0), False)
        return attn.to_out(causal.squeeze(0).flatten(-2, -1)), attn.to_add_out(full.squeeze(0).flatten(-2, -1))


class InverseAVStep(nn.Module):
    """Fixed AV-61 packing profile with tensor-only per-step inputs."""

    def __init__(self, transformer: Cosmos3OmniTransformer, fixture: dict) -> None:
        super().__init__()
        self.transformer = transformer
        self.kwargs = fixture["kwargs"]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_latents: torch.Tensor,
        action_latents: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        k = self.kwargs
        outputs = self.transformer(
            input_ids=input_ids,
            text_indexes=k["text_indexes"],
            position_ids=position_ids,
            und_len=k["und_len"],
            sequence_length=k["sequence_length"],
            vision_tokens=[vision_latents],
            vision_token_shapes=k["vision_token_shapes"],
            vision_sequence_indexes=k["vision_sequence_indexes"],
            vision_mse_loss_indexes=k["vision_mse_loss_indexes"],
            vision_timesteps=k["vision_timesteps"],
            vision_noisy_frame_indexes=k["vision_noisy_frame_indexes"],
            action_tokens=[action_latents],
            action_token_shapes=k["action_token_shapes"],
            action_sequence_indexes=k["action_sequence_indexes"],
            action_mse_loss_indexes=k["action_mse_loss_indexes"],
            action_timesteps=action_timesteps,
            action_noisy_frame_indexes=k["action_noisy_frame_indexes"],
            action_domain_ids=k["action_domain_ids"],
            return_dict=False,
        )
        return outputs[2][0]


def move_constants(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_constants(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_constants(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_constants(item, device) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    fixture = move_constants(fixture, "cuda")
    transformer = Cosmos3OmniTransformer.from_pretrained(
        args.checkpoint,
        subfolder="transformer",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device="cuda", dtype=torch.float16).eval()
    for layer in transformer.layers:
        layer.self_attn.set_processor(ExportableCosmos3AttnProcessor())
    wrapper = InverseAVStep(transformer, fixture).cuda().eval()
    k = fixture["kwargs"]
    inputs = (
        k["input_ids"],
        k["position_ids"],
        k["vision_tokens"][0].half(),
        k["action_tokens"][0].half(),
        k["action_timesteps"],
    )
    with torch.inference_mode():
        actual = wrapper(*inputs)
    expected = fixture["outputs"][2][0].float()
    error = (actual.float() - expected).abs()
    validation = {
        "source": "official Diffusers pinned Cosmos3-Edge implementation",
        "precision": "FP16",
        "profile": "inverse_dynamics AV, 61 RGB frames -> 60x9 action",
        "input_shapes": {
            "input_ids": list(inputs[0].shape),
            "position_ids": list(inputs[1].shape),
            "vision_latents": list(inputs[2].shape),
            "action_latents": list(inputs[3].shape),
            "action_timesteps": list(inputs[4].shape),
        },
        "output_shape": list(actual.shape),
        "mae_fp16_vs_captured_bf16": float(error.mean()),
        "max_abs_error_fp16_vs_captured_bf16": float(error.max()),
        "cosine_similarity_fp16_vs_captured_bf16": float(
            torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.flatten(), dim=0)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)

    torch.onnx.export(
        wrapper,
        inputs,
        str(args.output_dir / "model.onnx"),
        input_names=["input_ids", "position_ids", "vision_latents", "action_latents", "action_timesteps"],
        output_names=["action_velocity"],
        opset_version=20,
        do_constant_folding=False,
        external_data=True,
        dynamo=False,
    )
    print(f"Exported {args.output_dir / 'model.onnx'}", flush=True)


if __name__ == "__main__":
    main()
