"""Diagnostic: isolate the CUDA assert in 10B trajectory call.
Run with CUDA_LAUNCH_BLOCKING=1 for full synchronous traceback.
"""
import os, sys
import torch
import numpy as np

BASE      = "/data/wesleyferreiramaia/wokzone-alpamayo"
A10B_PATH = f"{BASE}/models/Alpamayo-1.5-10B"

sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))

print("Loading Alpamayo1_5 10B...")
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper as alp_helper

model = Alpamayo1_5.from_pretrained(A10B_PATH, dtype=torch.bfloat16).cuda().eval()
proc  = alp_helper.get_processor(model.tokenizer)

# ── Vocab diagnostics ──────────────────────────────────────────────────────────
print(f"\n[VOCAB]")
print(f"  tokenizer.vocab_size           : {model.tokenizer.vocab_size}")
print(f"  len(tokenizer)                 : {len(model.tokenizer)}")
embed = model.vlm.model.embed_tokens
print(f"  vlm embed_tokens.weight.shape  : {embed.weight.shape}")
print(f"  embed num_embeddings           : {embed.num_embeddings}")

# ── Build dummy frame (solid grey 722x406 uint8) ──────────────────────────────
frame_tensor = torch.full((1, 3, 406, 722), 128, dtype=torch.uint8)
print(f"\n[INPUT] frame_tensor: {frame_tensor.shape}  dtype={frame_tensor.dtype}")

messages = alp_helper.create_message(frames=frame_tensor, camera_indices=None)

print("[PROC] apply_chat_template ...")
inputs = proc.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)

ids = inputs["input_ids"]
print(f"  input_ids shape : {ids.shape}")
print(f"  input_ids min   : {ids.min().item()}")
print(f"  input_ids max   : {ids.max().item()}")
print(f"  embed table size: {embed.num_embeddings}")
print(f"  any OOB?        : {(ids >= embed.num_embeddings).any().item()}")

if "pixel_values" in inputs:
    pv = inputs["pixel_values"]
    print(f"  pixel_values    : {pv.shape}  dtype={pv.dtype}")

# ── Attempt fuse_traj_tokens alone ────────────────────────────────────────────
inputs_cuda = alp_helper.to_device(inputs, "cuda")
NUM_HIST = 4
ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
ego_rot = (torch.eye(3, dtype=torch.float32, device="cuda")
           .view(1, 1, 1, 3, 3).expand(1, 1, NUM_HIST, 3, 3).contiguous())

print("\n[TEST] fuse_traj_tokens ...")
try:
    import copy
    ids_fused = model.fuse_traj_tokens(
        inputs_cuda["input_ids"].clone(),
        {"ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot},
    )
    print(f"  fuse_traj_tokens OK — ids_fused: {ids_fused.shape}")
    print(f"  fused min={ids_fused.min().item()}  max={ids_fused.max().item()}")
except Exception as e:
    import traceback
    print(f"  fuse_traj_tokens FAILED: {e}")
    traceback.print_exc()

# ── Full trajectory call ───────────────────────────────────────────────────────
model_inputs = {
    "tokenized_data": alp_helper.to_device(inputs, "cuda"),
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

print("\n[TEST] sample_trajectories_from_data_with_vlm_rollout ...")
try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            num_traj_samples=1,
            max_generation_length=50,
        )
    print(f"SUCCESS! pred_xyz: {pred_xyz.shape}")
    print(f"  first 5 waypoints xy: {pred_xyz[0,0,0,:5,:2].cpu().float()}")
except Exception as e:
    import traceback
    print(f"\nFAILED: {e}")
    traceback.print_exc()
