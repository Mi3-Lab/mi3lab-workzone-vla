"""Diagnóstico aprofundado + testa 3 abordagens para evitar o masked_scatter crash.

Abordagem A: proc_qa com tokenizer substituído (Cosmos image pipeline + Alpamayo tokenizer)
Abordagem B: ego_history=None (skip history fusion, sem masked_scatter)
Abordagem C: usar proc_traj original mas verificar colisão de IDs
"""
import os, sys
import torch
import numpy as np
from PIL import Image
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
    "CUDA_LAUNCH_BLOCKING": "1",
})

from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo1_5 import helper as alp_helper
from safetensors.torch import load_file
from transformers import AutoProcessor
import glob

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"
CKPT      = f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"

print("[LOAD] Carregando modelo 2B...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()

# Diagnosticar IDs de tokens especiais
print("\n[TOKENS] IDs de tokens especiais no model.tokenizer:")
tok = model.tokenizer
special = ["<|traj_history|>", "<|traj_future|>", "<|image_pad|>",
           "<|traj_history_start|>", "<|cot_start|>"]
for s in special:
    ids = tok.encode(s, add_special_tokens=False)
    print(f"  {s!r:35s} → {ids}")

print(f"\n  traj_token_ids (do config): {getattr(model.config, 'traj_token_ids', 'N/A')}")
hist_pad_id = getattr(model.config, 'traj_token_ids', {}).get('history', None)
print(f"  history pad ID: {hist_pad_id}")

# Frame de teste
cap = cv2.VideoCapture(f"{BASE}/videos/boston.mp4")
ret, frame = cap.read(); cap.release()
pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
W, H = pil.size
frame_np = np.array(pil)
frame_t  = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W) uint8

NUM_HIST = 4
ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
ego_rot = torch.eye(3, dtype=torch.float32).view(1,1,1,3,3).expand(1,1,NUM_HIST,3,3).contiguous().cuda()

def try_traj(label, proc, ego_xyz_arg, ego_rot_arg):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    try:
        messages = alp_helper.create_message(frames=frame_t, camera_indices=None)
        inputs_t = proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        ids = inputs_t["input_ids"]
        print(f"  input_ids: {ids.shape}  min={ids.min().item()}  max={ids.max().item()}")

        if hist_pad_id is not None:
            n_hist = (ids == hist_pad_id).sum().item()
            print(f"  tokens com ID={hist_pad_id} (<|traj_history|>): {n_hist}  (esperado 48)")

        inputs_t = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs_t.items()}
        model_inputs = {
            "tokenized_data": inputs_t,
            "ego_history_xyz": ego_xyz_arg,
            "ego_history_rot": ego_rot_arg,
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = model.sample_trajectories_from_data(
                data=model_inputs, num_traj_samples=1, max_generation_length=200,
            )
        xy = pred_xyz[0, 0, 0, :, :2].cpu().float().numpy()
        print(f"  ✓ SUCCESS! pred_xyz: {pred_xyz.shape}")
        print(f"  Primeiros waypoints (x_lat, y_fwd) m: {xy[:3].tolist()}")

        # Projeção pixels
        pts = []
        for xm, ym in xy:
            if ym > 0.3:
                u = int(W/2 + 500*xm/ym)
                v = int(H/2 + 500*1.5/ym)
                if 0 <= u < W and 0 <= v < H:
                    pts.append((u, v))
        print(f"  Waypoints projetados: {len(pts)} / {len(xy)}")
        return True
    except Exception as e:
        import traceback
        print(f"  ✗ FALHOU: {e}")
        traceback.print_exc()
        return False

# Abordagem A: proc_qa (Cosmos-2B) + model.tokenizer
print("\n>>> Abordagem A: Cosmos-2B processor + Alpamayo tokenizer")
proc_a = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
proc_a.tokenizer = model.tokenizer  # adicionar tokens especiais Alpamayo
ok_a = try_traj("A: cosmos_proc + alpamayo_tokenizer", proc_a, ego_xyz, ego_rot)

# Abordagem B: proc_traj original, ego_history=None (skip fuse_traj_tokens)
print("\n>>> Abordagem B: alp_helper.get_processor + ego_history=None")
proc_b = alp_helper.get_processor(model.tokenizer)
ok_b = try_traj("B: qwen3vl2b_proc + ego=None", proc_b, None, None)

# Abordagem C: proc_a + ego_history=None
print("\n>>> Abordagem C: Cosmos-2B processor + ego_history=None")
ok_c = try_traj("C: cosmos_proc + ego=None", proc_a, None, None)

print("\n" + "="*60)
print(f"Resumo: A={ok_a}  B={ok_b}  C={ok_c}")
