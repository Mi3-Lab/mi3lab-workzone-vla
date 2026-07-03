"""Teste isolado: 2B trajectory com NUM_HIST=16 (correção do masked_scatter).

DeltaTrajectoryTokenizer.encode() retorna T×3 tokens.
tokens_per_history_traj=48 → T=16 (não 4).
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
proc_traj = alp_helper.get_processor(model.tokenizer)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Frame de teste
cap = cv2.VideoCapture(f"{BASE}/videos/boston.mp4")
ret, frame = cap.read(); cap.release()
pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
W, H = pil.size
frame_np = np.array(pil)
frame_t  = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()  # (1,3,H,W) uint8
print(f"[FRAME] {W}x{H}")

# NUM_HIST=16: DeltaTrajectoryTokenizer produz 16×3=48 tokens ✓
# Forward ego history: v=8 m/s → estimate_t0_states extrai v0≠0 → trajetória frente
NUM_HIST = 16
_dt, _v = 0.1, 8.0
_t = torch.arange(NUM_HIST, dtype=torch.float32) * _dt
ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
ego_xyz[0, 0, :, 0] = _t.cuda() * _v   # x=forward: [0, 0.8, ..., 12.0] m
ego_rot = torch.eye(3, dtype=torch.float32, device="cuda").view(1,1,1,3,3).expand(1,1,NUM_HIST,3,3).contiguous()
print(f"[EGO]  ego_xyz: {ego_xyz.shape}  x_fwd_last={ego_xyz[0,0,-1,0].item():.1f}m")

# Tokenizar mensagem de trajetória
messages  = alp_helper.create_message(frames=frame_t, camera_indices=None)
inputs_t  = proc_traj.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)
hist_pad_id = model.config.traj_token_ids.get("history")
ids = inputs_t["input_ids"]
n_hist_tokens = (ids == hist_pad_id).sum().item()
print(f"[TOKS] input_ids {ids.shape}  max={ids.max().item()}  <|traj_history|>×{n_hist_tokens} (esperado 48)")

inputs_t = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs_t.items()}
model_inputs = {
    "tokenized_data": inputs_t,
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

print("\n[TRAJ] Chamando sample_trajectories_from_data...")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot = model.sample_trajectories_from_data(
        data=model_inputs, num_traj_samples=1, max_generation_length=200,
    )

print(f"\n  ✓ SUCCESS! pred_xyz: {pred_xyz.shape}")
xyz = pred_xyz[0, 0, 0, :, :].cpu().float().numpy()
print(f"  Primeiros 5 waypoints (x_fwd, y_lat, z) m:")
for i, row in enumerate(xyz[:5]):
    print(f"    [{i}] x_fwd={row[0]:+.2f}  y_lat={row[1]:+.2f}  z={row[2]:+.2f}")
print(f"  Último waypoint: x_fwd={xyz[-1,0]:+.2f}  y_lat={xyz[-1,1]:+.2f}")

# Projeção corrigida: UnicycleAccelCurvature: x=forward, y=lateral(left+)
f_px, h_cam, cx, cy = 500.0, 1.5, W/2.0, H*0.5
pts = []
for row in xyz:
    x_fwd, y_lat = float(row[0]), float(row[1])
    if x_fwd < 0.3:
        continue
    u = int(cx - f_px * y_lat / x_fwd)
    v = int(cy + f_px * h_cam / x_fwd)
    if 0 <= u < W and 0 <= v < H:
        pts.append((u, v))
print(f"\n  Waypoints projetados: {len(pts)} / {len(xyz)}")
if pts:
    print(f"  Início: {pts[0]}   Fim: {pts[-1]}")

# Salvar frame com trajetória sobreposta
frame_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
for j in range(1, len(pts)):
    cv2.line(frame_out, pts[j-1], pts[j], (0, 255, 0), 2)
for p in pts[::3]:
    cv2.circle(frame_out, p, 4, (0, 200, 255), -1)
out_path = f"{BASE}/alpamayo-recipes/recipes/alpamayo1_5_sft/traj_test_out.jpg"
cv2.imwrite(out_path, frame_out)
print(f"\n  Frame salvo: {out_path}")
print("\n=== DONE ===")
