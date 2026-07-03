"""Teste rápido: 2B classification + trajectory no mesmo modelo.

Verifica:
  1. modelo carrega ok
  2. Q&A funciona (descrição)
  3. sample_trajectories_from_data funciona
  4. pred_xyz tem shape correto e valores razoáveis
  5. projeção para pixels funciona
"""
import os, sys
import torch
import numpy as np
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo1_5 import helper as alp_helper
from safetensors.torch import load_file
from transformers import AutoProcessor
import glob

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"
CKPT      = f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"

# ── 1. Carregar modelo ────────────────────────────────────────────────────────
print("=" * 60)
print("[1] Carregando modelo 2B...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)

sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
if not sd:
    sd = load_file(os.path.join(CKPT, "model.safetensors"), device="cpu")

missing, unexpected = model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()

proc_qa   = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
proc_traj = alp_helper.get_processor(model.tokenizer)

print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Verificar atributos de trajetória
print(f"\n  traj_tokenizer    : {type(getattr(model, 'traj_tokenizer', None)).__name__}")
print(f"  hist_traj_tokenizer: {type(getattr(model, 'hist_traj_tokenizer', None)).__name__}")
print(f"  future_token_start : {getattr(model, 'future_token_start_idx', 'N/A')}")
print(f"  hist_token_start   : {getattr(model, 'hist_token_start_idx', 'N/A')}")

# ── 2. Carregar frame de teste ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[2] Carregando frame de boston.mp4...")
import cv2
cap = cv2.VideoCapture(f"{BASE}/videos/boston.mp4")
ret, frame = cap.read()
cap.release()
assert ret, "Não conseguiu ler o frame"
pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
W, H = pil.size
print(f"  Frame: {W}x{H}")

# ── 3. Teste Q&A ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3] Teste Q&A (descrição)...")
msgs = [{"role": "user", "content": [
    {"type": "image", "image": pil},
    {"type": "text",  "text": "Describe the work zone elements visible in this scene."},
]}]
text   = proc_qa.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc_qa(text=[text], images=[pil], return_tensors="pt", padding=True)
inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.vlm.generate(**inputs, max_new_tokens=80, do_sample=False,
                              pad_token_id=proc_qa.tokenizer.eos_token_id)
n = inputs["input_ids"].shape[1]
desc = proc_qa.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()
print(f"  Descrição: {desc[:120]}")

# ── 4. Teste trajetória ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4] Teste sample_trajectories_from_data...")

frame_np = np.array(pil)
frame_t  = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W) uint8
print(f"  frame_tensor: {frame_t.shape}  dtype={frame_t.dtype}")

messages = alp_helper.create_message(frames=frame_t, camera_indices=None)
inputs_t = proc_traj.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)
print(f"  input_ids shape : {inputs_t['input_ids'].shape}")
print(f"  input_ids min   : {inputs_t['input_ids'].min().item()}")
print(f"  input_ids max   : {inputs_t['input_ids'].max().item()}")
if "pixel_values" in inputs_t:
    print(f"  pixel_values    : {inputs_t['pixel_values'].shape}  dtype={inputs_t['pixel_values'].dtype}")

inputs_t = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs_t.items()}

NUM_HIST = 4
ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
ego_rot = torch.eye(3, dtype=torch.float32, device="cuda").view(1,1,1,3,3).expand(1,1,NUM_HIST,3,3).contiguous()

model_inputs = {
    "tokenized_data": inputs_t,
    "ego_history_xyz": ego_xyz,
    "ego_history_rot": ego_rot,
}

try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot = model.sample_trajectories_from_data(
            data=model_inputs,
            num_traj_samples=1,
            max_generation_length=200,
        )
    print(f"\n  ✓ SUCCESS!")
    print(f"  pred_xyz shape: {pred_xyz.shape}")
    xy = pred_xyz[0, 0, 0, :, :2].cpu().float().numpy()
    print(f"  Primeiros 5 waypoints (x_lat, y_fwd) em metros:")
    for i, (xm, ym) in enumerate(xy[:5]):
        print(f"    [{i}] x={xm:.2f}m  y={ym:.2f}m")

    # Projeção para pixels
    f_px, h_cam = 500.0, 1.5
    cx, cy = W / 2.0, H * 0.5
    pts = []
    for xm, ym in xy:
        if ym > 0.3:
            u = int(cx + f_px * xm / ym)
            v = int(cy + f_px * h_cam / ym)
            if 0 <= u < W and 0 <= v < H:
                pts.append((u, v))
    print(f"\n  Waypoints projetados: {len(pts)} / {len(xy)}")
    if pts:
        print(f"  Primeiro pixel: {pts[0]}")
        print(f"  Último  pixel: {pts[-1]}")

except Exception as e:
    import traceback
    print(f"\n  ✗ FALHOU: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Teste concluído.")
