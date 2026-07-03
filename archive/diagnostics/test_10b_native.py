"""Alpamayo-10B BASE no formato NATIVO — como ele gera antes de work zone.

Curiosidade do usuário + validação da Fase A (KD geral):
  - clipe real PhysicalAI-AV (4 câmeras, ego history real) já baixado em /data
  - 10B base intocado (A1), sample_trajectories_from_data_with_vlm_rollout
  - imprime o Chain-of-Causation e o minADE vs trajetória GT do clipe

Se o CoT for rico e o minADE < 1m, é este comportamento que o KD geral
transfere ao 2B — e prova que a trajetória ruim da demo é problema de
formato de entrada, não de capacidade.
"""
import os, sys
import numpy as np
import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

CLIP = f"{BASE}/data/aiav_samples/030c760c-ae38-49aa-9ad8-f5650a545d26.pt"

print("[LOAD] clipe AIAV nativo...")
data = torch.load(CLIP, weights_only=False)
print(f"  image_frames: {tuple(data['image_frames'].shape)}")

print("[LOAD] Alpamayo-1.5-10B BASE (A1, pesos intocados)...")
model = Alpamayo1_5.from_pretrained(
    f"{BASE}/models/Alpamayo-1.5-10B-A1-format", dtype=torch.bfloat16).to("cuda")
model = model.eval()
processor = helper.get_processor(model.tokenizer)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

messages = helper.create_message(
    frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"])
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt")
model_inputs = helper.to_device({
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=3, max_generation_length=256, return_extra=True)

print("\n" + "=" * 76)
print("CHAIN-OF-CAUSATION (raciocínio nativo do 10B, sem work zone):")
print("=" * 76)
for i, cot in enumerate(extra["cot"]):
    print(f"\n--- amostra {i+1} ---\n{cot}")

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().float().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
print("\n" + "=" * 76)
print(f"TRAJETÓRIA vs GT do clipe:")
print(f"  ADE por amostra: {[f'{d:.2f}m' for d in diff]}")
print(f"  minADE: {diff.min():.2f} m  (esperado < 1.0 m)")
xyz0 = pred_xyz.cpu().float().numpy()[0, 0, 0]
print(f"  waypoint final (amostra 1): x_fwd={xyz0[-1,0]:+.1f}m  y_lat={xyz0[-1,1]:+.1f}m")
print("\n=== DONE ===")
