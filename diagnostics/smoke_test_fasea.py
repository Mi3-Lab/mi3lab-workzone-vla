"""Smoke test: PAIDataset + nav preprocessor + modelo 2B limpo, 1 amostra.

Valida o pipeline da Fase A antes de gastar horas de GPU em treino real:
carrega o dataset, pega 1 item, roda um forward pass, confirma shapes e loss.
"""
import os, sys
import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo.data.pai import PAIDataset
from alpamayo.processor.qwen_processor import get_preprocess_data_fn_from_model_config, collate_fn_from_model_config
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from safetensors.torch import load_file
import glob

print("[1] Carregando modelo 2B limpo...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=f"{BASE}/checkpoints/student_2b_clean_init",
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-2B",
)
model = model.to(torch.bfloat16).cuda().eval()
print(f"    OK — VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("[2] Instanciando preprocessor nav...")
preprocess_fn = get_preprocess_data_fn_from_model_config(
    model_config=model.config,
    chat_template_version="r1_5",
    components_order=["image", "traj_history", "route", "prompt", "traj_future"],
    components_prompt=["traj_future"],
    label_components=["traj_future"],
    include_camera_ids=True,
    include_frame_nums=True,
    generation_mode=False,
)
print("    OK")

print("[3] Instanciando PAIDataset (chunk 214, 1 amostra)...")
ds = PAIDataset(
    local_dir=f"{BASE}/data/PhysicalAI-AV",
    chunk_ids=[214],
    reasoning_metadata="reasoning/ood_reasoning.parquet",
    vla_preprocess_args={
        "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        "chat_template_version": "r1_5",
        "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "label_components": ["traj_future"],
        "include_camera_ids": True,
        "include_frame_nums": True,
        "generation_mode": False,
    },
    model_config=model.config,
)
print(f"    OK — {len(ds)} clipes no chunk 214")

print("[4] Pegando 1 amostra...")
sample = ds[0]
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")
    elif isinstance(v, dict):
        print(f"    {k}: dict com chaves {list(v.keys())}")
    else:
        print(f"    {k}: {type(v).__name__} = {str(v)[:60]}")

print("[5] Montando batch via collate_fn...")
collate_fn = collate_fn_from_model_config(chat_template_version="r1_5")
batch = collate_fn([sample])
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

print("[6] Forward pass...")
batch_cuda = {}
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch_cuda[k] = v.cuda()
    elif isinstance(v, dict):
        batch_cuda[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv) for kk, vv in v.items()}
    else:
        batch_cuda[k] = v

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(**batch_cuda)
print(f"    loss = {out.loss.item():.4f}")
print(f"    logits shape = {tuple(out.logits.shape)}")

print("\n=== SMOKE TEST PASSOU ===")
