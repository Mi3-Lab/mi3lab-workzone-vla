"""Compute eval_loss on val.parquet for a saved checkpoint (no Trainer/Hydra)."""

import os, sys, glob, math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CKPT      = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/checkpoints/sft_stage1_roadwork/checkpoint-18000"
A1_CKPT   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VLM_PATH  = f"{BASE}/models/Cosmos-Reason2-8B"
DATA_ROOT = f"{BASE}/data/roadwork/lingoqa_roadwork"

print(f"Checkpoint : {CKPT}")
print(f"GPU        : {torch.cuda.get_device_name(0)}")

from omegaconf import OmegaConf
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo.data.lingoqa import LingoQADataset
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
from safetensors.torch import load_file
from functools import partial

# ── Load base model ──────────────────────────────────────────────────────────
print("Loading base model...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=A1_CKPT,
    vlm_name_or_path=VLM_PATH,
)

# ── Load SFT weights ─────────────────────────────────────────────────────────
print("Loading SFT weights...")
shards = sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors")))
state_dict = {}
for f in shards:
    state_dict.update(load_file(f, device="cpu"))
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

model = model.to(torch.bfloat16).cuda().eval()

# ── Resolve config (DictConfig → plain object) ───────────────────────────────
raw_cfg = OmegaConf.to_container(model.config, resolve=True) if hasattr(model.config, '_metadata') else None

class PlainConfig:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

model_config = PlainConfig(raw_cfg) if raw_cfg is not None else model.config

# ── Val dataset ───────────────────────────────────────────────────────────────
VLA_PREPROCESS_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "question", "answer"],
    "components_prompt": ["answer"],
    "label_components": ["answer"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": False,
}

print("Loading val dataset...")
val_dataset = LingoQADataset(
    data_root=DATA_ROOT,
    parquet_name="val.parquet",
    model_config=model_config,
    vla_preprocess_args=VLA_PREPROCESS_ARGS,
)
print(f"  {len(val_dataset)} examples")

collate_fn = partial(collate_fn_from_model_config, model_config=model_config, chat_template_version="r1_5")
loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    return x

# ── Eval loop ─────────────────────────────────────────────────────────────────
print("Running eval loop...")
total_loss = 0.0
n_batches = 0

with torch.no_grad():
    for batch in tqdm(loader, total=len(loader)):
        batch = to_cuda(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        loss = out.loss
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            n_batches += 1

avg_loss = total_loss / max(n_batches, 1)
perplexity = math.exp(avg_loss)

print(f"\n=== EVAL RESULTS (checkpoint-18000) ===")
print(f"  eval_loss   : {avg_loss:.4f}")
print(f"  perplexity  : {perplexity:.2f}")
print(f"  batches     : {n_batches}/{len(loader)}")
print(f"========================================")
