"""Run inference on N val.parquet examples and show model output vs ground truth.

Usage:
    python inference_val.py [checkpoint_dir] [n_samples]
"""
import os, sys, glob, random, textwrap
import torch
import pandas as pd
from PIL import Image

BASE      = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ["WANDB_DISABLED"]        = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

CKPT      = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/checkpoints/sft_stage1_roadwork/checkpoint-18000"
N         = int(sys.argv[2]) if len(sys.argv) > 2 else 10
A1_CKPT   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VLM_PATH  = f"{BASE}/models/Cosmos-Reason2-8B"
DATA_ROOT = f"{BASE}/data/roadwork/lingoqa_roadwork"
OUT_FILE  = f"{BASE}/logs/inference_val_results.txt"

print(f"Checkpoint : {CKPT}")
print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"Samples    : {N}")

from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoProcessor
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=A1_CKPT,
    vlm_name_or_path=VLM_PATH,
)
print("Loading SFT weights...")
shards = sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors")))
sd = {}
for f in shards:
    sd.update(load_file(f, device="cpu"))
missing, _ = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}")
model = model.to(torch.bfloat16).cuda().eval()

# ── Load processor (Qwen3-VL) ─────────────────────────────────────────────────
print("Loading processor...")
processor = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)

# ── Sample from val.parquet ───────────────────────────────────────────────────
print("Loading val.parquet...")
df = pd.read_parquet(f"{DATA_ROOT}/val.parquet")

# Sample diverse question types
random.seed(42)
sampled = df.sample(N, random_state=42)

results = []
print(f"\nRunning inference on {N} samples...\n")

for idx, (_, row) in enumerate(sampled.iterrows()):
    question   = row["question"]
    gt_answer  = row["answer"]
    seg_id     = row["segment_id"]
    img_paths  = row["images"]

    # Load first image
    img_path = os.path.join(DATA_ROOT, img_paths[0])
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  [{idx+1}] Image not found: {img_path} — skipping")
        continue

    # Build chat messages for Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.vlm.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

        # Decode only new tokens
        n_input = inputs["input_ids"].shape[1]
        pred = processor.tokenizer.decode(generated[0][n_input:], skip_special_tokens=True).strip()

    except Exception as e:
        pred = f"[ERRO: {e}]"

    results.append({
        "idx": idx + 1,
        "segment": seg_id,
        "question": question,
        "ground_truth": gt_answer,
        "model_output": pred,
    })

    # Print live
    print(f"[{idx+1}/{N}] {seg_id}")
    print(f"  Q: {question[:120]}")
    print(f"  GT: {gt_answer[:200]}")
    print(f"  MODEL: {pred[:200]}")
    print()

# ── Save to file ──────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    f.write(f"Inference results — {CKPT}\n")
    f.write("=" * 80 + "\n\n")
    for r in results:
        f.write(f"[{r['idx']}] SEGMENT: {r['segment']}\n")
        f.write(f"QUESTION:\n  {r['question']}\n\n")
        f.write(f"GROUND TRUTH:\n")
        for line in textwrap.wrap(r['ground_truth'], 78):
            f.write(f"  {line}\n")
        f.write(f"\nMODEL OUTPUT:\n")
        for line in textwrap.wrap(r['model_output'], 78):
            f.write(f"  {line}\n")
        f.write("\n" + "-" * 80 + "\n\n")

print(f"\nResultados salvos em: {OUT_FILE}")
