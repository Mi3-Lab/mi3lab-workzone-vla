#!/bin/bash
#SBATCH --job-name=alp_eval_v4
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:25:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/eval_v4_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/eval_v4_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
SFT_DIR=$BASE/alpamayo-recipes/recipes/alpamayo1_5_sft

source $SFT_DIR/.venv/bin/activate
export HF_HOME=$BASE/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Recebe checkpoint como argumento (ex: checkpoint-500)
CKPT_NAME=${1:-"checkpoint-500"}
CKPT_PATH=$BASE/checkpoints/sft_stage4_v4/$CKPT_NAME

cd $SFT_DIR
python - "$CKPT_PATH" << 'PYEOF'
import sys, os
sys.path.insert(0, '/data/wesleyferreiramaia/wokzone-alpamayo/alpamayo1.5/src')
os.environ.update({"WANDB_DISABLED":"true","TOKENIZERS_PARALLELISM":"false",
                    "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
import torch
from functools import partial
from safetensors.torch import load_file
from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo.processor.qwen_processor import QwenProcessor, collate_fn_from_model_config
from alpamayo.data.lingoqa import LingoQADataset

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
A1_BASE = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_PATH = f"{BASE}/models/Cosmos-Reason2-2B"
CKPT_PATH = sys.argv[1]
DATA_ROOT = f"{BASE}/data/roadwork/lingoqa_roadwork"

print("=" * 60)
print(f"Alpamayo 2B v4 — eval geração")
print(f"Checkpoint: {CKPT_PATH}")
print("=" * 60)

print("\n[1] Carregando modelo...")
student = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_PATH)
sd = load_file(f"{CKPT_PATH}/model.safetensors", device="cpu")
missing, unexpected = student.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
student = student.to(torch.bfloat16).cuda().eval()
model_config = student.config

# Sanity check: embed_tokens deve ter norma ~539 (Cosmos-2B)
et_norm = student.vlm.model.language_model.embed_tokens.weight.float().norm().item()
print(f"  embed_tokens norm: {et_norm:.2f}  (esperado ~539 se pesos Cosmos intactos)")

print("\n[2] Construindo pipeline...")
qwen_proc = QwenProcessor(
    vlm_name_or_path=model_config.vlm_name_or_path,
    traj_vocab_size=model_config.traj_vocab_size,
    min_pixels=model_config.min_pixels,
    max_pixels=model_config.max_pixels,
    include_camera_ids=True, include_frame_nums=True,
    chat_template_version="r1_5",
)
processor = qwen_proc.processor

vla_args = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "question", "answer"],
    "components_prompt": ["answer"],
    "label_components": ["answer"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": True,
}
dataset = LingoQADataset(data_root=DATA_ROOT, parquet_name="val.parquet",
                          model_config=model_config, vla_preprocess_args=vla_args)
collate = partial(collate_fn_from_model_config, model_config=model_config,
                   chat_template_version="r1_5", include_camera_ids=True, include_frame_nums=True)

print(f"  Dataset val: {len(dataset)} amostras")

print("\n[3] Gerando respostas (5 amostras)...")
for i in range(5):
    sample = dataset[i]
    batch = collate([sample])
    inputs = {k: v.cuda() for k, v in batch["tokenized_data"].items() if isinstance(v, torch.Tensor)}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = student.vlm.generate(
            **inputs, max_new_tokens=200,
            do_sample=True, temperature=0.3, top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    n_in = inputs["input_ids"].shape[1]
    pred = processor.tokenizer.decode(gen[0][n_in:], skip_special_tokens=True).strip()

    print(f"\n--- Amostra {i+1} ---")
    print(f"Q:  {sample.get('question','?')[:120]}")
    print(f"GT: {sample.get('answer','')[:200]}")
    print(f"2B: {pred[:200]}")
    print()

print("=" * 60)
print("EVAL DONE")
PYEOF
