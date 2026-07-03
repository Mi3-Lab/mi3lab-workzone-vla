"""Teste: Inferência qualitativa student (2B) vs teacher (10B).

Usa o MESMO pipeline de pré-processamento do treino (QwenProcessor + chat
template r1_5, generation_mode=True) em vez de montar o prompt manualmente,
para garantir que os tokens especiais (<|question_start|>, <|answer_start|>,
...) sejam tokenizados exatamente como na fase de treino.

Usage:
    python eval_student_vs_teacher.py [n_samples]
"""
import os, sys, glob, random, textwrap
from functools import partial

import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
os.environ["WANDB_DISABLED"]        = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

N              = int(sys.argv[1]) if len(sys.argv) > 1 else 10
STUDENT_CKPT   = f"{BASE}/checkpoints/sft_stage4_v2/checkpoint-8622"
TEACHER_CKPT   = f"{BASE}/checkpoints/sft_stage1_roadwork_v3/checkpoint-11000"
A1_BASE        = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
STUDENT_VLM    = f"{BASE}/models/Cosmos-Reason2-2B"
TEACHER_VLM    = f"{BASE}/models/Cosmos-Reason2-8B"
DATA_ROOT      = f"{BASE}/data/roadwork/lingoqa_roadwork"
OUT_FILE       = f"{BASE}/logs/eval_student_vs_teacher.txt"

from omegaconf import OmegaConf
from safetensors.torch import load_file
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo.data.lingoqa import LingoQADataset
from alpamayo.processor.qwen_processor import QwenProcessor, collate_fn_from_model_config


def load_weights(model, ckpt_dir):
    shards = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    if not shards:
        single = os.path.join(ckpt_dir, "model.safetensors")
        shards = [single] if os.path.exists(single) else []
    sd = {}
    for f in shards:
        sd.update(load_file(f, device="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded OK — missing={len(missing)}  unexpected={len(unexpected)}")
    return model


class PlainConfig:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def plain_config(model):
    raw_cfg = OmegaConf.to_container(model.config, resolve=True) if hasattr(model.config, "_metadata") else None
    return PlainConfig(raw_cfg) if raw_cfg is not None else model.config


VLA_PREPROCESS_ARGS_TMPL = dict(
    chat_template_version="r1_5",
    components_order=["image", "question", "answer"],
    components_prompt=["answer"],
    label_components=["answer"],
    include_camera_ids=True,
    include_frame_nums=True,
    generation_mode=True,
)


def build_eval_pipeline(model_config):
    """Dataset + processor + collate_fn usando exatamente o preprocess de treino."""
    qwen_proc = QwenProcessor(
        vlm_name_or_path=model_config.vlm_name_or_path,
        traj_vocab_size=model_config.traj_vocab_size,
        min_pixels=model_config.min_pixels,
        max_pixels=model_config.max_pixels,
        include_camera_ids=True,
        include_frame_nums=True,
        chat_template_version="r1_5",
    )
    processor = qwen_proc.processor  # builds + caches extended tokenizer (special tokens added)

    vla_preprocess_args = {
        "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        **VLA_PREPROCESS_ARGS_TMPL,
    }
    dataset = LingoQADataset(
        data_root=DATA_ROOT,
        parquet_name="val.parquet",
        model_config=model_config,
        vla_preprocess_args=vla_preprocess_args,
    )
    collate = partial(
        collate_fn_from_model_config,
        model_config=model_config,
        chat_template_version="r1_5",
        include_camera_ids=True,
        include_frame_nums=True,
    )
    return dataset, processor, collate


def run_inference(model, collate, processor, sample, max_new_tokens=256):
    batch = collate([sample])
    inputs = {
        k: v.cuda() for k, v in batch["tokenized_data"].items() if isinstance(v, torch.Tensor)
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=4,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    n_input = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(generated[0][n_input:], skip_special_tokens=True).strip()


print("=" * 70)
print(f"EVAL: Student (2B) vs Teacher (10B)  —  {N} amostras  [pipeline oficial r1_5]")
print("=" * 70)

# ── Student ───────────────────────────────────────────────────────────────
print("\n[1/2] Carregando student (2B)...")
student = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=STUDENT_VLM)
load_weights(student, STUDENT_CKPT)
student = student.to(torch.bfloat16).cuda().eval()
student_config = plain_config(student)
student_dataset, student_processor, student_collate = build_eval_pipeline(student_config)
print("  Student pronto.")

# ── Teacher ───────────────────────────────────────────────────────────────
print("\n[2/2] Carregando teacher (10B)...")
teacher = TrainableReasoningVLA.from_alpamayo_checkpoint(checkpoint_path=A1_BASE, vlm_name_or_path=TEACHER_VLM)
load_weights(teacher, TEACHER_CKPT)
teacher = teacher.to(torch.bfloat16).cuda().eval()
teacher_config = plain_config(teacher)
teacher_dataset, teacher_processor, teacher_collate = build_eval_pipeline(teacher_config)
print("  Teacher pronto.")

assert len(student_dataset) == len(teacher_dataset)
random.seed(42)
indices = random.sample(range(len(student_dataset)), N)
print(f"\n{len(student_dataset)} amostras totais → {N} selecionadas\n")

results = []
sep = "─" * 70

for i, idx in enumerate(indices):
    s_sample = student_dataset[idx]
    t_sample = teacher_dataset[idx]
    question  = s_sample["question"]
    gt_answer = s_sample["answer"]

    try:
        pred_student = run_inference(student, student_collate, student_processor, s_sample)
    except Exception as e:
        pred_student = f"[ERRO student: {e}]"

    try:
        pred_teacher = run_inference(teacher, teacher_collate, teacher_processor, t_sample)
    except Exception as e:
        pred_teacher = f"[ERRO teacher: {e}]"

    results.append({
        "idx": i + 1,
        "question": question, "ground_truth": gt_answer,
        "student": pred_student, "teacher": pred_teacher,
    })

    print(sep)
    print(f"[{i+1}/{N}]")
    print(f"Q: {question[:120]}")
    print(f"\nGT:      {gt_answer[:250]}")
    print(f"STUDENT: {pred_student[:250]}")
    print(f"TEACHER: {pred_teacher[:250]}")
    print()

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write(f"Student (2B) vs Teacher (10B) — {N} amostras [pipeline oficial r1_5]\n")
    f.write("=" * 70 + "\n\n")
    for r in results:
        f.write(f"[{r['idx']}]\n")
        f.write(f"QUESTION:\n  {r['question']}\n\n")
        f.write("GROUND TRUTH:\n")
        for line in textwrap.wrap(r["ground_truth"], 68):
            f.write(f"  {line}\n")
        f.write("\nSTUDENT (2B):\n")
        for line in textwrap.wrap(r["student"], 68):
            f.write(f"  {line}\n")
        f.write("\nTEACHER (10B):\n")
        for line in textwrap.wrap(r["teacher"], 68):
            f.write(f"  {line}\n")
        f.write("\n" + "─" * 70 + "\n\n")

print(f"\nResultados salvos em: {OUT_FILE}")
