"""Smoke test: KD "puro" (teacher 10B original + student 2B fresco) + PhysicalAI-AV.

Valida o pipeline de KD geral (sem work zone) antes de gastar horas de GPU:
carrega teacher e student, pega 1 amostra do PhysicalAI-AV, roda forward de
ambos, calcula loss_sft + loss_kd (mesma lógica do KDTrainer.compute_loss),
confirma shapes e mede VRAM.
"""
import os, sys
import torch
import torch.nn.functional as F

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
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_sft.models.kd_model import build_student_model

print("[1] Carregando teacher (Alpamayo-1.5-10B-A1-format, ORIGINAL, sem fine-tune)...")
teacher = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=f"{BASE}/models/Alpamayo-1.5-10B-A1-format",
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-8B",
)
teacher = teacher.to(torch.bfloat16).cuda().eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"    OK — VRAM apos teacher: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("[2] Construindo student (Cosmos-Reason2-2B, do zero, mesma arquitetura Alpamayo)...")
student = build_student_model(
    alpamayo_base_path=f"{BASE}/models/Alpamayo-1.5-10B-A1-format",
    student_vlm_path=f"{BASE}/models/Cosmos-Reason2-2B",
)
student = student.to(torch.bfloat16).cuda().eval()
print(f"    OK — VRAM apos student: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("[3] Instanciando PAIDataset (chunk 214, 1 amostra, SEM reasoning_metadata)...")
ds = PAIDataset(
    local_dir=f"{BASE}/data/PhysicalAI-AV",
    chunk_ids=[214],
    use_default_keyframe=True,
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
    model_config=student.config,
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
batch = collate_fn_from_model_config(
    [sample], model_config=student.config, chat_template_version="r1_5"
)
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

batch_cuda = {}
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch_cuda[k] = v.cuda()
    elif isinstance(v, dict):
        batch_cuda[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv) for kk, vv in v.items()}
    else:
        batch_cuda[k] = v

print("[6] Forward pass do student (loss_sft)...")
td_orig = batch_cuda.get("tokenized_data", {})
student_inputs = {**batch_cuda, "tokenized_data": dict(td_orig)}
teacher_inputs = {**batch_cuda, "tokenized_data": dict(td_orig)}

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    student_out = student(**student_inputs)
loss_sft = student_out.loss
logits_s = student_out.logits
print(f"    loss_sft = {loss_sft.item():.4f}")
print(f"    logits_s shape = {tuple(logits_s.shape)}")

print("[7] Forward pass do teacher...")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    teacher_out = teacher(**teacher_inputs)
logits_t = teacher_out.logits.to(logits_s.device, dtype=torch.float32)
print(f"    logits_t shape = {tuple(logits_t.shape)}")
print(f"    vocab match: {logits_s.shape[-1] == logits_t.shape[-1]}")

print("[8] Calculando loss_kd (KL(teacher||student), tau=0.7, restrito a labels_mask)...")
shift_s = logits_s[..., :-1, :].float()
shift_t = logits_t[..., :-1, :]
labels_mask = batch_cuda.get("labels_mask")
if labels_mask is not None:
    mask = labels_mask[:, 1:]
    n_valid = mask.sum().item()
    print(f"    tokens supervisionados (labels_mask): {n_valid} de {mask.numel()}")
    shift_s = shift_s[mask]
    shift_t = shift_t[mask]
tau = 0.7
p_t = F.softmax(shift_t / tau, dim=-1)
log_p_s = F.log_softmax(shift_s / tau, dim=-1)
loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)
print(f"    loss_kd = {loss_kd.item():.4f}")

lam = 0.5
loss_total = (1 - lam) * loss_sft + lam * loss_kd
print(f"    loss_total (lambda=0.5) = {loss_total.item():.4f}")

print(f"\n[9] VRAM total alocada: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"    VRAM pico (max_memory_allocated): {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

print("\n=== SMOKE TEST KD GERAL PASSOU ===")
