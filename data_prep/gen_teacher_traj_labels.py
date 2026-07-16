"""Geracao de labels de destilacao: o professor Alpamayo-1.5-10B real gera
K trajetorias por clipe dos chunks de treino; salvamos a MELHOR (menor ADE
vs GT) + a propria distancia ao GT (pra filtro de qualidade no treino).

Receita = "Phase 1: output-level trajectory distillation" documentada pelo
proprio time que deployou o Alpamayo no Thor (docs/current_model_distillation
_deployment_plan.md do fork kimsunguk0): teacher target com prioridade pro
GT quando divergem muito.

Suporta sharding pra paralelizar em varios jobs:
    SHARD=0 NUM_SHARDS=4 python gen_teacher_traj_labels.py
Saida: data/teacher_traj_labels/shard_{i}.pt  (dict clip_id -> label)
"""
import os, sys, glob, re

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
_RECIPES = f"{BASE}/alpamayo-recipes/recipes"
sys.path.insert(0, f"{_RECIPES}/alpamayo1_5_sft")
sys.path.insert(0, _RECIPES)
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla")
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": f"{BASE}/.hf_cache", "WANDB_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
})
import training.alpamayo_r1_vocab_patch  # noqa: E402

import torch  # noqa: E402
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1  # noqa: E402
from alpamayo.data.pai import PAIDataset  # noqa: E402

TEACHER = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
PAI_DIR = f"{BASE}/data/PhysicalAI-AV"
OUT_DIR = f"{BASE}/data/teacher_traj_labels"
os.makedirs(OUT_DIR, exist_ok=True)

# chunks de TREINO (3126 = validacao, fica fora)
TRAIN_CHUNKS = [15, 17, 34, 120, 133, 137, 139, 148, 149, 153, 174, 214, 224,
                270, 276, 317, 420, 609, 727, 728, 968, 982, 1519, 1657, 1786,
                1790, 1799, 1862, 1984, 2277, 2368, 2372, 2443, 2447, 2599,
                2634, 2868, 3125]
K_SAMPLES = int(os.environ.get("K_SAMPLES", 6))
SHARD = int(os.environ.get("SHARD", 0))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", 1))
OUT_PATH = f"{OUT_DIR}/shard_{SHARD}_of_{NUM_SHARDS}.pt"

print(f"[shard {SHARD}/{NUM_SHARDS}] K={K_SAMPLES}")
print("[1] Carregando professor 10B...")
model = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype=torch.bfloat16)
model = model.cuda().eval()
print("    OK")

vla_preprocess_args = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": False,
    "include_frame_nums": False,
    "generation_mode": True,
}
ds = PAIDataset(
    local_dir=PAI_DIR, chunk_ids=TRAIN_CHUNKS,
    num_history_steps=16, num_future_steps=64, time_step=0.1,
    use_default_keyframe=True,
    model_config=model.config, vla_preprocess_args=vla_preprocess_args,
)
idxs = list(range(SHARD, len(ds), NUM_SHARDS))
print(f"[2] Dataset: {len(ds)} clipes | este shard: {len(idxs)}")

# retomada: pula o que ja foi salvo
labels = {}
if os.path.exists(OUT_PATH):
    labels = torch.load(OUT_PATH)
    print(f"    retomando: {len(labels)} labels ja salvos")

import functools  # noqa: E402
from alpamayo.processor.qwen_processor import collate_fn_from_model_config  # noqa: E402
collate = functools.partial(collate_fn_from_model_config,
                            model_config=model.config,
                            chat_template_version="r1_5")

done = 0
for n, i in enumerate(idxs):
    sample = ds[i]
    if sample is None:
        continue
    clip_id = sample["clip_id"]
    if clip_id in labels:
        continue
    batch = collate([sample])

    def _to_cuda(x):
        if isinstance(x, torch.Tensor):
            return x.cuda()
        if isinstance(x, dict):
            return {k: _to_cuda(v) for k, v in x.items()}
        return x

    batch = _to_cuda(batch)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot = model.sample_trajectories_from_data(
            data=batch, num_traj_samples=K_SAMPLES, num_traj_sets=1,
            top_p=0.98, temperature=0.6, traj_only_generation=True,
            max_generation_length=256,
        )
    # pred_xyz: [1, 1, K, 64, 3]; GT: ego_future_xyz [1, (G,) 64, 3]
    gt = batch["ego_future_xyz"]
    gt = gt[:, -1] if gt.dim() == 4 else gt          # [1, 64, 3]
    diff = pred_xyz[0, 0, :, :, :2] - gt[0, None, :, :2]   # [K, 64, 2]
    ade_k = diff.norm(dim=-1).mean(dim=-1)                 # [K]
    best = int(ade_k.argmin())
    labels[clip_id] = {
        "teacher_xyz": pred_xyz[0, 0, best].cpu().to(torch.float32),   # [64, 3]
        "teacher_rot": pred_rot[0, 0, best].cpu().to(torch.float32),   # [64, 3, 3]
        "teacher_ade_to_gt": float(ade_k[best]),
    }
    done += 1
    if done % 25 == 0:
        torch.save(labels, OUT_PATH)
        print(f"  [{n+1}/{len(idxs)}] salvos {len(labels)} labels "
              f"(ultimo ade_to_gt={labels[clip_id]['teacher_ade_to_gt']:.3f}m)")

torch.save(labels, OUT_PATH)
print(f"\n=== SHARD {SHARD} CONCLUIDO: {len(labels)} labels em {OUT_PATH} ===")
ades = [v["teacher_ade_to_gt"] for v in labels.values()]
if ades:
    import statistics
    print(f"teacher ADE-to-GT: media {statistics.mean(ades):.3f}m | "
          f"mediana {statistics.median(ades):.3f}m")
