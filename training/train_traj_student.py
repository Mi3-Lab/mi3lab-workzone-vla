"""Treina o preditor compacto de trajetoria (467M) nos mesmos dados/splits do VLA.

Comparabilidade: mesmos chunks de treino (3644 clipes), mesmo chunk de validacao
(3126), mesmo action space (UnicycleAccelCurvatureActionSpace) e a metrica e'
calculada igual a do evaluate_hf (minADE = melhor de M modos, distancia media no
plano xy). Assim o numero sai lado a lado com os baselines:
    GT-direto 4.621 | destilado 4.589 | KD velocidade 4.590 | cotrain 4.572
    professor 10B   1.262
"""
import json
import os
import sys
import time

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla")
sys.path.insert(0, f"{BASE}/alpamayo-recipes/recipes")
os.environ.setdefault("HF_HOME", f"{BASE}/.hf_cache")

import torch
from torch.utils.data import DataLoader
from alpamayo.data.pai import PAIDataset

from training.traj_student_485m import TrajStudent485M, wta_loss

TRAIN_CHUNKS = [15, 17, 34, 120, 133, 137, 139, 148, 149, 153, 174, 214, 224,
                270, 276, 317, 420, 609, 727, 728, 968, 982, 1519, 1657, 1786,
                1790, 1799, 1862, 1984, 2277, 2368, 2372, 2443, 2447, 2599,
                2634, 2868, 3125]
OUT = f"{BASE}/checkpoints/traj_student_485m"
EPOCHS = int(os.environ.get("EPOCHS", 12))
BS = int(os.environ.get("BS", 8))
LR = float(os.environ.get("LR", 2e-4))
os.makedirs(OUT, exist_ok=True)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "image_frames": torch.stack([b["image_frames"] for b in batch]),
        # PAIDataset entrega [1, N, ...] (dim de grupo); tiramos o 1
        "ego_history_xyz": torch.stack([b["ego_history_xyz"][0] for b in batch]),
        "ego_history_rot": torch.stack([b["ego_history_rot"][0] for b in batch]),
        "ego_future_xyz": torch.stack([b["ego_future_xyz"][0] for b in batch]),
        # necessario pra derivar a ACAO alvo (accel, curvatura) via traj_to_action
        "ego_future_rot": torch.stack([b["ego_future_rot"][0] for b in batch]),
    }


def make_ds(chunks):
    return PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=chunks,
                      num_history_steps=16, num_future_steps=64, time_step=0.1,
                      use_default_keyframe=True)


print("[1] datasets...")
tr = make_ds(TRAIN_CHUNKS)
va = make_ds([3126])
print(f"    treino {len(tr)} | val {len(va)}")

dl_tr = DataLoader(tr, batch_size=BS, shuffle=True, num_workers=8,
                   collate_fn=collate, pin_memory=True, drop_last=True)
dl_va = DataLoader(va, batch_size=BS, shuffle=False, num_workers=4, collate_fn=collate)

print("[2] modelo...")
cfg = json.load(open(f"{BASE}/checkpoints/sft_action_2b_kd_velocity/checkpoint-684/config.json"))["action_space_cfg"]
cfg = {k: v for k, v in cfg.items() if k != "_target_"}
model = TrajStudent485M(action_space_cfg=cfg).cuda()
print(f"    {sum(p.numel() for p in model.parameters()):,} params")

def gt_action_of(b):
    """Acao (accel, curvatura) que gera o GT — o alvo primario de supervisao.

    traj_to_action e' o inverso exato do action_to_traj usado no forward, entao o
    alvo e' consistente com a integracao do modelo por construcao.
    """
    return model.action_space.traj_to_action(
        traj_history_xyz=b["ego_history_xyz"].cuda(),
        traj_history_rot=b["ego_history_rot"].cuda(),
        traj_future_xyz=b["ego_future_xyz"].cuda(),
        traj_future_rot=b["ego_future_rot"].cuda(),
    ).reshape(-1, 64, 2)


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = len(dl_tr) * EPOCHS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.05)
scaler = torch.amp.GradScaler()


@torch.no_grad()
def evaluate():
    model.eval()
    tot, n = 0.0, 0
    for b in dl_va:
        if b is None:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, xyz, _, _ = model(b["image_frames"].cuda(), b["ego_history_xyz"].cuda(),
                                 b["ego_history_rot"].cuda())
        gt = b["ego_future_xyz"].cuda()
        d = (xyz.float()[..., :2] - gt[:, None, :, :2]).norm(dim=-1).mean(-1)  # [B,M]
        tot += d.min(1).values.sum().item()
        n += d.shape[0]
    model.train()
    return tot / max(n, 1)


print(f"[3] treinando {EPOCHS} epocas, {steps} passos\n")
best, t0 = 9e9, time.time()
for ep in range(EPOCHS):
    for i, b in enumerate(dl_tr):
        if b is None:
            continue
        gt_act = gt_action_of(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action, xyz, _, logits = model(b["image_frames"].cuda(),
                                           b["ego_history_xyz"].cuda(),
                                           b["ego_history_rot"].cuda())
            loss, reg = wta_loss(action.float(), xyz.float(), logits.float(),
                                 gt_act.float(), b["ego_future_xyz"].cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if i % 50 == 0:
            print(f"  ep{ep} {i}/{len(dl_tr)} loss {loss.item():.3f} "
                  f"minADE_treino {reg.item():.3f} ({(time.time()-t0)/60:.0f}min)", flush=True)

    m = evaluate()
    print(f"=== epoca {ep}: minADE(val) {m:.4f}m ===", flush=True)
    if m < best:
        best = m
        torch.save({"model": model.state_dict(), "minade": m, "epoch": ep},
                   f"{OUT}/best.pt")
        print(f"    salvo (melhor ate agora)", flush=True)

print(f"\n=== MELHOR minADE(val) = {best:.4f}m ===")
print("baselines: nosso VLA 4.572 | professor 10B 1.262")
