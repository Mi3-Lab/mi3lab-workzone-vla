"""Curva de escala: o minADE ainda cai com mais dados, ou ja saturou?

A pergunta que sobrou. Seis abordagens (GT-direto, destilacao de trajetoria, KD
de campo de velocidade, cotrain do VLM, transplante do expert pre-treinado, e uma
arquitetura ResNet+DETR completamente diferente) convergiram todas em minADE
4.57-5.12, contra 1.262 do professor. Variamos alvo, densidade de supervisao, o
que congela, a inicializacao e a arquitetura. Nada moveu. A unica variavel nunca
tocada sao os DADOS: 3644 clipes de ~306k (1%).

Metodo: treina o MESMO modelo com 25%, 50% e 100% dos clipes que ja temos e mede
o minADE de validacao (chunk 3126, o mesmo dos baselines).

Como ler o resultado:
  - se o minADE CAI de 25% -> 50% -> 100% e ainda esta descendo no fim, o modelo
    e' limitado por dados => baixar mais chunks resolve, e a inclinacao estima
    quanto precisamos.
  - se ACHATA (ex: 50% e 100% praticamente iguais), mais dados NAO vao resolver,
    e o limite e' capacidade/arquitetura => a decisao muda completamente.

Usa o preditor 467M e nao o VLA de 4.23B de proposito: cada ponto custa ~15min em
vez de ~1.5h, e a PERGUNTA e' sobre a inclinacao da curva, nao sobre o valor
absoluto. Se ha sinal de dados, ele aparece aqui.
"""
import json
import os
import sys

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla")
sys.path.insert(0, f"{BASE}/alpamayo-recipes/recipes")
os.environ.setdefault("HF_HOME", f"{BASE}/.hf_cache")

import torch
from torch.utils.data import DataLoader, Subset
from alpamayo.data.pai import PAIDataset

from training.traj_student_485m import TrajStudent485M, wta_loss

TRAIN_CHUNKS = [15, 17, 34, 120, 133, 137, 139, 148, 149, 153, 174, 214, 224,
                270, 276, 317, 420, 609, 727, 728, 968, 982, 1519, 1657, 1786,
                1790, 1799, 1862, 1984, 2277, 2368, 2372, 2443, 2447, 2599,
                2634, 2868, 3125]
FRACTIONS = [0.25, 0.50, 1.00]
EPOCHS = int(os.environ.get("EPOCHS", 8))
BS, LR = 8, 1e-4


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "image_frames": torch.stack([b["image_frames"] for b in batch]),
        "ego_history_xyz": torch.stack([b["ego_history_xyz"][0] for b in batch]),
        "ego_history_rot": torch.stack([b["ego_history_rot"][0] for b in batch]),
        "ego_future_xyz": torch.stack([b["ego_future_xyz"][0] for b in batch]),
        "ego_future_rot": torch.stack([b["ego_future_rot"][0] for b in batch]),
    }


def make_ds(chunks):
    return PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=chunks,
                      num_history_steps=16, num_future_steps=64, time_step=0.1,
                      use_default_keyframe=True)


ACT_CFG = {k: v for k, v in json.load(open(
    f"{BASE}/checkpoints/sft_action_2b_kd_velocity/checkpoint-684/config.json"
))["action_space_cfg"].items() if k != "_target_"}

full_tr = make_ds(TRAIN_CHUNKS)
va = make_ds([3126])
dl_va = DataLoader(va, batch_size=BS, shuffle=False, num_workers=4, collate_fn=collate)
print(f"[setup] treino completo {len(full_tr)} clipes | val {len(va)}\n", flush=True)

# subconjuntos ANINHADOS e com semente fixa: 25% esta contido em 50%, que esta em
# 100%. Assim a unica variavel entre os pontos e' a QUANTIDADE de dados -- se
# fossem amostras independentes, parte da diferenca seria so sorte da amostragem.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(full_tr), generator=g).tolist()

results = {}
for frac in FRACTIONS:
    n = int(len(full_tr) * frac)
    ds = Subset(full_tr, perm[:n])
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=8,
                    collate_fn=collate, pin_memory=True, drop_last=True)

    torch.manual_seed(0)   # mesma init em todos os pontos
    model = TrajStudent485M(action_space_cfg=ACT_CFG).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = max(len(dl) * EPOCHS, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()

    print(f"=== {int(frac*100)}% dos dados = {n} clipes, {EPOCHS} epocas ===", flush=True)

    def gt_action_of(b):
        return model.action_space.traj_to_action(
            traj_history_xyz=b["ego_history_xyz"].cuda(),
            traj_history_rot=b["ego_history_rot"].cuda(),
            traj_future_xyz=b["ego_future_xyz"].cuda(),
            traj_future_rot=b["ego_future_rot"].cuda(),
        ).reshape(-1, 64, 2)

    best = 9e9
    for ep in range(EPOCHS):
        model.train()
        for b in dl:
            if b is None:
                continue
            gt_act = gt_action_of(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action, xyz, _, logits = model(b["image_frames"].cuda(),
                                               b["ego_history_xyz"].cuda(),
                                               b["ego_history_rot"].cuda())
                loss, _ = wta_loss(action.float(), xyz.float(), logits.float(),
                                   gt_act.float(), b["ego_future_xyz"].cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for b in dl_va:
                if b is None:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, xyz, _, _ = model(b["image_frames"].cuda(),
                                         b["ego_history_xyz"].cuda(),
                                         b["ego_history_rot"].cuda())
                gt = b["ego_future_xyz"].cuda()
                d = (xyz.float()[..., :2] - gt[:, None, :, :2]).norm(dim=-1).mean(-1)
                tot += d.min(1).values.sum().item()
                cnt += d.shape[0]
        m = tot / max(cnt, 1)
        best = min(best, m)
        print(f"  ep{ep}: minADE(val) {m:.4f}m", flush=True)

    results[frac] = best
    print(f"  >>> {int(frac*100)}% ({n} clipes): melhor minADE = {best:.4f}m\n", flush=True)
    del model, opt
    torch.cuda.empty_cache()

print("=== CURVA DE ESCALA ===")
for frac in FRACTIONS:
    print(f"  {int(frac*100):3d}% ({int(len(full_tr)*frac):5d} clipes): {results[frac]:.4f}m")
ks = sorted(results)
if len(ks) >= 2:
    d1 = results[ks[0]] - results[ks[1]]
    d2 = results[ks[1]] - results[ks[2]] if len(ks) > 2 else 0.0
    print(f"\nganho 25->50%: {d1:+.4f}m | 50->100%: {d2:+.4f}m")
    print("Se o ganho de 50->100% ainda for grande, mais dados resolvem.")
    print("Se ja for ~0, o limite nao e' dado -- e' capacidade/arquitetura.")
