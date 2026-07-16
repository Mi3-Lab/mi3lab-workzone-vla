"""Compara visualmente as trajetorias do professor 10B e do nosso 2B (cotrain).

Para cada clipe de validacao (chunk 3126 -- os MESMOS de onde saem as metricas
minADE 1.262 vs 4.572), amostra K trajetorias de cada modelo e plota tudo em BEV
(vista de cima, em metros) junto com o GT.

Por que BEV e nao projecao na imagem: o ADE e medido em metros no referencial do
ego, entao o BEV mostra exatamente a grandeza que a metrica mede. Projetar na
camera exigiria o modelo fisheye polinomial (f-theta) dessas cameras; uma
projecao mal calibrada distorceria a leitura da figura. A imagem entra ao lado,
so como contexto visual da cena.
"""
import os
import sys

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
import training.alpamayo_r1_vocab_patch  # noqa: E402  (patch de vocabulario)

import functools  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from alpamayo.data.pai import PAIDataset  # noqa: E402
from alpamayo.processor.qwen_processor import collate_fn_from_model_config  # noqa: E402
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1  # noqa: E402

TEACHER = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
STUDENT = f"{BASE}/checkpoints/sft_action_2b_cotrain/checkpoint-684"
OUT_DIR = f"{BASE}/outputs/traj_compare"
N_CLIPS = int(os.environ.get("N_CLIPS", 6))
K = int(os.environ.get("K_SAMPLES", 5))

os.makedirs(OUT_DIR, exist_ok=True)

VLA_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": False,
    "include_frame_nums": False,
    "generation_mode": True,
}


def _to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, dict):
        return {k: _to_cuda(v) for k, v in x.items()}
    return x


def sample_k(model, sample, k):
    """Monta um batch NOVO a cada chamada de proposito.

    sample_trajectories_from_data faz tokenized_data.pop("input_ids") -- ou seja,
    CONSOME o batch. Reusar o mesmo batch pro segundo modelo estoura
    KeyError: 'input_ids'. Entao cada modelo recebe o seu.
    """
    batch = _to_cuda(collate([sample]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=batch, num_traj_samples=k, num_traj_sets=1,
            top_p=0.98, temperature=0.6, traj_only_generation=True,
            max_generation_length=256,
        )
    return pred_xyz[0, 0].float().cpu().numpy(), batch  # [K, 64, 3]


def ade(pred, gt):
    """pred [K,64,3]; gt [64,3]. Distancia media no plano (x,y), por amostra."""
    return np.linalg.norm(pred[..., :2] - gt[None, :, :2], axis=-1).mean(axis=-1)


print("[1] Carregando professor 10B...")
teacher = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype=torch.bfloat16).cuda().eval()
print("[2] Carregando nosso 2B (cotrain)...")
student = TrainableAlpamayoR1.from_pretrained(STUDENT, dtype=torch.bfloat16).cuda().eval()

# um unico dataset/collate serve os dois: mesmo vocabulario (155697) e mesmo
# preprocessamento de imagem (720 patches) -- foi o que viabilizou o KD tambem.
ds_raw = PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=[3126],
                    use_default_keyframe=True)
ds = PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=[3126],
                use_default_keyframe=True, model_config=student.config,
                vla_preprocess_args=VLA_ARGS)
collate = functools.partial(collate_fn_from_model_config,
                            model_config=student.config,
                            chat_template_version="r1_5")
print(f"[3] {len(ds)} clipes de validacao; renderizando {N_CLIPS} com K={K}\n")

rows = []
for i in range(N_CLIPS):
    sample = ds[i]
    if sample is None:
        continue

    t_xyz, batch = sample_k(teacher, sample, K)   # [K,64,3]
    s_xyz, _ = sample_k(student, sample, K)

    gt = batch["ego_future_xyz"]
    gt = (gt[:, -1] if gt.dim() == 4 else gt)[0].float().cpu().numpy()  # [64,3]

    t_ade, s_ade = ade(t_xyz, gt), ade(s_xyz, gt)
    cid = sample["clip_id"][:8]
    rows.append((cid, t_ade.min(), s_ade.min()))
    print(f"  clipe {i} ({cid}): professor minADE {t_ade.min():.2f}m | "
          f"nosso 2B minADE {s_ade.min():.2f}m")

    # --- figura ---
    fig, (axi, axb) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    frames = ds_raw[i]["image_frames"]
    img = frames[2, -1].permute(1, 2, 0).numpy()   # camera frontal (front_tele)
    axi.imshow(np.clip(img * 255 if img.max() <= 1.01 else img, 0, 255).astype(np.uint8))
    axi.set_title(f"camera frontal — clipe {cid}", fontsize=11)
    axi.axis("off")

    for j, p in enumerate(t_xyz):
        axb.plot(p[:, 1], p[:, 0], color="#1f77b4", lw=1.6, alpha=.85,
                 label="Professor 10B (5 amostras)" if j == 0 else None)
    for j, p in enumerate(s_xyz):
        axb.plot(p[:, 1], p[:, 0], color="#ff7f0e", lw=1.6, alpha=.85,
                 label="Nosso 2B (5 amostras)" if j == 0 else None)
    axb.plot(gt[:, 1], gt[:, 0], color="k", lw=3.2, label="GT (real)", zorder=5)
    axb.plot(0, 0, marker="s", ms=9, color="k")   # ego

    axb.set_xlabel("lateral (m)   <- esquerda | direita ->")
    axb.set_ylabel("frente (m)")
    axb.set_title(f"BEV 6.4s — professor {t_ade.min():.2f}m | 2B {s_ade.min():.2f}m",
                  fontsize=11)
    axb.grid(alpha=.3)
    axb.legend(loc="upper left", fontsize=9)
    axb.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/clip_{i:02d}_{cid}.png", dpi=110)
    plt.close(fig)

print(f"\n=== resumo ({len(rows)} clipes) ===")
t_m = np.mean([r[1] for r in rows])
s_m = np.mean([r[2] for r in rows])
print(f"minADE medio: professor {t_m:.2f}m | nosso 2B {s_m:.2f}m")
print(f"figuras em {OUT_DIR}/")
