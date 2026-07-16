"""PARTE 2 (ILUSTRATIVA): trajetorias do professor 10B vs nosso 2B nos videos de
workzone que usamos nos testes (boston, denver, chicago, seattle).

*** LEIA ISTO ANTES DE OLHAR AS FIGURAS ***

Estes videos NAO TEM egomotion. A cabeca de acao do Alpamayo EXIGE ego_history
(16 passos do movimento recente do carro) como entrada -- e nao temos. Entao aqui
o historico e' FABRICADO: linha reta a ~10 m/s.

Consequencia: as trajetorias que saem daqui sao condicionadas num passado que eu
inventei. Elas mostram "o que cada modelo faria SE o carro viesse reto a 10 m/s e
visse esta cena". NAO servem pra julgar qual modelo e' melhor -- pra isso existe
o compare_traj_models.py, que roda nos clipes de validacao com egomotion real E
trajetoria de referencia.

Tambem nao ha GT aqui, entao nao ha contra o que medir erro.
"""
import glob
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
import training.alpamayo_r1_vocab_patch  # noqa: E402

import functools  # noqa: E402
import cv2  # noqa: E402
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
OUT_DIR = f"{BASE}/outputs/traj_compare_workzone"
K = 5
SPEED_MPS = 10.0          # velocidade fabricada
DT = 0.1                  # time_step do dataset

VIDEOS = [
    ("boston", f"{BASE}/videos/boston.mp4"),
    ("denver", f"{BASE}/videos/denver.mp4"),
    ("chicago", f"{BASE}/videos/chicago.mp4"),
    ("seattle_unseen", f"{BASE}/videos/seattle_unseen.mp4"),
]
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


print("[1] Carregando professor 10B...")
teacher = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype=torch.bfloat16).cuda().eval()
print("[2] Carregando nosso 2B (cotrain)...")
student = TrainableAlpamayoR1.from_pretrained(STUDENT, dtype=torch.bfloat16).cuda().eval()

collate = functools.partial(collate_fn_from_model_config,
                            model_config=student.config,
                            chat_template_version="r1_5")

# Um sample REAL do PAIDataset serve de molde: copiamos a estrutura exata
# (shapes/dtypes/chaves que o preprocessador espera) e trocamos so as imagens e
# o ego_history. Assim nao preciso adivinhar o formato do batch.
print("[3] Pegando um sample-molde do PAIDataset...")
ds = PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=[3126],
                use_default_keyframe=True, model_config=student.config,
                vla_preprocess_args=VLA_ARGS)
template = ds[0]
n_cam, n_frm, C, H, W = template["image_frames"].shape
print(f"    molde: image_frames [{n_cam} cams, {n_frm} frames, {C}, {H}, {W}]")

# ego_history FABRICADO: carro vindo reto a SPEED_MPS, terminando na origem.
# x = frente, y = lateral, z = altura. Rotacao = identidade (sem guinada).
n_hist = template["ego_history_xyz"].shape[1]
xs = -SPEED_MPS * DT * np.arange(n_hist - 1, -1, -1)     # ex: -1.5m ... 0m
hist_xyz = torch.zeros_like(template["ego_history_xyz"])
hist_xyz[0, :, 0] = torch.tensor(xs, dtype=hist_xyz.dtype)
hist_rot = torch.zeros_like(template["ego_history_rot"])
hist_rot[0, :] = torch.eye(3, dtype=hist_rot.dtype)


def sample_k(model, sample, k):
    # batch FRESCO por modelo: sample_trajectories_from_data faz
    # tokenized_data.pop("input_ids"), ou seja CONSOME o batch.
    batch = _to_cuda(collate([sample]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=batch, num_traj_samples=k, num_traj_sets=1,
            top_p=0.98, temperature=0.6, traj_only_generation=True,
            max_generation_length=256,
        )
    return pred_xyz[0, 0].float().cpu().numpy()


for name, path in VIDEOS:
    if not os.path.exists(path):
        print(f"  !! {name}: video nao encontrado em {path}")
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.6))   # frame ja dentro da obra
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  !! {name}: falha ao ler frame")
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # O PAIDataset entrega 4 cameras (cross_left, cross_right, front_tele,
    # rear_tele). O video de workzone tem UMA camera frontal. Replicamos esse
    # frame nas 4 posicoes -- ou seja, o modelo "ve" a mesma imagem frontal em
    # todas as vistas. E' mais uma aproximacao grosseira desta figura.
    img = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)                # [3,H,W]
    frames = t[None, None].repeat(n_cam, n_frm, 1, 1, 1)      # [cam,frm,3,H,W]

    sample = dict(template)
    sample["image_frames"] = frames.to(template["image_frames"].dtype)
    sample["ego_history_xyz"] = hist_xyz
    sample["ego_history_rot"] = hist_rot
    sample["clip_id"] = name
    # reaplica o preprocessador sobre os dados trocados
    sample["tokenized_data"] = ds.vla_preprocess_func(data=sample)

    t_xyz = sample_k(teacher, sample, K)
    s_xyz = sample_k(student, sample, K)
    print(f"  {name}: professor alcance {np.linalg.norm(t_xyz[:, -1, :2], axis=-1).mean():.1f}m | "
          f"2B alcance {np.linalg.norm(s_xyz[:, -1, :2], axis=-1).mean():.1f}m")

    fig, (axi, axb) = plt.subplots(1, 2, figsize=(15, 6.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    axi.imshow(rgb)
    axi.set_title(f"{name} — frame do video de workzone", fontsize=11)
    axi.axis("off")

    for j, p in enumerate(t_xyz):
        axb.plot(p[:, 1], p[:, 0], color="#1f77b4", lw=1.6, alpha=.85,
                 label="Professor 10B (5 amostras)" if j == 0 else None)
    for j, p in enumerate(s_xyz):
        axb.plot(p[:, 1], p[:, 0], color="#ff7f0e", lw=1.6, alpha=.85,
                 label="Nosso 2B (5 amostras)" if j == 0 else None)
    axb.plot(0, 0, marker="s", ms=9, color="k")
    axb.set_xlabel("lateral (m)   <- esquerda | direita ->")
    axb.set_ylabel("frente (m)")
    axb.set_title("BEV 6.4s — SEM trajetoria de referencia (GT)", fontsize=11)
    axb.grid(alpha=.3)
    axb.legend(loc="upper left", fontsize=9)
    axb.set_aspect("equal", adjustable="datalim")

    fig.suptitle("HISTORICO DE EGOMOTION FABRICADO (reta, 10 m/s) — ILUSTRATIVO, NAO E EVIDENCIA",
                 fontsize=13, color="#b00000", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT_DIR}/wz_{name}.png", dpi=110)
    plt.close(fig)

print(f"\nfiguras em {OUT_DIR}/")
