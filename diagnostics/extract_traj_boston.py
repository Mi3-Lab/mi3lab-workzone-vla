"""Extrai a trajetoria BEV (x=frente,y=lateral,z=altura, metros) prevista pelo
nosso Alpamayo (professor 10B -- unico checkpoint com cabeca de acao que
sobrou; o nosso 2B com acao foi deletado nesta sessao) no frame de workzone
do boston.mp4, salvando os arrays brutos (nao so' a figura) pra comparar com
o Cosmos3-Edge.

MESMA ressalva do compare_traj_workzone.py original: nao temos ego_history
real pra este video, entao o historico e' FABRICADO (reta a 10 m/s). A
trajetoria prevista condiciona num passado inventado -- serve pra ver "o que
o modelo faria se o carro viesse reto a 10m/s vendo esta cena", nao e'
validacao contra realidade.

MUDANCA vs a versao original: data/PhysicalAI-AV (usado so' como "molde" pra
pegar os shapes/dtypes certos do sample, nao pelo conteudo) tambem foi
deletado nesta sessao. Este script reconstroi o sample do zero, direto do
schema documentado em alpamayo1_5/src/alpamayo1_5/load_physical_aiavdataset.py
(docstring + corpo da funcao), sem depender do dataset real. Prints de
shape/dtype/min-max em cada passo pra pegar qualquer erro de reconstrucao
imediatamente, em vez de deixar passar silenciosamente.
"""
import functools
import json
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from alpamayo.processor.qwen_processor import collate_fn_from_model_config  # noqa: E402
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1  # noqa: E402

TEACHER = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VIDEO = f"{BASE}/videos/boston.mp4"
OUT_JSON = f"{BASE}/outputs/cosmos3edge_trajectory_test/alpamayo_boston_trajectory.json"
K = 5
SPEED_MPS = 10.0
DT = 0.1
N_CAM = 4
N_FRM = 4
N_HIST = 16
N_FUT = 64
CAMERA_INDICES = [0, 1, 2, 6]  # cross_left, front_wide, cross_right, front_tele (ordem oficial)

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


def sample_k(model, collate, sample, k):
    batch = _to_cuda(collate([sample]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=batch, num_traj_samples=k, num_traj_sets=1,
            top_p=0.98, temperature=0.6, traj_only_generation=True,
            max_generation_length=256,
        )
    return pred_xyz[0, 0].float().cpu().numpy()


print("[1] Carregando professor 10B (unico checkpoint com cabeca de acao disponivel)...")
teacher = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype=torch.bfloat16).cuda().eval()

collate = functools.partial(collate_fn_from_model_config,
                            model_config=teacher.config,
                            chat_template_version="r1_5")

print("[2] Lendo frame real de boston.mp4...")
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.6))
ok, frame = cap.read()
cap.release()
assert ok, f"falha ao ler frame de {VIDEO}"
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[0], rgb.shape[1]
print(f"    frame: {rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}")

print("[3] Construindo sample do zero (schema de load_physical_aiavdataset.py)...")
# image_frames: (N_cameras, num_frames, 3, H, W), uint8 [0,255] -- schema RAW
# documentado no docstring, mesmo formato que a camera.decode_images_from_timestamps
# devolveria. Mesmo frame real repetido nas 4 cameras x 4 frames (aproximacao
# grosseira, igual ao script original -- so' temos 1 camera frontal real).
frame_u8 = torch.from_numpy(rgb.copy()).permute(2, 0, 1)  # [3,H,W] uint8
image_frames = frame_u8[None, None].repeat(N_CAM, N_FRM, 1, 1, 1)  # [N_CAM,N_FRM,3,H,W] uint8

camera_indices = torch.tensor(CAMERA_INDICES, dtype=torch.int64)
relative_timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3]).unsqueeze(0).repeat(N_CAM, 1)
t0_us = 5_100_000
absolute_timestamps = (
    t0_us + torch.tensor([-300_000, -200_000, -100_000, 0]).unsqueeze(0).repeat(N_CAM, 1)
).long()

# ego_history: historico FABRICADO -- reta a SPEED_MPS m/s terminando na origem.
# x=frente, y=lateral, z=altura (convencao do proprio load_physical_aiavdataset).
xs = -SPEED_MPS * DT * np.arange(N_HIST - 1, -1, -1)  # ex: -1.5m ... 0m
ego_history_xyz = torch.zeros(1, N_HIST, 3, dtype=torch.float32)
ego_history_xyz[0, :, 0] = torch.tensor(xs, dtype=torch.float32)
ego_history_rot = torch.eye(3, dtype=torch.float32).expand(1, N_HIST, 3, 3).clone()

# ego_future: zeros -- so' usado pra construir o placeholder do prompt em
# generation_mode=True (labels_mask fica zerado, conteudo nao importa).
ego_future_xyz = torch.zeros(1, N_FUT, 3, dtype=torch.float32)
ego_future_rot = torch.eye(3, dtype=torch.float32).expand(1, N_FUT, 3, 3).clone()

sample = {
    "image_frames": image_frames,
    "camera_indices": camera_indices,
    "relative_timestamps": relative_timestamps,
    "absolute_timestamps": absolute_timestamps,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
    "ego_future_xyz": ego_future_xyz,
    "ego_future_rot": ego_future_rot,
    "t0_us": t0_us,
    "clip_id": "boston_workzone_fabricated",
}
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.float().min().item():.3f} max={v.float().max().item():.3f}")
    else:
        print(f"    {k}: {v}")

print("[4] Rodando o preprocessador VLA (mesmo usado no treino/inferencia real)...")
# Mesma chamada que PAIDataset.__init__ faz: so' embrulha em OmegaConf se
# model_config vier como dict puro; teacher.config e' um PretrainedConfig, vai
# direto, sem transformacao extra.
vla_preprocess_func = instantiate(VLA_ARGS, model_config=teacher.config)
sample["tokenized_data"] = vla_preprocess_func(data=sample)
print(f"    chaves de tokenized_data: {sorted(sample['tokenized_data'].keys())}")

print("[5] Amostrando trajetoria (K=5) do professor 10B...")
t_xyz = sample_k(teacher, collate, sample, K)
print(f"    shape da saida: {t_xyz.shape}")
print(f"    professor 10B: alcance medio (6.4s) = {np.linalg.norm(t_xyz[:, -1, :2], axis=-1).mean():.2f} m")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump({
    "convention": "x=frente(m), y=lateral(m), z=altura(m); DT=0.1s; historico FABRICADO reta 10m/s; "
                  "sample reconstruido do zero (data/PhysicalAI-AV foi deletado nesta sessao)",
    "teacher_10b_xyz": t_xyz.tolist(),   # [K, T, 3]
}, open(OUT_JSON, "w"), indent=2)
print(f"\nsalvo em {OUT_JSON}")
