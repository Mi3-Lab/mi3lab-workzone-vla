"""Video: trajetoria prevista mudando QUADRO A QUADRO, professor 10B vs nosso 2B.

Para cada video de workzone, roda os dois modelos a cada STRIDE quadros e escreve
um mp4 com [frame do video | painel BEV]. O painel mostra as 5 amostras de cada
modelo, atualizando conforme a cena evolui.

*** RESSALVA (esta escrita no proprio video) ***
Estes videos nao tem egomotion, e a cabeca de acao EXIGE ego_history. Aqui ele e'
FABRICADO (reta a 10 m/s) e mantido CONSTANTE o tempo todo. Ou seja: a unica coisa
que muda entre um quadro e outro e' a IMAGEM. Isso na verdade tem um lado bom --
isola o efeito da percepcao: toda variacao que voce ve na trajetoria vem do que o
modelo esta VENDO, nao do historico. Mas continua nao sendo evidencia de qualidade
absoluta, porque o prior de movimento e' inventado e nao ha GT.
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
import training.alpamayo_r1_vocab_patch  # noqa: E402

import functools  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from alpamayo.data.pai import PAIDataset  # noqa: E402
from alpamayo.processor.qwen_processor import collate_fn_from_model_config  # noqa: E402
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1  # noqa: E402

TEACHER = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
STUDENT = f"{BASE}/checkpoints/sft_action_2b_cotrain/checkpoint-684"
OUT_DIR = f"{BASE}/outputs/traj_videos"
K = 5
STRIDE = int(os.environ.get("STRIDE", 5))    # infere a cada 5 quadros (~6 Hz)
SPEED_MPS, DT = 10.0, 0.1

VIDEOS = ["boston", "denver", "chicago", "seattle_unseen"]

# painel BEV
PW, FWD_MAX, LAT_MAX = 460, 80.0, 30.0
C_TEACHER = (180, 119, 31)    # BGR — azul
C_STUDENT = (14, 127, 255)    # BGR — laranja

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


print("[1] professor 10B...")
teacher = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype=torch.bfloat16).cuda().eval()
print("[2] nosso 2B (cotrain)...")
student = TrainableAlpamayoR1.from_pretrained(STUDENT, dtype=torch.bfloat16).cuda().eval()

collate = functools.partial(collate_fn_from_model_config,
                            model_config=student.config,
                            chat_template_version="r1_5")

# sample real do PAIDataset como MOLDE: herda shapes/dtypes/chaves exatas que o
# preprocessador espera; trocamos so as imagens e o ego_history.
ds = PAIDataset(local_dir=f"{BASE}/data/PhysicalAI-AV", chunk_ids=[3126],
                use_default_keyframe=True, model_config=student.config,
                vla_preprocess_args=VLA_ARGS)
template = ds[0]
n_cam, n_frm, _, H_IN, W_IN = template["image_frames"].shape

n_hist = template["ego_history_xyz"].shape[1]
hist_xyz = torch.zeros_like(template["ego_history_xyz"])
hist_xyz[0, :, 0] = torch.tensor(-SPEED_MPS * DT * np.arange(n_hist - 1, -1, -1),
                                 dtype=hist_xyz.dtype)
hist_rot = torch.zeros_like(template["ego_history_rot"])
hist_rot[0, :] = torch.eye(3, dtype=hist_rot.dtype)


def infer(rgb):
    """Roda os 2 modelos num frame RGB. Retorna (teacher [K,64,3], student [K,64,3])."""
    img = cv2.resize(rgb, (W_IN, H_IN)).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)
    # o PAIDataset entrega 4 cameras; o video tem 1 (frontal) -> replicamos.
    frames = t[None, None].repeat(n_cam, n_frm, 1, 1, 1).to(template["image_frames"].dtype)

    sample = dict(template)
    sample["image_frames"] = frames
    sample["ego_history_xyz"] = hist_xyz
    sample["ego_history_rot"] = hist_rot
    sample["tokenized_data"] = ds.vla_preprocess_func(data=sample)

    out = []
    for model in (teacher, student):
        # batch FRESCO por modelo: sample_trajectories_from_data faz
        # tokenized_data.pop("input_ids") e CONSOME o batch.
        batch = _to_cuda(collate([sample]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, _ = model.sample_trajectories_from_data(
                data=batch, num_traj_samples=K, num_traj_sets=1,
                top_p=0.98, temperature=0.6, traj_only_generation=True,
                max_generation_length=256)
        out.append(xyz[0, 0].float().cpu().numpy())
    return out


def bev_panel(h, t_xyz, s_xyz):
    """Painel BEV: x=lateral, y=frente. Ego no centro-baixo."""
    p = np.full((h, PW, 3), 22, np.uint8)
    cx, y0 = PW // 2, h - 24
    sx, sy = (PW / 2 - 12) / LAT_MAX, (y0 - 30) / FWD_MAX

    for m in range(10, int(FWD_MAX) + 1, 10):      # grade a cada 10m
        y = int(y0 - m * sy)
        cv2.line(p, (8, y), (PW - 8, y), (55, 55, 55), 1)
        cv2.putText(p, f"{m}m", (10, y - 3), cv2.FONT_HERSHEY_SIMPLEX, .32, (110, 110, 110), 1)
    cv2.line(p, (cx, 30), (cx, y0), (55, 55, 55), 1)

    def draw(trajs, color):
        for tr in trajs:
            pts = np.stack([cx + tr[:, 1] * sx, y0 - tr[:, 0] * sy], 1).astype(np.int32)
            cv2.polylines(p, [pts], False, color, 2, cv2.LINE_AA)

    if t_xyz is not None:
        draw(t_xyz, C_TEACHER)
        draw(s_xyz, C_STUDENT)
    cv2.rectangle(p, (cx - 5, y0 - 4), (cx + 5, y0 + 6), (255, 255, 255), -1)  # ego

    cv2.putText(p, "Professor 10B", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, C_TEACHER, 1)
    cv2.putText(p, "Nosso 2B", (PW - 110, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, C_STUDENT, 1)
    return p


for name in VIDEOS:
    path = f"{BASE}/videos/{name}.mp4"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{OUT_DIR}/traj_{name}.mp4"
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PW, H))
    print(f"\n{name}: {n} quadros, inferindo a cada {STRIDE} (~{fps/STRIDE:.0f} Hz)")

    t_xyz = s_xyz = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            t_xyz, s_xyz = infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if i % (STRIDE * 20) == 0:
                print(f"  quadro {i}/{n}", flush=True)

        vis = frame.copy()
        cv2.putText(vis, "ego_history FABRICADO (reta 10 m/s) - ilustrativo",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, .42, (60, 60, 235), 1)
        wr.write(np.hstack([vis, bev_panel(H, t_xyz, s_xyz)]))

    cap.release()
    wr.release()
    print(f"  -> {out_path}")

print(f"\nvideos em {OUT_DIR}/")
