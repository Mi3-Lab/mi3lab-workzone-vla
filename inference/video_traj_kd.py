"""Inferência de TRAJETÓRIA em vídeo — student KD 2B (kd_general_driving).

Para cada janela do vídeo (câmera frontal única, 4 frames espaçados ~0.1s),
o modelo prevê K=6 trajetórias futuras de 6.4s (128 tokens discretos,
decodificação RESTRITA ao vocab de trajetória) e desenhamos num painel
bird's-eye-view (BEV) ao lado do frame — sem projeção na imagem, pois não
temos calibração da dashcam.

Limitações conhecidas (impressas no painel):
- histórico de ego sintético (velocidade constante assumida, sem IMU/GPS real)
- 1 câmera (o modelo foi treinado com 4 câmeras do PhysicalAI-AV)
- modelo de direção GERAL (clean init, sem o conhecimento work zone do stage 6)

Uso: python video_traj_kd.py <video_in.mp4> <video_out.mp4> [checkpoint]
"""
import os, sys, time
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import LogitsProcessor, LogitsProcessorList
from alpamayo.processor.qwen_processor import (
    get_preprocess_data_fn_from_model_config, collate_fn_from_model_config,
)
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

VIDEO_IN  = sys.argv[1]
VIDEO_OUT = sys.argv[2]
CKPT      = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/kd_general_driving"

FRONT_CAM_IDX = 1          # "Front camera" (constants do Alpamayo)
N_FRAMES      = 4          # frames por predição, espaçados ~0.1s
ASSUMED_SPEED = 7.0        # m/s — histórico sintético (dashcam urbana)
K_SAMPLES     = 6
PRED_EVERY_S  = 0.6        # nova predição a cada 0.6s de vídeo
TOP_P, TEMP   = 0.98, 0.6

PANEL_W  = 380
BEV_FWD  = 50.0            # metros à frente visíveis no BEV
BEV_LAT  = 15.0            # metros laterais (±)


class TrajOnlyLogits(LogitsProcessor):
    def __init__(self, start_idx, traj_vocab, full_vocab):
        allowed = torch.zeros(full_vocab, dtype=torch.bool)
        allowed[start_idx:start_idx + traj_vocab] = True
        self.block = ~allowed

    def __call__(self, input_ids, scores):
        scores[:, self.block.to(scores.device)] = float("-inf")
        return scores


def batch_to_cuda(batch):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cuda()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


print(f"[1] Carregando student KD: {CKPT}")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=CKPT,
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-2B",
).to(torch.bfloat16).cuda().eval()

proc = TrajOnlyLogits(
    model.future_token_start_idx,
    model.traj_tokenizer.vocab_size,
    model.vlm.get_input_embeddings().num_embeddings,
)
_orig_gen = model.vlm.generate
LAST_LOGPROBS = {"lp": None}   # logprob somado por amostra, preenchido a cada generate

def _constrained(*a, **kw):
    kw["logits_processor"] = LogitsProcessorList([proc])
    out = _orig_gen(*a, **kw)
    # Ranqueia as amostras: soma do logprob do token escolhido em cada passo
    if hasattr(out, "logits") and out.logits is not None:
        L = kw["input_ids"].shape[1]
        gen_tokens = out.sequences[:, L:]                       # [K, T]
        lp = torch.zeros(gen_tokens.shape[0], device=gen_tokens.device)
        for t, step_logits in enumerate(out.logits):            # T × [K, V]
            logp = torch.log_softmax(step_logits.float(), dim=-1)
            lp += logp.gather(1, gen_tokens[:, t:t+1]).squeeze(1)
        LAST_LOGPROBS["lp"] = lp.cpu().numpy()
    return out
model.vlm.generate = _constrained

preprocess_fn = get_preprocess_data_fn_from_model_config(
    model_config=model.config, chat_template_version="r1_5",
    components_order=["image", "traj_history", "route", "prompt", "traj_future"],
    components_prompt=["traj_future"],
    label_components=["traj_future"],
    include_camera_ids=True, include_frame_nums=True,
    generation_mode=True,
)

# Histórico sintético: 16 passos de 0.1s em linha reta a ASSUMED_SPEED (eixo x = frente)
t_hist = np.arange(-15, 1) * 0.1                       # -1.5s .. 0
hist_xyz = np.zeros((1, 16, 3), dtype=np.float32)
hist_xyz[0, :, 0] = ASSUMED_SPEED * t_hist
hist_rot = np.tile(np.eye(3, dtype=np.float32), (1, 16, 1, 1))
EGO_HIST_XYZ = torch.from_numpy(hist_xyz)
EGO_HIST_ROT = torch.from_numpy(hist_rot)


def predict(frames_rgb):
    """frames_rgb: lista de N_FRAMES arrays HxWx3 uint8 → pred_xyz [K, T, 3]."""
    imgs = torch.from_numpy(np.stack(frames_rgb)).permute(0, 3, 1, 2)  # [4,3,H,W]
    sample = {
        "image_frames": imgs.unsqueeze(0),                       # [1,4,3,H,W]
        "camera_indices": torch.tensor([FRONT_CAM_IDX]),
        "ego_history_xyz": EGO_HIST_XYZ.clone(),
        "ego_history_rot": EGO_HIST_ROT.clone(),
        "relative_timestamps": torch.tensor([[-0.3, -0.2, -0.1, 0.0]], dtype=torch.float32),
        "absolute_timestamps": torch.tensor([[0, 100000, 200000, 300000]], dtype=torch.int64),
    }
    sample["tokenized_data"] = preprocess_fn(data=sample)
    batch = batch_to_cuda(collate_fn_from_model_config(
        [sample], model_config=model.config, chat_template_version="r1_5"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=batch, top_p=TOP_P, temperature=TEMP,
            num_traj_samples=K_SAMPLES, num_traj_sets=1,
            max_generation_length=model.config.tokens_per_future_traj,
        )
    lp = LAST_LOGPROBS["lp"]
    best = int(np.argmax(lp)) if lp is not None else 0
    return pred_xyz[0, 0].float().cpu().numpy(), best            # [K, T, 3], idx


# ── Painel BEV ────────────────────────────────────────────────────────────────
try:
    F_HEAD = ImageFont.truetype("/usr/share/fonts/liberation-mono/LiberationMono-Bold.ttf", 22)
    F_SUB  = ImageFont.truetype("/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf", 13)
    F_TAG  = ImageFont.truetype("/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf", 12)
except OSError:
    F_HEAD = F_SUB = F_TAG = ImageFont.load_default()

DIM_COLORS = [(60, 110, 135), (65, 120, 95), (135, 110, 60),
              (130, 80, 80), (110, 85, 135), (95, 95, 95)]


def draw_panel(H, preds, best):
    panel = Image.new("RGB", (PANEL_W, H), (14, 14, 18))
    d = ImageDraw.Draw(panel)
    d.rectangle([0, 0, PANEL_W, 46], fill=(30, 40, 60))
    d.text((14, 10), "TRAJECTORY 6.4s", font=F_HEAD, fill=(140, 210, 255))
    d.text((14, 52), f"KD student 2B | K={K_SAMPLES} samples | constrained",
           font=F_SUB, fill=(170, 170, 170))
    d.text((14, 70), f"WHITE = most likely (logprob) | {ASSUMED_SPEED:.0f} m/s | 1 cam",
           font=F_SUB, fill=(230, 230, 230))

    top, bot = 96, H - 30
    bev_h = bot - top
    cx = PANEL_W // 2

    def to_px(x_fwd, y_lat):
        px = cx - (y_lat / BEV_LAT) * (PANEL_W / 2 - 20)
        py = bot - (x_fwd / BEV_FWD) * bev_h
        return px, py

    for gx in range(0, int(BEV_FWD) + 1, 10):                     # grid horizontal
        _, py = to_px(gx, 0)
        d.line([(16, py), (PANEL_W - 16, py)], fill=(38, 38, 46), width=1)
        d.text((PANEL_W - 46, py - 14), f"{gx}m", font=F_TAG, fill=(90, 90, 100))
    d.line([to_px(0, 0), to_px(BEV_FWD, 0)], fill=(38, 38, 46), width=1)

    if preds is not None:
        order = [k for k in range(preds.shape[0]) if k != best] + [best]
        for k in order:                                           # best por último (por cima)
            pts = [to_px(float(p[0]), float(p[1])) for p in preds[k]
                   if 0.0 <= p[0] <= BEV_FWD and abs(p[1]) <= BEV_LAT]
            if len(pts) >= 2:
                if k == best:
                    d.line(pts, fill=(255, 255, 255), width=5)
                else:
                    d.line(pts, fill=DIM_COLORS[k % len(DIM_COLORS)], width=2)

    ex, ey = to_px(0, 0)                                          # ego
    d.polygon([(ex, ey - 12), (ex - 8, ey + 4), (ex + 8, ey + 4)], fill=(255, 255, 255))
    return panel


print(f"[2] Video: {VIDEO_IN}")
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
gap = max(1, round(0.1 * fps))          # ~0.1s entre os 4 frames de entrada
pred_every = max(1, round(PRED_EVERY_S * fps))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PANEL_W, H))

buf, preds, best, n_pred = [], None, 0, 0
t0 = time.time()
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(buf) > (N_FRAMES - 1) * gap + 1:
        buf.pop(0)

    if idx % pred_every == 0 and len(buf) >= (N_FRAMES - 1) * gap + 1:
        frames_in = [buf[i] for i in range(0, (N_FRAMES - 1) * gap + 1, gap)][:N_FRAMES]
        preds, best = predict(frames_in)
        n_pred += 1
        d0 = np.linalg.norm(preds[:, -1, :2], axis=1)
        print(f"  t={idx/fps:5.1f}s pred#{n_pred}: dist final 6.4s = "
              f"{d0.min():.1f}-{d0.max():.1f} m | best=#{best} ({d0[best]:.1f} m)", flush=True)

    panel = draw_panel(H, preds, best)
    out = np.concatenate([frame, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)], axis=1)
    writer.write(out)
    idx += 1

cap.release()
writer.release()
print(f"Done! {idx} frames, {n_pred} predicoes, {time.time()-t0:.0f}s -> {VIDEO_OUT}")
