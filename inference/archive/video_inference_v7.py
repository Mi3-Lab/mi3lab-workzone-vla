"""Video inference v7 — Alpamayo 2B: classificação + trajetória no mesmo modelo.

O modelo 2B (TrainableReasoningVLA) já foi ensinado a:
  1. Responder Q&A de work zone (fine-tune ROADWork Stage 1 + KD Stage 3)
  2. Gerar tokens de trajetória diretamente via VLM (sample_trajectories_from_data)
     — decodificados pelo traj_tokenizer em coordenadas XY em metros (BEV).

Não é necessário carregar o modelo 10B. Um único modelo 2B faz tudo.

APPROACHING fix: cheque bilateral distingue
  "lane blocked ahead, veiculo ainda fora" (APPROACHING)
  de "markers em ambos os lados, veiculo dentro" (INSIDE).
"""
import os, sys, textwrap
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoProcessor
import glob

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

VIDEO_IN    = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/videos/boston.mp4"
VIDEO_OUT   = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/outputs/v7/v7_boston.mp4"
CKPT        = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"
STRIDE      = int(sys.argv[4]) if len(sys.argv) > 4 else 3
STATE_EVERY = 5    # state query every Nth description step
TRAJ_EVERY  = 30   # trajectory update every N frames (~1 s at 30 fps)
NUM_HIST    = 16   # history timesteps — DeltaTrajectoryTokenizer produz T×3=48 tokens

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"

QUESTION_DESC    = "Describe the work zone elements visible in this scene."
QUESTION_TRAFFIC = "How is traffic flow affected in this scene?"
QUESTION_ACTIVE  = "Is this an active work zone with workers present, or a passive zone?"

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {os.path.basename(CKPT)}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Carregar modelo 2B (único modelo — classification + trajectory) ─────────────
from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo1_5 import helper as alp_helper
from workzone_state import WZState

print("\nLoading Alpamayo 2B student (classification + trajectory)...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
ckpt_file = os.path.join(CKPT, "model.safetensors")
if os.path.exists(ckpt_file):
    sd = load_file(ckpt_file, device="cpu")
else:
    sd = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        sd.update(load_file(f, device="cpu"))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
model = model.to(torch.bfloat16).cuda().eval()

# Processor para Q&A (formato simples: image + question)
proc_qa = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)

# Processor para trajetória (formato Alpamayo: traj_history_start + frame tensor)
# Usa a base Qwen3-VL-2B-Instruct com o tokenizer do modelo (que tem tokens especiais)
proc_traj = alp_helper.get_processor(model.tokenizer)

# Ego history: velocidade forward constante ~8 m/s para estimate_t0_states extrair v0≠0.
# DeltaTrajectoryTokenizer.encode(fut_xyz=[1,16,3]) → 16×3=48 tokens ✓
# UnicycleAccelCurvatureActionSpace.estimate_t0_states: dxy=[0.8,...] → v0≈8 m/s
_dt_ego  = 0.1
_v_ego   = 8.0  # m/s (velocidade urbana típica)
_t_steps = torch.arange(NUM_HIST, dtype=torch.float32) * _dt_ego  # [0, 0.1, ..., 1.5]
_ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
_ego_xyz[0, 0, :, 0] = _t_steps * _v_ego  # x=forward: [0, 0.8, ..., 12.0] m
_ego_rot = (torch.eye(3, dtype=torch.float32, device="cuda")
            .view(1, 1, 1, 3, 3).expand(1, 1, NUM_HIST, 3, 3).contiguous())

print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("Model ready!\n")


# ── Q&A helpers ────────────────────────────────────────────────────────────────

def run_qa(pil_image, question, max_new_tokens=120):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": question},
    ]}]
    text   = proc_qa.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc_qa(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=proc_qa.tokenizer.eos_token_id,
            return_dict_in_generate=False, output_logits=False)
    n = inputs["input_ids"].shape[1]
    return proc_qa.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def parse_traffic(answer: str) -> str:
    a = answer.lower()
    if "fully blocked" in a:     return "FULLY_BLOCKED"
    if "partially blocked" in a: return "PARTIALLY_BLOCKED"
    if "lane shift" in a:        return "LANE_SHIFT"
    if "no work zone" in a or "no activity" in a or "clear" in a or "no road work" in a:
        return "NONE"
    return "UNKNOWN"


def parse_active_passive(answer: str) -> str:
    a = answer.lower()
    if "active" in a and ("worker" in a or "workers" in a):
        return "ACTIVE"
    if "passive" in a or "devices only" in a:
        return "PASSIVE"
    if "no work zone" in a or "no activity" in a or "no road work" in a:
        return "NONE"
    return "UNKNOWN"


def has_bilateral(desc: str) -> bool:
    """True quando a descrição menciona elementos de work zone em AMBOS os lados."""
    d = desc.lower()
    left  = any(k in d for k in ["left side", "left lane", "left sidewalk", "left shoulder",
                                   "left of the road", "both sides"])
    right = any(k in d for k in ["right side", "right lane", "right sidewalk", "right shoulder",
                                   "right of the road", "both sides"])
    return left and right


def classify_state(traffic: str, active_passive: str, desc: str = "") -> WZState:
    """Classificação: ACTIVE workers = INSIDE (máxima prioridade, independente de bilateral)."""
    # Workers ativos = definitivamente INSIDE, seja qual for o tráfego ou bilateral
    if active_passive == "ACTIVE":
        return WZState.INSIDE
    bilateral = has_bilateral(desc)
    if traffic == "FULLY_BLOCKED":
        return WZState.INSIDE
    if traffic in ("PARTIALLY_BLOCKED", "LANE_SHIFT") and bilateral:
        return WZState.INSIDE
    if traffic == "LANE_SHIFT":
        return WZState.INSIDE
    if traffic == "PARTIALLY_BLOCKED" and not bilateral:
        return WZState.APPROACHING
    if active_passive == "PASSIVE":
        return WZState.APPROACHING
    if active_passive == "NONE" and traffic == "NONE":
        return WZState.OUTSIDE
    return None


class DirectStateMachine:
    CONFIRM_NEEDED = {
        WZState.OUTSIDE: 3, WZState.APPROACHING: 2,
        WZState.INSIDE:  2, WZState.EXITING:     5,
    }
    # Transições válidas: uma vez INSIDE só pode ir para EXITING (nunca voltar a APPROACHING)
    REMAP = {
        (WZState.INSIDE,  WZState.APPROACHING): WZState.EXITING,
        (WZState.INSIDE,  WZState.OUTSIDE):     WZState.EXITING,
        (WZState.EXITING, WZState.APPROACHING): WZState.EXITING,
    }

    def __init__(self):
        self.state     = WZState.OUTSIDE
        self.candidate = WZState.OUTSIDE
        self.count     = 0

    def update(self, new_state) -> WZState:
        if new_state is None:
            return self.state
        # Remapear transições fisicamente inválidas
        new_state = self.REMAP.get((self.state, new_state), new_state)
        if new_state == self.candidate:
            self.count += 1
        else:
            self.candidate = new_state
            self.count     = 1
        if self.candidate != self.state and self.count >= self.CONFIRM_NEEDED[self.candidate]:
            prev       = self.state
            self.state = self.candidate
            self.count = 0
            print(f"    [TRANSITION] {prev.value.upper()} → {self.state.value.upper()}")
        return self.state


# ── Trajetória via 2B ───────────────────────────────────────────────────────────

def get_trajectory_2b(pil_image, W, H):
    """Gera trajetória via model.sample_trajectories_from_data (mesmo 2B usado p/ classificação).

    Usa o formato Alpamayo: frame uint8 tensor + traj_history_start tokens + ego history zeros.
    pred_xyz[0,0,0,:,:2]: (T, 2) metros — x=lateral, y=forward.
    Projetado com câmera pinhole: h=1.5m, f=500px.
    """
    # PIL → tensor uint8 (1, 3, H, W) — mesmo formato do load_physical_aiavdataset
    frame_np  = np.array(pil_image)
    frame_t   = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0)

    # Mensagem formato Alpamayo (trajectory prompt, não Q&A)
    messages = alp_helper.create_message(frames=frame_t, camera_indices=None)
    inputs   = proc_traj.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": _ego_xyz.clone(),
        "ego_history_rot": _ego_rot.clone(),
    }

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=model_inputs,
            num_traj_samples=1,
            max_generation_length=200,   # >= tokens_per_future_traj=128
        )

    # pred_xyz: [1, 1, 1, T_fut, 3]  (B, n_sets, K, T, 3)
    # Convenção UnicycleAccelCurvature: x=forward, y=lateral(left+), z=up
    xyz = pred_xyz[0, 0, 0, :, :].cpu().float().numpy()   # [T, 3] metros

    # Quality check: modelo treinado em AIAV (OOD para dashcam) gera curvatura excessiva.
    # Se o deslocamento lateral final > deslocamento forward, ou forward < 3m → fallback.
    x_end, y_end = float(xyz[-1, 0]), float(xyz[-1, 1])
    forward_ok = x_end >= 3.0 and abs(y_end) <= x_end
    if not forward_ok:
        # Fallback: trajetória reta para frente a ~8 m/s (placeholder até Stage 2)
        n = 30
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = np.linspace(2.0, 25.0, n)   # x=forward, y=0 (reta)
        print(f"    [TRAJ] fallback straight-ahead (x_end={x_end:.1f}m y_end={y_end:.1f}m)")
        return project_bev_to_pixel(xyz, W, H), True

    return project_bev_to_pixel(xyz, W, H), False


def project_bev_to_pixel(xyz_m, W, H, f=300.0, h=1.2):
    """Projeta trajetória BEV (metros) para pixels de câmera frontal dashcam.
    UnicycleAccelCurvature: xyz[0]=forward(x), xyz[1]=lateral(y, left+), xyz[2]=up.
    Dashcam FOV ~120° → f≈300px, h≈1.2m.
    Pinhole: u = cx - f*y_lat/x_fwd
             v = cy + f*h / x_fwd
    """
    cx, cy = W / 2.0, H * 0.5
    pts = []
    for row in xyz_m:
        x_fwd, y_lat = float(row[0]), float(row[1])
        if x_fwd < 1.0:   # pontos muito próximos projetam abaixo da imagem
            continue
        u = int(cx - f * y_lat / x_fwd)
        v = int(cy + f * h    / x_fwd)
        if 0 <= u < W and 0 <= v < H:
            pts.append((u, v))
    return pts


def draw_trajectory(frame_bgr, waypoints, state):
    colors = {
        WZState.OUTSIDE:     (100, 200, 100),
        WZState.APPROACHING: (50,  200, 255),
        WZState.INSIDE:      (50,   80, 255),
        WZState.EXITING:     (100, 200, 100),
    }
    color = colors.get(state, (200, 200, 200))
    if len(waypoints) < 2:
        return frame_bgr
    overlay = frame_bgr.copy()
    for i in range(len(waypoints) - 1):
        cv2.line(overlay, waypoints[i], waypoints[i + 1], color, 2)
    cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)
    for i, (x, y) in enumerate(waypoints):
        cv2.circle(frame_bgr, (x, y), max(3, 6 - i // 4), color, -1)
    return frame_bgr


# ── Video I/O ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)
cap      = cv2.VideoCapture(VIDEO_IN)
fps      = cap.get(cv2.CAP_PROP_FPS)
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

PANEL_W = 680
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W + PANEL_W, H))

try:
    font    = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 18)
    font_sm = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 14)
    font_lg = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",    22)
except Exception:
    font = font_sm = font_lg = ImageFont.load_default()

STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40,  40,  40),  "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80,  60,   0),  "fg": (255, 200,  50), "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80,  10,  10),  "fg": (255,  80,  80), "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10,  60,  10),  "fg": (80,  230,  80), "label": "EXITING"},
}

print(f"Processing {total_fr} frames @ {fps:.0f}fps ({total_fr/fps:.0f}s)\n")

sm            = DirectStateMachine()
current_text  = "Analyzing..."
current_state = WZState.OUTSIDE
traffic_ans   = "—"
active_ans    = "—"
waypoints       = []
traj_is_fallback = False
desc_step     = 0
frame_idx     = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sec = frame_idx / fps
    pil = None

    # Trajetória: atualiza a cada TRAJ_EVERY frames
    if frame_idx % TRAJ_EVERY == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"  t={sec:.1f}s [TRAJ] gerando via 2B...")
        try:
            waypoints, traj_is_fallback = get_trajectory_2b(pil, W, H)
            print(f"    → {len(waypoints)} waypoints projetados {'[fallback]' if traj_is_fallback else '[model]'}")
        except Exception as e:
            print(f"    [WARN] trajetória falhou: {e}")

    # Classificação: a cada STRIDE frames
    if frame_idx % STRIDE == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        desc         = run_qa(pil, QUESTION_DESC, max_new_tokens=80)
        current_text = desc

        if desc_step % STATE_EVERY == 0:
            traffic_ans  = run_qa(pil, QUESTION_TRAFFIC, max_new_tokens=60)
            active_ans   = run_qa(pil, QUESTION_ACTIVE,  max_new_tokens=40)
            traffic_tag  = parse_traffic(traffic_ans)
            active_tag   = parse_active_passive(active_ans)
            raw_state    = classify_state(traffic_tag, active_tag, desc)
            current_state = sm.update(raw_state)
            bilat  = has_bilateral(desc)
            print(f"  t={sec:.1f}s [Q] | {current_state.value.upper()} | "
                  f"traffic={traffic_tag} active={active_tag} bilateral={bilat}")
            print(f"    T: {traffic_ans[:80]}")
            print(f"    A: {active_ans[:80]}")
        else:
            current_state = sm.state

        print(f"  t={sec:.1f}s     | {current_state.value.upper()} | {desc[:80]}")
        desc_step += 1

    # Overlay trajetória no frame
    frame_out = draw_trajectory(frame.copy(), waypoints, current_state)

    # ── Painel lateral ─────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v7 — WZ State + Trajectory",
              font=font,    fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}",
              font=font_sm, fill=(120, 160, 200))

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 105], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"],       font=font_lg, fill=style["fg"])

    y = 112
    draw.text((10, y), "TRAFFIC:", font=font_sm, fill=(180, 220, 255))
    y += 16
    for line in textwrap.wrap(traffic_ans[:120], width=46)[:2]:
        draw.text((10, y), line, font=font_sm, fill=(200, 200, 180))
        y += 15
    y += 4

    draw.text((10, y), "ZONE TYPE:", font=font_sm, fill=(180, 220, 255))
    y += 16
    for line in textwrap.wrap(active_ans[:120], width=46)[:2]:
        draw.text((10, y), line, font=font_sm, fill=(200, 200, 180))
        y += 15
    y += 8

    traj_src   = "straight [Stage 2 pending]" if traj_is_fallback else "Alpamayo 2B"
    traj_label = f"TRAJECTORY: {traj_src} ({len(waypoints)} pts)"
    traj_color = (180, 180, 80) if traj_is_fallback else (100, 255, 140)
    draw.text((10, y), traj_label, font=font_sm, fill=traj_color)
    y += 20

    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 16
    for line in textwrap.wrap(current_text, width=46)[:11]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 15
        if y > H - 25:
            break

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 20, PANEL_W, H],                 fill=(20, 20, 20))
    draw.rectangle([0, H - 20, int(PANEL_W * progress), H], fill=(50, 100, 180))
    draw.text((5, H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s",
              font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    out_vid.write(np.hstack([frame_out, panel_bgr]))
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Saved: {VIDEO_OUT}  ({frame_idx} frames)")
