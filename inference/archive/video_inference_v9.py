"""Video inference v9 — estado de work zone 100% probabilístico (sem regras).

Substitui o pipeline v7/v8 (Q&A texto livre → regex → if/else → state machine
com contadores e REMAP) por:

  1. SCORING de respostas candidatas via logprob (teacher forcing) nas perguntas
     que o modelo FOI treinado (vocabulário ROADWork) — distribuições calibradas,
     não string matching.
  2. Pergunta direta de estado (zero-shot, validada no job 170269) como sensor
     adicional, com sharpening.
  3. Fusão temporal por filtro Bayesiano (HMM forward): a física está na matriz
     de transição (INSIDE→APPROACHING = prob 0), a suavização é inferência.

As matrizes de emissão P(resposta|estado) são parâmetros probabilísticos que o
Stage 2 pode aprender dos dados ROADWork; hoje são priors de engenharia.

Trajetória: modelo 2B com fallback reto quando OOD (até Stage 2).
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

VIDEO_IN   = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/videos/boston.mp4"
VIDEO_OUT  = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/outputs/v9/v9_boston.mp4"
CKPT       = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"

STATE_EVERY = 9    # atualização de crença a cada N frames (0.3 s @ 30fps)
DESC_EVERY  = 15   # descrição livre (painel) a cada N frames
TRAJ_EVERY  = 30   # trajetória a cada N frames
NUM_HIST    = 16   # DeltaTrajectoryTokenizer: 16×3 = 48 tokens

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {os.path.basename(CKPT)}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo1_5 import helper as alp_helper
from workzone_state import WZState

print("\nLoading Alpamayo 2B (probabilistic state + trajectory)...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()

proc_qa   = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
proc_traj = alp_helper.get_processor(model.tokenizer)

# Ego history forward 8 m/s (v0 para o unicycle action space)
_t_steps = torch.arange(NUM_HIST, dtype=torch.float32) * 0.1
_ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
_ego_xyz[0, 0, :, 0] = _t_steps * 8.0
_ego_rot = (torch.eye(3, dtype=torch.float32, device="cuda")
            .view(1, 1, 1, 3, 3).expand(1, 1, NUM_HIST, 3, 3).contiguous())

print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("Model ready!\n")


# ══ Sensores probabilísticos ═════════════════════════════════════════════════
STATES      = [WZState.OUTSIDE, WZState.APPROACHING, WZState.INSIDE, WZState.EXITING]
STATE_NAMES = ["OUTSIDE", "APPROACHING", "INSIDE", "EXITING"]

# Perguntas do fine-tune ROADWork + candidatos com o vocabulário EXATO do treino
Q_TRAFFIC = "How is traffic flow affected in this scene?"
TRAFFIC_CANDS = [
    "Lane partially blocked. One or more lanes are obstructed",
    "Lane shift in effect. Traffic is redirected",
    "Lane or road fully blocked. Traffic cannot proceed",
    "No lane alteration detected. Road appears clear",
]
Q_ACTIVE = "Is this an active work zone with workers present, or a passive zone?"
ACTIVE_CANDS = [
    "Active work zone: workers are present.",
    "Passive work zone: no workers visible, only control devices.",
]
# Pergunta direta de estado (zero-shot — sinal fraco mas informativo, job 170269)
Q_STATE = (
    "This is a dashcam view from a driving vehicle. Regarding the road work zone, "
    "what is the vehicle's current status? Answer with exactly one word: "
    "OUTSIDE (no work zone nearby), APPROACHING (work zone visible ahead), "
    "INSIDE (currently passing through the work zone), or EXITING (leaving the work zone)."
)
STATE_CANDS = STATE_NAMES
BETA_STATE  = 2.0   # sharpening do sensor zero-shot (distribuições quase uniformes)

# Matrizes de emissão P(resposta | estado) — priors; Stage 2 pode aprendê-las.
# EGO-CÊNTRICO: perguntas de CENA ("há workers?") disparam com a zona visível à
# frente — não discriminam APPROACHING de INSIDE. Suas linhas APP/INS/EXI são
# quase iguais; só separam "zona presente" de "zona ausente" (OUTSIDE).
#                     partial  shift   full   none
TRAFFIC_EMIT = np.array([
    [0.10, 0.03, 0.02, 0.85],   # OUTSIDE
    [0.50, 0.14, 0.13, 0.23],   # APPROACHING
    [0.52, 0.16, 0.14, 0.18],   # INSIDE  (≈ APPROACHING: sensor de cena)
    [0.30, 0.06, 0.05, 0.59],   # EXITING
])
#                     active  passive
ACTIVE_EMIT = np.array([
    [0.10, 0.90],   # OUTSIDE
    [0.45, 0.55],   # APPROACHING (workers visíveis à frente ≠ estar dentro)
    [0.50, 0.50],   # INSIDE  (≈ APPROACHING: sensor de cena)
    [0.25, 0.75],   # EXITING
])

# HMM: transição por passo de 0.3 s. INSIDE→APPROACHING/OUTSIDE = 0 (física).
# APP→INS lento: a fase de aproximação dura ~3-4 s a menos que haja evidência
# ego-cêntrica forte (hoje: só a pergunta direta; Stage 2: modelo treinado).
TRANS = np.array([
    # OUT   APP   INS   EXI
    [0.95, 0.05, 0.00, 0.00],   # OUTSIDE
    [0.02, 0.91, 0.07, 0.00],   # APPROACHING
    [0.00, 0.00, 0.96, 0.04],   # INSIDE
    [0.10, 0.02, 0.08, 0.80],   # EXITING (pode re-entrar em outra zona)
])
PRIOR = np.array([0.40, 0.25, 0.30, 0.05])  # sem viés forte: vídeo pode começar dentro


@torch.no_grad()
def candidate_probs(pil, question, candidates):
    """Distribuição sobre respostas candidatas via logprob médio por token."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": question},
    ]}]
    text = proc_qa.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    base = proc_qa(text=[text], images=[pil], return_tensors="pt")
    n = base["input_ids"].shape[1]

    scores = []
    for cand in candidates:
        cand_ids = proc_qa.tokenizer.encode(cand, add_special_tokens=False)
        ids  = torch.cat([base["input_ids"],
                          torch.tensor([cand_ids], dtype=torch.long)], dim=1).cuda()
        attn = torch.ones_like(ids)
        kw = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in base:
                kw[k] = base[k].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(**kw)
        lp = torch.log_softmax(out.logits[0, n - 1:-1].float(), dim=-1)
        tok_lp = [lp[i, t].item() for i, t in enumerate(cand_ids)]
        scores.append(sum(tok_lp) / len(tok_lp))
    s = np.array(scores)
    p = np.exp(s - s.max())
    return p / p.sum()


def state_emission(pil):
    """Verossimilhança e(s) fundindo os 3 sensores probabilísticos."""
    p_traffic = candidate_probs(pil, Q_TRAFFIC, TRAFFIC_CANDS)
    p_active  = candidate_probs(pil, Q_ACTIVE,  ACTIVE_CANDS)
    p_state   = candidate_probs(pil, Q_STATE,   STATE_CANDS)
    p_state   = p_state ** BETA_STATE
    p_state  /= p_state.sum()

    e = (TRAFFIC_EMIT @ p_traffic) * (ACTIVE_EMIT @ p_active) * p_state
    return e / e.sum(), p_traffic, p_active, p_state


class BayesFilter:
    def __init__(self):
        self.belief = PRIOR.copy()

    def update(self, emission):
        b = (self.belief @ TRANS) * emission
        self.belief = b / b.sum()
        return self.belief

    @property
    def state(self):
        return STATES[int(self.belief.argmax())]


# ══ Descrição livre (só para o painel) ═══════════════════════════════════════
Q_DESC = "Describe the work zone elements visible in this scene."

def run_qa(pil_image, question, max_new_tokens=80):
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


# ══ Trajetória (modelo + fallback reto até Stage 2) ══════════════════════════

def get_trajectory_2b(pil_image, W, H):
    frame_np = np.array(pil_image)
    frame_t  = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0)
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
            data=model_inputs, num_traj_samples=1, max_generation_length=200,
        )
    xyz = pred_xyz[0, 0, 0, :, :].cpu().float().numpy()

    x_end, y_end = float(xyz[-1, 0]), float(xyz[-1, 1])
    if x_end >= 3.0 and abs(y_end) <= x_end:
        return project_bev_to_pixel(xyz, W, H), False
    n = 30
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = np.linspace(2.0, 25.0, n)
    return project_bev_to_pixel(xyz, W, H), True


def project_bev_to_pixel(xyz_m, W, H, f=300.0, h=1.2):
    cx, cy = W / 2.0, H * 0.5
    pts = []
    for row in xyz_m:
        x_fwd, y_lat = float(row[0]), float(row[1])
        if x_fwd < 1.0:
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


# ══ Video I/O ════════════════════════════════════════════════════════════════
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
    WZState.OUTSIDE:     {"bg": (40,  40,  40), "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80,  60,   0), "fg": (255, 200,  50), "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80,  10,  10), "fg": (255,  80,  80), "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10,  60,  10), "fg": (80,  230,  80), "label": "EXITING"},
}
BAR_COLORS = [(180, 180, 180), (255, 200, 50), (255, 80, 80), (80, 230, 80)]

TRAFFIC_SHORT = ["partial block", "lane shift", "fully blocked", "clear"]
ACTIVE_SHORT  = ["ACTIVE workers", "passive devices"]

print(f"Processing {total_fr} frames @ {fps:.0f}fps ({total_fr/fps:.0f}s)\n")

bf              = BayesFilter()
belief          = PRIOR.copy()
current_state   = bf.state
current_text    = "Analyzing..."
p_traffic       = np.ones(len(TRAFFIC_CANDS)) / len(TRAFFIC_CANDS)
p_active        = np.ones(2) / 2
waypoints       = []
traj_is_fallback = False
frame_idx       = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    sec = frame_idx / fps
    pil = None

    if frame_idx % TRAJ_EVERY == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            waypoints, traj_is_fallback = get_trajectory_2b(pil, W, H)
        except Exception as e:
            print(f"    [WARN] trajetória falhou: {e}")

    if frame_idx % STATE_EVERY == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emission, p_traffic, p_active, p_state = state_emission(pil)
        prev_state = bf.state
        belief = bf.update(emission)
        current_state = bf.state
        if current_state != prev_state:
            print(f"  t={sec:5.1f}s [TRANSITION] {prev_state.value.upper()} → {current_state.value.upper()}")
        bstr = "  ".join(f"{n[:3]}={v:.2f}" for n, v in zip(STATE_NAMES, belief))
        print(f"  t={sec:5.1f}s belief: {bstr}  | traffic={TRAFFIC_SHORT[p_traffic.argmax()]}"
              f" active={ACTIVE_SHORT[p_active.argmax()]}")

    if frame_idx % DESC_EVERY == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_text = run_qa(pil, Q_DESC, max_new_tokens=80)

    frame_out = draw_trajectory(frame.copy(), waypoints, current_state)

    # ── Painel ────────────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v9 — Bayesian WZ State", font=font, fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}", font=font_sm, fill=(120, 160, 200))

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 105], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"], font=font_lg, fill=style["fg"])
    draw.text((PANEL_W - 150, 74), f"P={belief.max():.2f}", font=font, fill=style["fg"])

    # Barras de crença (belief) — o coração do v9
    y = 115
    draw.text((10, y), "STATE BELIEF (Bayes filter):", font=font_sm, fill=(180, 220, 255))
    y += 18
    bar_max = PANEL_W - 190
    for i, (name, p) in enumerate(zip(STATE_NAMES, belief)):
        draw.text((10, y), f"{name:<11s}", font=font_sm, fill=(200, 200, 200))
        draw.rectangle([120, y + 2, 120 + int(bar_max * p), y + 12], fill=BAR_COLORS[i])
        draw.text((130 + int(bar_max * p), y), f"{p:.2f}", font=font_sm, fill=(160, 160, 160))
        y += 17
    y += 6

    # Sensores (distribuições in-domain)
    draw.text((10, y), "SENSORS (logprob scoring):", font=font_sm, fill=(180, 220, 255))
    y += 17
    ti = int(p_traffic.argmax())
    ai = int(p_active.argmax())
    draw.text((10, y), f"traffic: {TRAFFIC_SHORT[ti]} ({p_traffic[ti]:.2f})",
              font=font_sm, fill=(200, 200, 180))
    y += 16
    draw.text((10, y), f"workers: {ACTIVE_SHORT[ai]} ({p_active[ai]:.2f})",
              font=font_sm, fill=(200, 200, 180))
    y += 20

    traj_src   = "straight [Stage 2 pending]" if traj_is_fallback else "Alpamayo 2B"
    traj_color = (180, 180, 80) if traj_is_fallback else (100, 255, 140)
    draw.text((10, y), f"TRAJECTORY: {traj_src} ({len(waypoints)} pts)",
              font=font_sm, fill=traj_color)
    y += 22

    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 16
    for line in textwrap.wrap(current_text, width=46)[:10]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 15
        if y > H - 25:
            break

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 20, PANEL_W, H],                 fill=(20, 20, 20))
    draw.rectangle([0, H - 20, int(PANEL_W * progress), H], fill=(50, 100, 180))
    draw.text((5, H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    out_vid.write(np.hstack([frame_out, panel_bgr]))
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Saved: {VIDEO_OUT}  ({frame_idx} frames)")
