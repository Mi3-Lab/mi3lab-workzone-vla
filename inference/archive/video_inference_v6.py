"""Video inference v6 — ROADWork Q&A classification + optical flow trajectory.

State: same as v5c (ROADWork Q&A: traffic flow + active/passive).
Trajectory: estimated from dense optical flow between consecutive frames.
  - Projects 20 waypoints from vehicle position toward road ahead
  - Accounts for vehicle turning via lateral flow component
  - Placeholder until Stage 2 (expert diffusion head SFT) is complete

After Stage 2, trajectory will come from the Alpamayo expert module directly.
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
})

VIDEO_IN  = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/videos/boston.mp4"
VIDEO_OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/outputs/v6/v6_boston.mp4"
CKPT      = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"
STRIDE    = int(sys.argv[4]) if len(sys.argv) > 4 else 3
STATE_EVERY = 5

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"

QUESTION_DESC    = "Describe the work zone elements visible in this scene."
QUESTION_TRAFFIC = "How is traffic flow affected in this scene?"
QUESTION_ACTIVE  = "Is this an active work zone with workers present, or a passive zone?"

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {os.path.basename(CKPT)}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

from alpamayo1_5_sft.models.kd_model import build_student_model
from workzone_state import WZState

print("\nLoading Alpamayo 2B student model...")
model_obj = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
ckpt_file = os.path.join(CKPT, "model.safetensors")
if os.path.exists(ckpt_file):
    sd = load_file(ckpt_file, device="cpu")
else:
    sd = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        sd.update(load_file(f, device="cpu"))
missing, unexpected = model_obj.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
model_obj = model_obj.to(torch.bfloat16).cuda().eval()
processor = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print("Model ready!\n")


def run_inference(pil_image, question, max_new_tokens=120):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model_obj.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id)
    n = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def parse_traffic(answer: str) -> str:
    a = answer.lower()
    if "fully blocked" in a:     return "FULLY_BLOCKED"
    if "partially blocked" in a: return "PARTIALLY_BLOCKED"
    if "lane shift" in a:        return "LANE_SHIFT"
    if "no work zone" in a or "no activity" in a or "no road work" in a or "clear" in a:
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


def classify_state(traffic: str, active_passive: str):
    if traffic in ("FULLY_BLOCKED", "PARTIALLY_BLOCKED", "LANE_SHIFT"):
        return WZState.INSIDE
    if active_passive == "ACTIVE":
        return WZState.INSIDE
    if active_passive == "PASSIVE" and traffic in ("NONE", "UNKNOWN"):
        return WZState.APPROACHING
    if active_passive == "NONE" and traffic == "NONE":
        return WZState.OUTSIDE
    return None


class DirectStateMachine:
    CONFIRM_NEEDED = {
        WZState.OUTSIDE: 3, WZState.APPROACHING: 2,
        WZState.INSIDE: 2,  WZState.EXITING: 2,
    }

    def __init__(self):
        self.state     = WZState.OUTSIDE
        self.candidate = WZState.OUTSIDE
        self.count     = 0

    def update(self, new_state) -> WZState:
        if new_state is None:
            return self.state
        if new_state == self.candidate:
            self.count += 1
        else:
            self.candidate = new_state
            self.count     = 1
        if self.candidate != self.state and self.count >= self.CONFIRM_NEEDED[self.candidate]:
            prev = self.state
            self.state = self.candidate
            self.count = 0
            print(f"    [TRANSITION] {prev.value.upper()} → {self.state.value.upper()}")
        return self.state


# ── Optical flow trajectory ────────────────────────────────────────────────────

def estimate_trajectory(prev_gray, curr_gray, H, W, n_waypoints=20, state=WZState.OUTSIDE):
    """Estimate forward trajectory using dense optical flow.

    Computes background motion from road surface region.
    Projects N waypoints from vehicle position toward road ahead.
    Accounts for lateral vehicle motion (turning).
    """
    if prev_gray is None:
        return _straight_trajectory(H, W, n_waypoints)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Sample flow from road surface region (lower 40% of image, excluding very bottom)
    y1, y2 = int(H * 0.55), int(H * 0.90)
    road_flow = flow[y1:y2, :, :]
    dx = float(np.median(road_flow[:, :, 0]))  # horizontal: negative = moving right
    dy = float(np.median(road_flow[:, :, 1]))  # vertical: negative = moving forward

    # Lateral offset per step: inverse of background motion
    # If background flows right (dx>0), vehicle is drifting left → trajectory curves left
    lateral_per_step = -dx * 0.3  # damping factor

    # Forward speed estimation: how much background moves up per frame
    forward_speed = max(abs(dy), 1.0)

    waypoints = []
    cx = W / 2.0

    for i in range(1, n_waypoints + 1):
        # Non-linear spacing: closer waypoints dense, farther waypoints spread
        t = i / n_waypoints
        # y: start near bottom, converge toward horizon (H*0.35)
        y = H - 10 - t * (H * 0.55)
        # x: center + accumulated lateral drift (perspective-scaled: less drift at horizon)
        x = cx + lateral_per_step * i * (1 - t * 0.7)
        waypoints.append((int(np.clip(x, 5, W - 5)), int(np.clip(y, 5, H - 5))))

    return waypoints


def _straight_trajectory(H, W, n_waypoints=20):
    """Fallback: straight ahead toward image center."""
    cx = W / 2
    return [
        (int(cx), int(H - 10 - i * (H * 0.55) / n_waypoints))
        for i in range(1, n_waypoints + 1)
    ]


def draw_trajectory(frame_bgr, waypoints, state):
    """Overlay trajectory dots and connecting line on frame."""
    state_colors = {
        WZState.OUTSIDE:     (100, 200, 100),   # green
        WZState.APPROACHING: (50,  200, 255),   # yellow-ish
        WZState.INSIDE:      (50,   80, 255),   # red
        WZState.EXITING:     (100, 200, 100),   # green
    }
    color = state_colors.get(state, (200, 200, 200))

    if len(waypoints) < 2:
        return frame_bgr

    overlay = frame_bgr.copy()
    for i in range(len(waypoints) - 1):
        cv2.line(overlay, waypoints[i], waypoints[i + 1], color, 2)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Draw dots at each waypoint (larger for closer ones)
    for i, (x, y) in enumerate(waypoints):
        radius = max(3, 6 - i // 4)
        cv2.circle(frame_bgr, (x, y), radius, color, -1)

    return frame_bgr


# ── Video setup ────────────────────────────────────────────────────────────────
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
waypoints     = []
prev_gray     = None
desc_step     = 0
frame_idx     = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sec = frame_idx / fps

    # Optical flow trajectory estimation every frame
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    waypoints = estimate_trajectory(prev_gray, curr_gray, H, W, n_waypoints=20, state=current_state)
    prev_gray = curr_gray

    # Overlay trajectory on frame
    frame_with_traj = draw_trajectory(frame.copy(), waypoints, current_state)

    if frame_idx % STRIDE == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        desc = run_inference(pil, QUESTION_DESC, max_new_tokens=80)
        current_text = desc

        if desc_step % STATE_EVERY == 0:
            traffic_ans = run_inference(pil, QUESTION_TRAFFIC,  max_new_tokens=60)
            active_ans  = run_inference(pil, QUESTION_ACTIVE,   max_new_tokens=40)
            traffic_tag = parse_traffic(traffic_ans)
            active_tag  = parse_active_passive(active_ans)
            raw_state   = classify_state(traffic_tag, active_tag)
            current_state = sm.update(raw_state)
            marker = "[Q]"
            print(f"  t={sec:.1f}s {marker} | {current_state.value.upper()} | traffic={traffic_tag} active={active_tag}")
        else:
            current_state = sm.state
            marker = "   "

        print(f"  t={sec:.1f}s {marker} | {current_state.value.upper()} | {desc[:80]}")
        desc_step += 1

    # ── Panel ──────────────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v6 — State + Trajectory", font=font,    fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}",                   font=font_sm, fill=(120, 160, 200))

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

    draw.text((10, y), "TRAJECTORY: optical flow (Stage 2 pending)", font=font_sm, fill=(100, 180, 100))
    y += 20

    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 16
    for line in textwrap.wrap(current_text, width=46)[:11]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 15
        if y > H - 25:
            break

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 20, PANEL_W, H], fill=(20, 20, 20))
    draw.rectangle([0, H - 20, int(PANEL_W * progress), H], fill=(50, 100, 180))
    draw.text((5, H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    out_vid.write(np.hstack([frame_with_traj, panel_bgr]))
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Saved: {VIDEO_OUT}  ({frame_idx} frames)")
