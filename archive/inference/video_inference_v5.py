"""Video inference v5 — state classification via direct model query (no keyword heuristics)."""
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
sys.path.insert(0, os.path.join(BASE, "mi3lab-workzone-vla"))   # workzone_state.py mora aqui
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
})

VIDEO_IN  = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/videos/boston.mp4"
VIDEO_OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/outputs/v5/v5_boston.mp4"
CKPT      = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"
STRIDE    = int(sys.argv[4]) if len(sys.argv) > 4 else 3   # description every Nth frame
STATE_EVERY = 5   # run state query every Nth description (= every STRIDE*STATE_EVERY frames)

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"

QUESTION_DESC  = "Describe the work zone elements visible in this scene."
QUESTION_STATE = (
    "Look at this driving scene. Classify the vehicle's relationship to any work zone "
    "using exactly one word from this list: OUTSIDE, APPROACHING, INSIDE, EXITING.\n"
    "- OUTSIDE: no work zone visible or relevant\n"
    "- APPROACHING: work zone signs/elements visible ahead, not yet inside\n"
    "- INSIDE: vehicle is actively inside the work zone\n"
    "- EXITING: vehicle has passed through and is leaving\n"
    "Answer with exactly one word."
)

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {os.path.basename(CKPT)}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"State query every {STRIDE * STATE_EVERY} frames ({STATE_EVERY} desc steps)")

from alpamayo1_5_sft.models.kd_model import build_student_model
from workzone_state import WZState, parse_model_state

print("\nLoading Alpamayo 2B v4 student model...")
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
processor = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print("Model ready!\n")

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(pil_image, question, max_new_tokens=120):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id)
    n = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()

# ── Simple hysteresis state machine (direct classification) ───────────────────
class DirectStateMachine:
    """Requires N consecutive same answers before transitioning."""
    CONFIRM_NEEDED = {
        WZState.OUTSIDE:     3,
        WZState.APPROACHING: 2,
        WZState.INSIDE:      2,
        WZState.EXITING:     2,
    }

    def __init__(self):
        self.state     = WZState.OUTSIDE
        self.candidate = WZState.OUTSIDE
        self.count     = 0

    def update(self, raw_answer: str) -> WZState:
        parsed = parse_model_state(raw_answer)
        if parsed is None:
            return self.state  # unrecognized → hold

        if parsed == self.candidate:
            self.count += 1
        else:
            self.candidate = parsed
            self.count     = 1

        if self.candidate != self.state and self.count >= self.CONFIRM_NEEDED[self.candidate]:
            self.state = self.candidate
            self.count = 0

        return self.state

# ── Video setup ───────────────────────────────────────────────────────────────
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
last_state_answer = "OUTSIDE"
desc_step     = 0
frame_idx     = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sec = frame_idx / fps

    if frame_idx % STRIDE == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Description query (every STRIDE frames)
        desc = run_inference(pil, QUESTION_DESC, max_new_tokens=120)
        current_text = desc

        # State query (every STATE_EVERY description steps)
        if desc_step % STATE_EVERY == 0:
            state_ans = run_inference(pil, QUESTION_STATE, max_new_tokens=10)
            last_state_answer = state_ans.strip().upper().split()[0] if state_ans.strip() else "OUTSIDE"
            current_state = sm.update(last_state_answer)
            marker = "[Q]"  # state query ran this frame
        else:
            current_state = sm.state
            marker = "   "

        print(f"  t={sec:.1f}s {marker} | {current_state.value.upper()} | model={last_state_answer}")
        print(f"    {desc[:120]}")
        desc_step += 1

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v5 — Direct Classification", font=font,    fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}",                     font=font_sm, fill=(120, 160, 200))

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 105], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"],       font=font_lg, fill=style["fg"])
    draw.text((10, 98), f"model said: {last_state_answer}", font=font_sm, fill=(180, 180, 180))

    y = 115
    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 18
    for line in textwrap.wrap(current_text, width=46)[:18]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 17
        if y > H - 25:
            break

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 20, PANEL_W, H], fill=(20, 20, 20))
    draw.rectangle([0, H - 20, int(PANEL_W * progress), H], fill=(50, 100, 180))
    draw.text((5, H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    out_vid.write(np.hstack([frame, panel_bgr]))
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Saved: {VIDEO_OUT}  ({frame_idx} frames)")
