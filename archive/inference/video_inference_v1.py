"""Annotated video inference — runs model on every frame and overlays reasoning + work zone state."""
import os, sys, glob, textwrap
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
sys.path.insert(0, os.path.join(BASE, "mi3lab-workzone-vla"))   # workzone_state.py mora aqui
os.environ["WANDB_DISABLED"]        = "true"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VIDEO_IN  = sys.argv[1] if len(sys.argv) > 1 else \
    f"{BASE}/data/roadwork/videos/boston_042e1caf93114d3286c11ba14ddaa759_000001_00300_snippet.mp4"
VIDEO_OUT = sys.argv[2] if len(sys.argv) > 2 else \
    f"{BASE}/logs/alpamayo_video_reasoning.mp4"
CKPT      = sys.argv[3] if len(sys.argv) > 3 else \
    f"{BASE}/checkpoints/sft_stage1_roadwork_v2/checkpoint-4500"
A1_CKPT   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
VLM_PATH  = f"{BASE}/models/Cosmos-Reason2-8B"

QUESTION_DESC  = "Describe the work zone elements visible in this scene."
FRAME_STRIDE   = int(sys.argv[4]) if len(sys.argv) > 4 else 3  # run model every Nth frame

print(f"Video   : {os.path.basename(VIDEO_IN)}")
print(f"Output  : {VIDEO_OUT}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")

# ── Load model ────────────────────────────────────────────────────────────────
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoProcessor
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from workzone_state import WorkZoneStateMachine, WZState, keyword_score

print("\nLoading model (one time)...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=A1_CKPT, vlm_name_or_path=VLM_PATH)
shards = sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors")))
sd = {}
for f in shards:
    sd.update(load_file(f, device="cpu"))
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()
processor = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)
print("Model ready!\n")

# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(pil_image, question, max_new_tokens=80):
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

# ── Video setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN)
fps       = cap.get(cv2.CAP_PROP_FPS)
total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

PANEL_W   = 680
OUT_W     = W + PANEL_W
OUT_H     = H

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (OUT_W, OUT_H))

try:
    font    = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 18)
    font_sm = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 14)
    font_lg = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",    22)
except Exception:
    font = font_sm = font_lg = ImageFont.load_default()

# ── State colors & labels ─────────────────────────────────────────────────────
STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40,  40,  40),  "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80,  60,   0),  "fg": (255, 200,  50), "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80,  10,  10),  "fg": (255,  80,  80), "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10,  60,  10),  "fg": (80,  230,  80), "label": "EXITING"},
}

BAD_PATTERNS = [
    "No objects satisfy", "No relevant objects", "No specific details",
    "No work zone detected", "No Description", "No traffic lights",
    "There is one moving", "ego vehicle", "ego car",
]

def is_valid_workzone(text):
    if not text or len(text) < 20:
        return False
    if any(p in text for p in BAD_PATTERNS):
        return False
    wz_keywords = ["cone", "barrier", "sign", "worker", "construction",
                   "work zone", "road work", "lane", "traffic control"]
    return any(k in text.lower() for k in wz_keywords)

infer_fps = fps / FRAME_STRIDE
print(f"Processing {total_fr} frames at {fps:.0f}fps ({total_fr/fps:.0f}s)...")
print(f"Frame stride: {FRAME_STRIDE} → model runs at {infer_fps:.1f}fps ({total_fr//FRAME_STRIDE} inferences)\n")

# ── Main loop ──────────────────────────────────────────────────────────────────
sm = WorkZoneStateMachine(infer_fps=infer_fps)
current_text    = "Analyzing..."
last_valid_text = ""
current_state   = WZState.OUTSIDE
valid           = False
frame_idx       = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sec = frame_idx / fps

    # ── 1. Description inference (every FRAME_STRIDE frames) ──────────────
    if frame_idx % FRAME_STRIDE == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"  t={sec:.1f}s — desc...", end=" ", flush=True)
        raw = run_inference(pil, QUESTION_DESC, max_new_tokens=80)
        print(f"({len(raw)} chars)", end=" ", flush=True)

        valid = is_valid_workzone(raw)
        if valid:
            current_text    = raw
            last_valid_text = raw
        elif last_valid_text:
            current_text = last_valid_text
        else:
            current_text = raw

        # ── 2. State machine update ─────────────────────────────────────
        if valid:
            kw_scores = keyword_score(raw)
        else:
            kw_scores = {WZState.OUTSIDE: 1.0, WZState.APPROACHING: 0.0,
                         WZState.INSIDE: 0.0, WZState.EXITING: 0.0}

        current_state = sm.update(kw_scores)
        print(f"| STATE: {current_state.value.upper()} {'[raw]' if valid else '[HOLD]'}")
        print(f"    >> {raw[:120]}")

    # ── 4. Build panel ─────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, OUT_H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    # Header
    draw.rectangle([0, 0, PANEL_W, 50], fill=(30, 60, 30))
    draw.text((10, 8),  "Alpamayo 1.5 — Work Zone Reasoning", font=font,    fill=(100, 255, 100))
    draw.text((10, 28), f"t = {sec:.1f}s  |  frame {frame_idx}",            font=font_sm, fill=(150, 200, 150))

    # Work zone state badge
    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 100], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"],       font=font_lg, fill=style["fg"])
    # Confidence bar
    conf = sm.confidence
    bar_w = int((PANEL_W - 20) * conf)
    draw.rectangle([10, 96, PANEL_W - 10, 100], fill=(40, 40, 40))
    draw.rectangle([10, 96, 10 + bar_w,   100], fill=style["fg"])

    # EMA score bars (small, for debugging / visual)
    draw.text((10, 106), "Scores (EMA):", font=font_sm, fill=(120, 120, 120))
    score_labels = [
        (WZState.OUTSIDE,     "OUT "),
        (WZState.APPROACHING, "APPR"),
        (WZState.INSIDE,      "IN  "),
        (WZState.EXITING,     "EXIT"),
    ]
    sx = 10
    for wz, lbl in score_labels:
        sc  = sm.ema_scores[wz]
        col = STATE_STYLE[wz]["fg"]
        bw  = int(120 * sc)
        draw.text((sx, 122), lbl,                 font=font_sm, fill=col)
        draw.rectangle([sx, 138, sx + 120, 144],  fill=(40, 40, 40))
        draw.rectangle([sx, 138, sx + bw,  144],  fill=col)
        sx += 160

    # Valid/hold indicator
    hold_color = (100, 200, 100) if valid else (150, 100, 100)
    draw.text((10, 150), "[live]" if valid else "[holding last valid]",
              font=font_sm, fill=hold_color)

    # Description
    y_desc = 170
    draw.text((10, y_desc), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    wrapped = textwrap.wrap(current_text, width=46)
    y = y_desc + 18
    for line in wrapped[:20]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 17
        if y > OUT_H - 30:
            break

    # Progress bar
    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, OUT_H - 20, PANEL_W, OUT_H], fill=(20, 20, 20))
    draw.rectangle([0, OUT_H - 20, int(PANEL_W * progress), OUT_H], fill=(50, 150, 50))
    draw.text((5, OUT_H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    combined  = np.hstack([frame, panel_bgr])
    out_vid.write(combined)
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Video saved to: {VIDEO_OUT}")
print(f"Frames processed: {frame_idx}")
