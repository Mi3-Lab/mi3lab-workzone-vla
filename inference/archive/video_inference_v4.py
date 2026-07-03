"""Video inference — Alpamayo 2B v4 (student, Cosmos-Reason2-2B backbone)."""
import os, sys, glob, textwrap
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoProcessor

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
})

VIDEO_IN  = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/boston.mp4"
VIDEO_OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/logs/v4_boston.mp4"
CKPT      = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage4_v4/checkpoint-8622"
STRIDE    = int(sys.argv[4]) if len(sys.argv) > 4 else 3

A1_BASE      = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B    = f"{BASE}/models/Cosmos-Reason2-2B"

QUESTION = "Describe the work zone elements visible in this scene."

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {os.path.basename(CKPT)}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Load student model ────────────────────────────────────────────────────────
from alpamayo1_5_sft.models.kd_model import build_student_model
from workzone_state import WorkZoneStateMachine, WZState, keyword_score

print("\nLoading Alpamayo 2B v4 student model...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)

ckpt_file = os.path.join(CKPT, "model.safetensors")
if os.path.exists(ckpt_file):
    sd = load_file(ckpt_file, device="cpu")
else:
    shards = sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors")))
    sd = {}
    for f in shards:
        sd.update(load_file(f, device="cpu"))

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
model = model.to(torch.bfloat16).cuda().eval()

processor = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
print("Model ready!\n")

# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(pil_image, question, max_new_tokens=100):
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
                   "work zone", "road work", "lane", "traffic control",
                   "drum", "fence", "vehicle"]
    return any(k in text.lower() for k in wz_keywords)

infer_fps = fps / STRIDE
print(f"Processing {total_fr} frames @ {fps:.0f}fps ({total_fr/fps:.0f}s)")
print(f"Stride {STRIDE} → {total_fr//STRIDE} inferences @ {infer_fps:.1f}fps\n")

sm              = WorkZoneStateMachine(infer_fps=infer_fps)
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

    if frame_idx % STRIDE == 0:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"  t={sec:.1f}s", end=" ", flush=True)
        raw = run_inference(pil, QUESTION)
        print(f"({len(raw)}c)", end=" ", flush=True)

        valid = is_valid_workzone(raw)
        if valid:
            current_text    = raw
            last_valid_text = raw
        elif last_valid_text:
            current_text = last_valid_text
        else:
            current_text = raw

        kw_scores = keyword_score(raw) if valid else \
            {WZState.OUTSIDE: 1.0, WZState.APPROACHING: 0.0,
             WZState.INSIDE: 0.0, WZState.EXITING: 0.0}
        current_state = sm.update(kw_scores)
        print(f"| {current_state.value.upper()} {'[live]' if valid else '[hold]'}")
        print(f"    {raw[:120]}")

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v4 — Work Zone (student)", font=font,    fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}",                   font=font_sm, fill=(120, 160, 200))

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 100], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"],       font=font_lg, fill=style["fg"])
    conf  = sm.confidence
    bar_w = int((PANEL_W - 20) * conf)
    draw.rectangle([10, 96, PANEL_W - 10, 100], fill=(40, 40, 40))
    draw.rectangle([10, 96, 10 + bar_w,   100], fill=style["fg"])

    draw.text((10, 106), "Scores (EMA):", font=font_sm, fill=(120, 120, 120))
    score_labels = [(WZState.OUTSIDE,"OUT "),(WZState.APPROACHING,"APPR"),
                    (WZState.INSIDE,"IN  "),(WZState.EXITING,"EXIT")]
    sx = 10
    for wz, lbl in score_labels:
        sc  = sm.ema_scores[wz]
        col = STATE_STYLE[wz]["fg"]
        bw  = int(120 * sc)
        draw.text((sx, 122), lbl, font=font_sm, fill=col)
        draw.rectangle([sx, 138, sx + 120, 144], fill=(40, 40, 40))
        draw.rectangle([sx, 138, sx + bw,  144], fill=col)
        sx += 160

    hold_color = (100, 200, 100) if valid else (150, 100, 100)
    draw.text((10, 150), "[live]" if valid else "[holding last valid]",
              font=font_sm, fill=hold_color)

    y = 170
    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 18
    for line in textwrap.wrap(current_text, width=46)[:20]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 17
        if y > H - 30:
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
