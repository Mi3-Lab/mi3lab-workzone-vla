"""Testa os frames que quebraram no pipeline TensorRT (repeticao/template
generico) diretamente no BF16, pra isolar se e bug do pipeline TRT ou
fragilidade real do modelo/checkpoint em geracoes longas.
"""
import os, sys
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
Q_DESC = "Describe the work zone elements visible in this scene."

BROKEN_FRAMES = [
    ("seattle_unseen", 90),   # ".DATA EAST EAST EAST..."
    ("boston", 36),           # "SUBEXTEND EXTEND EXTEND..."
    ("chicago", 18),          # ".DATA COLLECTION IN CHARACTERISTICS..."
    ("boston", 18),           # ".DATA COLLECTION IN REALTIME USING THE INTERNET..."
]

print("[1] Carregando BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

for city, idx in BROKEN_FRAMES:
    img_path = f"{BASE}/models/workzone-2b-stage6-engines/video_frames/{city}_{idx:05d}.jpg"
    if not os.path.exists(img_path):
        print(f"[{city}#{idx}] frame nao encontrado em {img_path}, pulando")
        continue
    pil = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": Q_DESC},
    ]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out_greedy = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        out_sample = model.generate(**inputs, max_new_tokens=80, do_sample=True,
                                     temperature=0.4, top_p=0.9, top_k=40)
    ans_greedy = tok.decode(out_greedy[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ans_sample = tok.decode(out_sample[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{city}#{idx}]")
    print(f"  BF16 greedy : {ans_greedy!r}")
    print(f"  BF16 sample : {ans_sample!r}")

print("\n=== COMPARACAO CONCLUIDA ===")
