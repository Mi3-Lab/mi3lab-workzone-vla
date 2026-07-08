"""Mede latencia do pipeline BF16 atual (HuggingFace generate()) pra comparar
com os numeros reais medidos no engine TensorRT INT4 (job 175272):
  Vision 27.17ms + Prefill 25.76ms + Generation 2.74ms/tok (384.8 tok/s)
  Total ~108ms pra ~20 tokens gerados.
"""
import os, sys, time
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
IMG = f"{BASE}/models/workzone-2b-stage6-engines/test_frame.jpg"
QUESTION = "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?"

print("[1] Carregando modelo BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

pil = Image.open(IMG).convert("RGB")
msgs = [{"role": "user", "content": [
    {"type": "image", "image": pil}, {"type": "text", "text": QUESTION},
]}]
inputs = proc.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to("cuda")
print(f"    input tokens: {inputs['input_ids'].shape[1]}")

print("[2] Warmup (3x)...")
for _ in range(3):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20, do_sample=False)
torch.cuda.synchronize()

print("[3] Medindo latencia (10 runs, max_new_tokens=20, greedy)...")
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

ans = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"    resposta: {ans!r}")
print(f"\n=== LATENCIA BF16 (HuggingFace generate(), 20 tokens) ===")
print(f"    runs: {times}")
print(f"    media: {sum(times)/len(times):.2f} ms")
print(f"    min: {min(times):.2f} ms  max: {max(times):.2f} ms")
print(f"    tokens/s (media, incluindo prefill+vision): {20*1000/(sum(times)/len(times)):.1f}")

print("\n=== BENCH BF16 CONCLUIDO ===")
