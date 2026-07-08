"""Exporta o VLM do checkpoint stage7 (work zone) para formato HF padrão.

O checkpoint sft_stage6_general é um TrainableReasoningVLA (wrapper Alpamayo)
cujo estado é 100% VLM (sem expert). Para quantizar/deployar no Jetson com
toolchains padrão (llama.cpp GGUF, TensorRT-LLM, modelopt), precisamos de um
diretório Qwen3VLForConditionalGeneration puro + tokenizer/processor com o
vocabulário expandido idêntico ao usado na inferência.

Estratégia: carregar via from_alpamayo_checkpoint (mesmo caminho do v16 —
garante consistência) e salvar model.vlm + model.tokenizer + model.processor.

Saída: models/workzone-2b-stage7-hf/
"""
import os, sys
import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

CKPT = f"{BASE}/checkpoints/sft_stage7_negatives/checkpoint-1269"
OUT  = f"{BASE}/models/workzone-2b-stage7-hf"

print(f"[1] Carregando {CKPT} via from_alpamayo_checkpoint...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=CKPT,
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-2B",
).to(torch.bfloat16).eval()

vlm = model.vlm
print(f"    classe VLM : {vlm.__class__.__name__}")
print(f"    vocab      : {vlm.get_input_embeddings().num_embeddings}")
print(f"    tokenizer  : {len(model.tokenizer)} tokens")

print(f"[2] Salvando VLM em {OUT} ...")
os.makedirs(OUT, exist_ok=True)
vlm.save_pretrained(OUT, safe_serialization=True)

print("[3] Salvando tokenizer + processor...")
model.tokenizer.save_pretrained(OUT)
model.processor.save_pretrained(OUT)

print("[4] Sanity check: recarregando do diretório exportado...")
del model, vlm
torch.cuda.empty_cache()

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
m2 = AutoModelForImageTextToText.from_pretrained(
    OUT, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()
proc2 = AutoProcessor.from_pretrained(OUT)
tok2 = AutoTokenizer.from_pretrained(OUT)
print(f"    reload OK — {m2.__class__.__name__}, vocab {m2.get_input_embeddings().num_embeddings}, tokenizer {len(tok2)}")

# geração de teste com uma imagem real de work zone (frame do vídeo de boston)
import cv2
from PIL import Image
cap = cv2.VideoCapture(f"{BASE}/videos/boston.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, frame = cap.read()
cap.release()
assert ok
pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

msgs = [{"role": "user", "content": [
    {"type": "image", "image": pil},
    {"type": "text", "text": "Describe the work zone elements visible in this scene."},
]}]
inputs = proc2.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to("cuda")
with torch.no_grad():
    out = m2.generate(**inputs, max_new_tokens=80, do_sample=False)
ans = tok2.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[5] Resposta do modelo exportado (frame boston):\n    {ans!r}")

import subprocess
size = subprocess.run(["du", "-sh", OUT], capture_output=True, text=True).stdout.split()[0]
print(f"\n[6] Tamanho do export: {size}")
print("\n=== EXPORT CONCLUIDO ===")
