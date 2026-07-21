"""Testa se o checkpoint HF Qwen3VL (montado a partir do texto real do
Cosmos3-Edge) produz texto coerente. Se o MLP gated estiver realmente
quebrando o calculo (gate_proj estranho do Qwen misturado com up_proj real
do Cosmos3-Edge), a saida deve ser lixo/degenerada -- exatamente o mesmo
sintoma que vimos na tentativa 1 do Cosmos3-Edge (pesos errados).
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

PATH = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-hf"

print("[1] Carregando o modelo completo (Qwen3VLForConditionalGeneration), testando so' texto...")
tok = AutoTokenizer.from_pretrained(PATH)
model = Qwen3VLForConditionalGeneration.from_pretrained(PATH, dtype=torch.bfloat16, device_map="cuda")
model.eval()

PROMPTS = [
    "Are there any road work indicators in this scene? Answer Yes or No.",
    "Describe a modern robotics research laboratory in one sentence.",
    "The capital of France is",
]

for p in PROMPTS:
    inputs = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[PROMPT] {p}")
    print(f"[SAIDA]  {text}")

print("\n=== FIM DO TESTE ===")
