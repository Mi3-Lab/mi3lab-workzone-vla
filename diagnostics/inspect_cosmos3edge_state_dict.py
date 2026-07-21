"""Inspeciona a estrutura real de pesos (state_dict) do
Cosmos3EdgeForConditionalGeneration -- a classe que REALMENTE carrega o
Cosmos3-Edge (confirmada em uso em todos os testes de raciocinio/trajetoria
de hoje), diferente da Cosmos3OmniModel que o script oficial de export
(convert_model_to_vlm_safetensors.py) assume e que NAO funciona pro Edge.

Objetivo: descobrir a convencao de nomes real, pra escrever um mapeador
customizado pro formato Qwen3VLForConditionalGeneration (que o
TensorRT-Edge-LLM ja suporta oficialmente), sem depender do script quebrado.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from huggingface_hub import snapshot_download

import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401 -- registra cosmos3_edge
from transformers import AutoModelForImageTextToText, Qwen3VLForConditionalGeneration

print("[1] Baixando/localizando snapshot...", flush=True)
snapshot_dir = snapshot_download("nvidia/Cosmos3-Edge")
print(f"    {snapshot_dir}", flush=True)

print("[2] Carregando Cosmos3-Edge (classe nativa correta)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    snapshot_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa",
)
print(f"    classe: {type(model).__name__}", flush=True)

sd = model.state_dict()
print(f"\n[3] {len(sd)} tensores no state_dict. Primeiras 60 chaves (ordem original):", flush=True)
for k in list(sd.keys())[:60]:
    print(f"  {k}  {tuple(sd[k].shape)}")

print(f"\n[4] Chaves unicas por PREFIXO de 1o nivel:", flush=True)
prefixes = sorted(set(k.split(".")[0] for k in sd.keys()))
for p in prefixes:
    n = sum(1 for k in sd if k.startswith(p + "."))
    print(f"  {p}: {n} tensores")

print(f"\n[5] Buscando chaves candidatas a 'torre de texto' (contendo 'language' ou 'text' ou 'layers'):", flush=True)
text_like = [k for k in sd if "language" in k.lower() or "text" in k.lower()]
print(f"  {len(text_like)} chaves com 'language'/'text' no nome, amostra:")
for k in text_like[:20]:
    print(f"    {k}  {tuple(sd[k].shape)}")

print(f"\n[6] Buscando chaves candidatas a 'torre de geracao/difusao' (contendo 'moe_gen', 'diffusion', 'gen', 'expert'):", flush=True)
gen_like = [k for k in sd if any(s in k.lower() for s in ["moe_gen", "diffusion", "_gen", "expert"])]
print(f"  {len(gen_like)} chaves, amostra:")
for k in gen_like[:20]:
    print(f"    {k}  {tuple(sd[k].shape)}")

print(f"\n[7] lm_head / embed_tokens (pra confirmar tie_word_embeddings e vocab_size):", flush=True)
for k in sd:
    if "lm_head" in k or "embed_tokens" in k:
        print(f"    {k}  {tuple(sd[k].shape)}")

print(f"\n[8] Para comparacao -- chaves esperadas por Qwen3VLForConditionalGeneration 'vazio' (so' a estrutura, nao os pesos reais baixados):", flush=True)
try:
    from transformers import Qwen3VLConfig
    cfg = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    empty = Qwen3VLForConditionalGeneration(cfg)
    qwen_keys = list(empty.state_dict().keys())
    print(f"  {len(qwen_keys)} chaves no Qwen3VL vazio. Primeiras 30:")
    for k in qwen_keys[:30]:
        print(f"    {k}")
    del empty
except Exception as e:
    print(f"  falhou ao construir Qwen3VL vazio localmente (esperado se precisar baixar config): {e}")

print("\n=== FIM DA INSPECAO ===")
