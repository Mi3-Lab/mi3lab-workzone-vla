"""Verifica se o prefill_sdpa.onnx exportado roda de verdade via ONNX
Runtime e da' a MESMA resposta que o PyTorch original (nao so' "salvou sem
erro" -- precisa RODAR e bater com o resultado esperado).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ONNX_PATH = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-onnx-test/prefill_sdpa.onnx"
tok = AutoTokenizer.from_pretrained(
    "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
)

print("[1] Carregando sessao ONNX Runtime...", flush=True)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(ONNX_PATH, providers=providers)
print(f"    providers ativos: {sess.get_providers()}", flush=True)
print(f"    inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}", flush=True)
print(f"    outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}", flush=True)

inputs = tok("The capital of France is", return_tensors="np")
input_ids = inputs["input_ids"].astype(np.int64)
print(f"\n[2] Rodando inferencia ONNX com input_ids shape {input_ids.shape}...", flush=True)

input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: input_ids})
logits = out[0]
print(f"    logits shape: {logits.shape}", flush=True)

pred_id = int(np.argmax(logits[0]))
pred_token = tok.decode([pred_id])
print(f"    proximo token previsto (ONNX): {pred_token!r} (esperado: ' the')", flush=True)

if pred_token.strip() == "the":
    print("\n[OK] ONNX RUNTIME PRODUZ A MESMA RESPOSTA QUE O PYTORCH ORIGINAL!")
else:
    print(f"\n[DIVERGENCIA] token diferente do esperado")

print("=== FIM ===")
