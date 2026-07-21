"""Testa se da' pra exportar a torre de texto do Cosmos3-Edge (arquitetura
NATIVA, Nemotron3DenseVLTextForCausalLM) direto pra ONNX via torch.onnx.export,
sem depender do TensorRT-Edge-LLM (que exige formato Qwen3VL, incompativel).

ONNX Runtime tem execution provider TensorRT oficialmente suportado no
Jetson -- um ONNX generico + onnxruntime-genai/onnxruntime quantization
tooling e' um caminho de deploy edge legitimo, diferente do TensorRT-Edge-LLM
especifico que usamos no 2B.

So' testa export do forward basico (1 passo, sem KV-cache dinamico) primeiro
-- se isso funcionar, da' pra evoluir pra export com cache depois.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import cosmos_framework.inference.inference as inf_mod

_captured = {}
_orig_create = inf_mod.OmniInference.create.__func__


class _Stop(SystemExit):
    pass


def _patched(cls, setup_args, /):
    pipe = _orig_create(cls, setup_args)
    _captured["model"] = pipe.model
    raise _Stop()


inf_mod.OmniInference.create = classmethod(_patched)

sys.argv = [
    "inference.py", "--parallelism-preset=latency",
    "-i", "inputs/reasoner/reasoner.json", "-o", "outputs/_onnx_scratch",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main
print("[1] Carregando modelo (caminho oficial do CLI)...", flush=True)
try:
    main()
except _Stop:
    pass

model = _captured["model"]
lm = model.net.language_model
lm.eval()
print(f"    classe: {type(lm).__name__}", flush=True)

print("\n[2] Preparando entrada de exemplo (input_ids simples)...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
)
inputs = tok("The capital of France is", return_tensors="pt").to(lm.device)
print(f"    input_ids shape: {inputs.input_ids.shape}", flush=True)

print("\n[3] Testando forward() simples (sem gerar, so' 1 passo)...", flush=True)
try:
    with torch.no_grad():
        out = lm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    print(f"    forward OK, logits shape: {out.logits.shape}", flush=True)
except Exception as e:
    print(f"    [ERRO no forward] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n[4] Tentando torch.onnx.export (dynamo=True, export nativo novo)...", flush=True)
OUT_ONNX = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-onnx-test"
import os
os.makedirs(OUT_ONNX, exist_ok=True)

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

wrapped = Wrapper(lm)
try:
    onnx_program = torch.onnx.export(
        wrapped,
        (inputs.input_ids, inputs.attention_mask),
        dynamo=True,
        report=True,
    )
    onnx_program.save(f"{OUT_ONNX}/model.onnx")
    print(f"    [OK] export dynamo=True salvo em {OUT_ONNX}/model.onnx", flush=True)
except Exception as e:
    print(f"    [ERRO export dynamo=True] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()

    print("\n[4b] Tentando torch.onnx.export classico (dynamo=False)...", flush=True)
    try:
        torch.onnx.export(
            wrapped,
            (inputs.input_ids, inputs.attention_mask),
            f"{OUT_ONNX}/model_legacy.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                         "attention_mask": {0: "batch", 1: "seq"},
                         "logits": {0: "batch", 1: "seq"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"    [OK] export classico salvo em {OUT_ONNX}/model_legacy.onnx", flush=True)
    except Exception as e2:
        print(f"    [ERRO export classico tambem] {type(e2).__name__}: {e2}", flush=True)
        import traceback; traceback.print_exc()

print("\n=== FIM ===")
