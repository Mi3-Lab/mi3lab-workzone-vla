"""v2 do teste de export ONNX: usa o caminho REAL e simples
(Nemotron3DenseVLTextModel.reasoner_forward -> _impl_reasoner_forward).
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
    "-i", "inputs/reasoner/reasoner.json", "-o", "outputs/_onnx_scratch_v2",
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
print(f"    classe do LM: {type(lm).__name__}, classe interna: {type(lm.model).__name__}", flush=True)

from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache

print("\n[2] Preparando entrada (input_ids simples)...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
)
inputs = tok("The capital of France is", return_tensors="pt").to(lm.device)
print(f"    input_ids shape: {inputs.input_ids.shape}", flush=True)

num_layers = lm.model.config.num_hidden_layers if hasattr(lm.model, "config") else 28
print(f"    num_layers inferido: {num_layers}", flush=True)

print("\n[3] Testando reasoner_forward() direto (prefill, cache vazio)...", flush=True)
try:
    cache = ReasonerKVCache.empty(num_layers)
    with torch.no_grad():
        hidden = lm.model.reasoner_forward(inputs.input_ids, cache)
        logits = lm.lm_head(hidden[:, -1, :])
    print(f"    OK -- hidden shape: {tuple(hidden.shape)}, logits shape: {tuple(logits.shape)}", flush=True)
    pred_token = logits.argmax(dim=-1)
    print(f"    proximo token previsto: {tok.decode(pred_token)!r}", flush=True)
except Exception as e:
    print(f"    [ERRO] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n[4] Tentando torch.onnx.export do prefill (wrapper simples, cache vazio)...", flush=True)


class PrefillWrapper(torch.nn.Module):
    def __init__(self, text_model, lm_head, num_layers):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_layers = num_layers

    def forward(self, input_ids):
        cache = ReasonerKVCache.empty(self.num_layers)
        hidden = self.text_model.reasoner_forward(input_ids, cache)
        logits = self.lm_head(hidden[:, -1, :])
        return logits


wrapper = PrefillWrapper(lm.model, lm.lm_head, num_layers).eval()

OUT_ONNX = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-onnx-test"
import os
os.makedirs(OUT_ONNX, exist_ok=True)

try:
    onnx_program = torch.onnx.export(
        wrapper,
        (inputs.input_ids,),
        dynamo=True,
        report=True,
    )
    onnx_program.save(f"{OUT_ONNX}/prefill.onnx")
    print(f"    [OK] export dynamo=True salvo em {OUT_ONNX}/prefill.onnx", flush=True)
except Exception as e:
    print(f"    [ERRO export dynamo=True] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()

    print("\n[4b] Tentando export classico (dynamo=False, opset 17)...", flush=True)
    try:
        torch.onnx.export(
            wrapper,
            (inputs.input_ids,),
            f"{OUT_ONNX}/prefill_legacy.onnx",
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"    [OK] export classico salvo em {OUT_ONNX}/prefill_legacy.onnx", flush=True)
    except Exception as e2:
        print(f"    [ERRO export classico tambem] {type(e2).__name__}: {e2}", flush=True)
        import traceback; traceback.print_exc()

print("\n=== FIM ===")
