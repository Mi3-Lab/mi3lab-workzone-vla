"""v3: monkey-patch o backend de atencao (flash2/flash3/cudnn/natten) pra
usar torch.nn.functional.scaled_dot_product_attention (SDPA nativo do
PyTorch, que TEM suporte de export ONNX), em vez do kernel flash_attn cru
(que o ONNX nao consegue traduzir -- erro do teste anterior).
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

# --- monkey-patch: substitui os backends de atencao por SDPA nativo ---
import cosmos_framework.model.attention.frontend as attn_frontend


def _sdpa_replacement(query, key, value, is_causal=False, causal_type=None, scale=None,
                      cumulative_seqlen_Q=None, cumulative_seqlen_KV=None,
                      max_seqlen_Q=None, max_seqlen_KV=None, return_lse=False,
                      backend_kwargs=None, deterministic=False):
    # entrada: heads-last [B, S, H, D]. SDPA quer [B, H, S, D].
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale,
                                         enable_gqa=enable_gqa)
    return out.transpose(1, 2)  # volta pra [B, S, H, D]


for _name in list(attn_frontend.BACKEND_MAP.keys()):
    attn_frontend.BACKEND_MAP[_name] = _sdpa_replacement
print(f"[patch] BACKEND_MAP substituido por SDPA nativo: {list(attn_frontend.BACKEND_MAP.keys())}", flush=True)

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
    "-i", "inputs/reasoner/reasoner.json", "-o", "outputs/_onnx_scratch_v3",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main
print("[1] Carregando modelo (caminho oficial do CLI, com SDPA patcheado)...", flush=True)
try:
    main()
except _Stop:
    pass

model = _captured["model"]
lm = model.net.language_model
lm.eval()

from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
)
inputs = tok("The capital of France is", return_tensors="pt").to(lm.device)
num_layers = lm.model.config.num_hidden_layers if hasattr(lm.model, "config") else 28

print("\n[2] Testando reasoner_forward() com SDPA patcheado (confirmar que ainda da resposta certa)...", flush=True)
cache = ReasonerKVCache.empty(num_layers)
with torch.no_grad():
    hidden = lm.model.reasoner_forward(inputs.input_ids, cache)
    logits = lm.lm_head(hidden[:, -1, :])
pred_token = logits.argmax(dim=-1)
print(f"    proximo token previsto (com SDPA): {tok.decode(pred_token)!r} (esperado: ' the')", flush=True)

print("\n[3] Tentando torch.onnx.export com SDPA...", flush=True)


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
        return logits.float()  # bf16 -> fp32: numpy/onnxruntime nao tem bfloat16 nativo


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
    onnx_program.save(f"{OUT_ONNX}/prefill_sdpa.onnx")
    print(f"    [OK] export dynamo=True salvo em {OUT_ONNX}/prefill_sdpa.onnx", flush=True)
except Exception as e:
    print(f"    [ERRO export dynamo=True] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()

print("\n=== FIM ===")
