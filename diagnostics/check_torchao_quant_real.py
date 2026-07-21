"""Confirma se o quantize_() do torchao REALMENTE reduziu o armazenamento, ou
se so' rodou sem erro sem mudar nada (minha medicao anterior deu 0.0%, o que
e' suspeito -- precisa verificar de verdade, nao assumir)."""
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

import sys
sys.argv = [
    "inference.py", "--parallelism-preset=latency",
    "-i", "inputs/reasoner/reasoner.json", "-o", "outputs/_check_quant",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main
try:
    main()
except _Stop:
    pass

model = _captured["model"]
lm = model.net.language_model

target = lm.model.layers[0].self_attn.q_proj
print(f"ANTES -- tipo do peso: {type(target.weight)}, dtype={target.weight.dtype}, "
     f"numel={target.weight.numel()}, element_size={target.weight.element_size()}, "
     f"bytes_logicos={target.weight.numel()*target.weight.element_size()}")
try:
    print(f"  .weight.data.__class__ = {target.weight.data.__class__}")
    inner = target.weight
    for attr in ["layout_tensor", "_layout", "tensor_impl", "qdata", "int_data", "_data"]:
        if hasattr(inner, attr):
            v = getattr(inner, attr)
            print(f"  .{attr} presente: type={type(v)}", end="")
            if torch.is_tensor(v):
                print(f" dtype={v.dtype} numel={v.numel()} element_size={v.element_size()} "
                     f"bytes_reais={v.numel()*v.element_size()}")
            else:
                print()
except Exception as e:
    print(f"  erro inspecionando: {e}")

from torchao.quantization import quantize_, Int8WeightOnlyConfig
qconfig = Int8WeightOnlyConfig()


def _filter_fn(module, fqn):
    return isinstance(module, torch.nn.Linear) and "moe_gen" not in fqn


quantize_(lm, qconfig, filter_fn=_filter_fn)

target2 = lm.model.layers[0].self_attn.q_proj
print(f"\nDEPOIS -- tipo do peso: {type(target2.weight)}, dtype={target2.weight.dtype}, "
     f"numel={target2.weight.numel()}, element_size={target2.weight.element_size()}, "
     f"bytes_logicos={target2.weight.numel()*target2.weight.element_size()}")
inner2 = target2.weight
print(f"  .weight.data.__class__ = {inner2.data.__class__ if hasattr(inner2,'data') else 'N/A'}")
for attr in dir(inner2):
    if attr.startswith("_") and not attr.startswith("__"):
        try:
            v = getattr(inner2, attr)
            if torch.is_tensor(v):
                print(f"  .{attr}: tensor dtype={v.dtype} numel={v.numel()} "
                     f"element_size={v.element_size()} bytes_reais={v.numel()*v.element_size()}")
        except Exception:
            pass

# medicao real via torch.save + tamanho de arquivo (a prova definitiva)
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    torch.save(target2.weight, f.name)
    size_quantized = os.path.getsize(f.name)
    os.unlink(f.name)
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    torch.save(target.weight.clone().detach() if not hasattr(target.weight, '_data') else torch.randn(1), f.name)
    os.unlink(f.name)

print(f"\ntamanho serializado do peso quantizado (torch.save real): {size_quantized} bytes")
print(f"tamanho logico bf16 original seria: {target.weight.numel()*2} bytes")
print("=== FIM ===")
