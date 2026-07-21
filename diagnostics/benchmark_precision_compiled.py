"""Igual benchmark_precision.py, mas aplica torch.compile() na torre de
texto DEPOIS de quantizar -- torchao e' desenhado pra ser usado JUNTO com
torch.compile pra ganhar os kernels fundidos de verdade (dequant+matmul
fundidos via inductor/triton). Sem isso, o eager mode faz dequant e matmul
como 2 ops separadas, o que explica a lentidao vista no benchmark anterior.

Uso: python benchmark_precision_compiled.py <int8|int4>
"""
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import cosmos_framework.inference.inference as inf_mod

PRECISION = sys.argv[1]
assert PRECISION in ("int8", "int4")

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
OUT_JSON = f"{BASE}/outputs/cosmos3edge_trajectory_test/benchmark_{PRECISION}_compiled.json"

_orig_create = inf_mod.OmniInference.create.__func__
results = {"precision": f"{PRECISION}_compiled"}


def real_serialized_bytes(m):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        size = os.path.getsize(f.name)
        os.unlink(f.name)
    return size


def _patched_create(cls, setup_args, /):
    pipe = _orig_create(cls, setup_args)
    model = pipe.model
    lm = model.net.language_model

    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

    def _filter_fn(module, fqn):
        return isinstance(module, torch.nn.Linear) and "moe_gen" not in fqn

    if PRECISION == "int8":
        quantize_(lm, Int8WeightOnlyConfig(), filter_fn=_filter_fn)
    elif PRECISION == "int4":
        quantize_(lm, Int4WeightOnlyConfig(group_size=128,
                                          int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D),
                  filter_fn=_filter_fn)

    size_bytes = real_serialized_bytes(lm)
    results["size_gb"] = size_bytes / 1e9
    print(f"[{PRECISION}] tamanho serializado real: {size_bytes/1e9:.3f} GB", flush=True)

    print(f"[{PRECISION}] aplicando torch.compile()...", flush=True)
    t0 = time.time()
    lm.forward = torch.compile(lm.forward, mode="max-autotune", fullgraph=False)
    results["compile_setup_s"] = time.time() - t0
    print(f"[{PRECISION}] torch.compile() configurado (compilacao real acontece no 1o forward)", flush=True)

    return pipe


inf_mod.OmniInference.create = classmethod(_patched_create)

sys.argv = [
    "inference.py", "--parallelism-preset=latency",
    "-i", "inputs/workzone/gate_test.json", "inputs/workzone/ego_test.json",
    "inputs/workzone/active_test.json", "inputs/workzone/sign_test.json",
    "inputs/workzone/desc_test.json",
    "-o", f"outputs/bench_{PRECISION}_compiled_qa",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main

print(f"\n[{PRECISION}] rodando as 5 perguntas (1a chamada inclui compilacao)...", flush=True)
t0 = time.time()
try:
    main()
    t_qa_total = time.time() - t0
    print(f"[{PRECISION}] 5 perguntas (com compile): {t_qa_total:.2f}s total", flush=True)
    results["qa"] = {"total_time_s": t_qa_total, "answers": {}}
    for name in ["gate_test", "ego_test", "active_test", "sign_test", "desc_test"]:
        d = json.load(open(f"outputs/bench_{PRECISION}_compiled_qa/{name}/sample_outputs.json"))
        results["qa"]["answers"][name] = d["outputs"][0]["content"]["reasoner_text"]
except Exception as e:
    print(f"[{PRECISION}] [ERRO] torch.compile no forward() customizado falhou: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    results["error"] = f"{type(e).__name__}: {e}"

import os
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
print(f"\n[ok] resultados salvos em {OUT_JSON}")
print("=== FIM ===")
