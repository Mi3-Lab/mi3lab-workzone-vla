"""Bateria de testes completa pra UMA precisao (bf16/int8/int4) do
Cosmos3-Edge: tamanho real, as 5 perguntas de texto (com latencia por
pergunta), e trajetoria (inverse_dynamics em boston.mp4, com latencia).
Salva tudo num JSON estruturado pra comparar depois entre precisoes.

Uso: python benchmark_precision.py <bf16|int8|int4>
"""
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import cosmos_framework.inference.inference as inf_mod

PRECISION = sys.argv[1]
assert PRECISION in ("bf16", "int8", "int4")

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
OUT_JSON = f"{BASE}/outputs/cosmos3edge_trajectory_test/benchmark_{PRECISION}.json"

_orig_create = inf_mod.OmniInference.create.__func__
results = {"precision": PRECISION}


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

    if PRECISION != "bf16":
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

    return pipe


inf_mod.OmniInference.create = classmethod(_patched_create)

# --- 5 perguntas de texto, com timing ---
sys.argv = [
    "inference.py", "--parallelism-preset=latency",
    "-i", "inputs/workzone/gate_test.json", "inputs/workzone/ego_test.json",
    "inputs/workzone/active_test.json", "inputs/workzone/sign_test.json",
    "inputs/workzone/desc_test.json",
    "-o", f"outputs/bench_{PRECISION}_qa",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main

print(f"\n[{PRECISION}] rodando as 5 perguntas...", flush=True)
t0 = time.time()
main()
t_qa_total = time.time() - t0
print(f"[{PRECISION}] 5 perguntas: {t_qa_total:.2f}s total", flush=True)

results["qa"] = {"total_time_s": t_qa_total, "answers": {}}
for name in ["gate_test", "ego_test", "active_test", "sign_test", "desc_test"]:
    d = json.load(open(f"outputs/bench_{PRECISION}_qa/{name}/sample_outputs.json"))
    results["qa"]["answers"][name] = d["outputs"][0]["content"]["reasoner_text"]

# --- trajetoria (inverse_dynamics), com timing ---
sys.argv = [
    "inference.py", "--parallelism-preset=latency",
    "-i", "inputs/workzone/trajectory_test.json",
    "-o", f"outputs/bench_{PRECISION}_traj",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
print(f"\n[{PRECISION}] rodando trajetoria (inverse_dynamics)...", flush=True)
t0 = time.time()
main()
t_traj = time.time() - t0
print(f"[{PRECISION}] trajetoria: {t_traj:.2f}s total", flush=True)

d = json.load(open(f"outputs/bench_{PRECISION}_traj/trajectory_test/sample_outputs.json"))
action = d["outputs"][0]["content"]["action"]
results["trajectory"] = {"total_time_s": t_traj, "action_60x9": action}

import os
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(results, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
print(f"\n[ok] resultados salvos em {OUT_JSON}")
print("=== FIM ===")
