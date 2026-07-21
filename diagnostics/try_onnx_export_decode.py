"""Export ONNX da etapa de DECODE (token a token com KV-cache) do reasoner
do Cosmos3-Edge -- completa o prefill_sdpa.onnx ja exportado/verificado.

Interface do grafo de decode:
  entradas: input_ids [B,1], position_ids [B,1],
            past_keys [num_layers, B, T_past, H_kv, D],
            past_values [num_layers, B, T_past, H_kv, D]
  saidas:   logits [B, vocab] (fp32),
            new_keys/new_values [num_layers, B, T_past+1, H_kv, D]

Com T=1 fixo, o branch `is_causal = T > 1 and ...` em reasoner_forward vira
estaticamente False -- sem controle de fluxo dependente de dado no grafo.
Atencao patcheada pra SDPA (mesmo patch do export do prefill, ja verificado
nao mudar o resultado).

Verificacao no final: roda 3 passos de decode via ONNX Runtime encadeando o
cache e compara os tokens com o PyTorch original.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

import cosmos_framework.model.attention.frontend as attn_frontend


def _sdpa_replacement(query, key, value, is_causal=False, causal_type=None, scale=None,
                      cumulative_seqlen_Q=None, cumulative_seqlen_KV=None,
                      max_seqlen_Q=None, max_seqlen_KV=None, return_lse=False,
                      backend_kwargs=None, deterministic=False):
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale,
                                         enable_gqa=enable_gqa)
    return out.transpose(1, 2)


for _name in list(attn_frontend.BACKEND_MAP.keys()):
    attn_frontend.BACKEND_MAP[_name] = _sdpa_replacement
print("[patch] atencao -> SDPA nativo", flush=True)

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
    "-i", "inputs/reasoner/reasoner.json", "-o", "outputs/_onnx_scratch_decode",
    "--checkpoint-path", "Cosmos3-Edge", "--seed=0", "--no-guardrails",
]
from cosmos_framework.scripts.inference import main
print("[1] Carregando modelo...", flush=True)
try:
    main()
except _Stop:
    pass

model = _captured["model"]
lm = model.net.language_model
lm.eval()

from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache
from transformers import AutoTokenizer

SNAP = "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
tok = AutoTokenizer.from_pretrained(SNAP)

num_layers = lm.model.config.num_hidden_layers
print(f"    num_layers: {num_layers}", flush=True)

# referencia PyTorch: prefill + 3 decodes gulosos
prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt").to(lm.device)
ref_tokens = []
with torch.no_grad():
    cache = ReasonerKVCache.empty(num_layers)
    hidden = lm.model.reasoner_forward(inputs.input_ids, cache)
    next_id = lm.lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
    ref_tokens.append(int(next_id))
    for _ in range(2):
        past_len = cache.seq_len
        pos = torch.tensor([[past_len]], device=lm.device)
        hidden = lm.model.reasoner_forward(next_id, cache, position_ids=pos)
        next_id = lm.lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
        ref_tokens.append(int(next_id))
    # guarda o estado do cache POS-PREFILL pra alimentar o ONNX depois
    cache_ref = ReasonerKVCache.empty(num_layers)
    _ = lm.model.reasoner_forward(inputs.input_ids, cache_ref)
print(f"[2] referencia PyTorch (3 tokens): {[tok.decode([t]) for t in ref_tokens]}", flush=True)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, text_model, lm_head, num_layers):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_layers = num_layers

    def forward(self, input_ids, position_ids, past_keys, past_values):
        cache = ReasonerKVCache(
            keys=list(past_keys.unbind(0)),
            values=list(past_values.unbind(0)),
        )
        hidden = self.text_model.reasoner_forward(input_ids, cache, position_ids=position_ids)
        logits = self.lm_head(hidden[:, -1, :]).float()
        new_keys = torch.stack(cache.keys, dim=0)
        new_values = torch.stack(cache.values, dim=0)
        return logits, new_keys, new_values


wrapper = DecodeWrapper(lm.model, lm.lm_head, num_layers).eval()

past_keys = torch.stack(cache_ref.keys, dim=0)      # [L,B,T,Hkv,D]
past_values = torch.stack(cache_ref.values, dim=0)
T_past = past_keys.shape[2]
dec_ids = torch.tensor([[ref_tokens[0]]], device=lm.device)
dec_pos = torch.tensor([[T_past]], device=lm.device)
print(f"[3] shapes de exemplo: past_keys {tuple(past_keys.shape)}, input_ids {tuple(dec_ids.shape)}", flush=True)

# sanity: o wrapper reproduz o passo 2 da referencia?
with torch.no_grad():
    lg, nk, nv = wrapper(dec_ids, dec_pos, past_keys, past_values)
print(f"    wrapper sanity: proximo token = {tok.decode([int(lg.argmax())])!r} "
      f"(esperado {tok.decode([ref_tokens[1]])!r})", flush=True)

print("\n[4] Exportando decode step pra ONNX (dynamo, eixo T_past dinamico)...", flush=True)
OUT = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-onnx-test"
import os
os.makedirs(OUT, exist_ok=True)

DYN = torch.export.Dim("t_past", min=1, max=8192)
try:
    onnx_program = torch.onnx.export(
        wrapper,
        (dec_ids, dec_pos, past_keys, past_values),
        dynamo=True,
        dynamic_shapes={
            "input_ids": {},
            "position_ids": {},
            "past_keys": {2: DYN},
            "past_values": {2: DYN},
        },
        report=True,
    )
    onnx_program.save(f"{OUT}/decode_sdpa.onnx")
    print(f"    [OK] salvo em {OUT}/decode_sdpa.onnx", flush=True)
except Exception as e:
    print(f"    [ERRO export] {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n[5] Verificando via ONNX Runtime: 2 passos de decode encadeados...", flush=True)
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(f"{OUT}/decode_sdpa.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
in_names = [i.name for i in sess.get_inputs()]
print(f"    inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}", flush=True)


def to_np_bf16_as_f32(t):
    return t.float().cpu().numpy()


# ONNX pode ter tipado o cache como bf16 -- checa o tipo esperado
key_type = [i.type for i in sess.get_inputs() if "past_keys" in i.name or i.name == in_names[2]][0]
print(f"    tipo do cache no grafo: {key_type}", flush=True)

import ml_dtypes  # numpy bfloat16

def torch_to_np(t, ort_type):
    if "bfloat16" in ort_type:
        return t.cpu().float().numpy().astype(ml_dtypes.bfloat16)
    return t.cpu().numpy()

onnx_tokens = []
cur_ids = dec_ids
cur_pos = dec_pos
cur_k, cur_v = past_keys, past_values
for step in range(2):
    feed = {
        in_names[0]: cur_ids.cpu().numpy(),
        in_names[1]: cur_pos.cpu().numpy(),
        in_names[2]: torch_to_np(cur_k, key_type),
        in_names[3]: torch_to_np(cur_v, key_type),
    }
    logits_np, nk_np, nv_np = sess.run(None, feed)
    tid = int(np.argmax(logits_np[0]))
    onnx_tokens.append(tid)
    cur_ids = torch.tensor([[tid]])
    cur_pos = cur_pos + 1
    cur_k = torch.tensor(nk_np.astype(np.float32) if nk_np.dtype != np.float32 else nk_np).to(torch.bfloat16) if "bfloat16" in key_type else torch.tensor(nk_np)
    cur_v = torch.tensor(nv_np.astype(np.float32) if nv_np.dtype != np.float32 else nv_np).to(torch.bfloat16) if "bfloat16" in key_type else torch.tensor(nv_np)

expected = ref_tokens[1:3]
print(f"    tokens ONNX:     {[tok.decode([t]) for t in onnx_tokens]}", flush=True)
print(f"    tokens PyTorch:  {[tok.decode([t]) for t in expected]}", flush=True)
if onnx_tokens == expected:
    print("\n[OK] DECODE STEP ONNX BATE COM O PYTORCH (2 passos encadeados)!")
else:
    print("\n[DIVERGENCIA] tokens diferentes")
print("=== FIM ===")
