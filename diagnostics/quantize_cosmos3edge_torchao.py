"""Quantiza o Cosmos3-Edge NA PROPRIA arquitetura dele (sem converter pra
Qwen3VL -- isso quebrou, ver test_cosmos3edge_lm_hf.py), usando torchao
(generico, nao se importa com nome de camada/gate, roda em qualquer GPU CUDA
inclusive L40S/A100/H200, nao so' Blackwell).

Estrategia:
  1. Carrega o modelo pelo MESMO caminho do CLI oficial (OmniInference.create,
     interceptado -- ja' provado dar pesos corretos, ao contrario de
     Cosmos3EdgeForConditionalGeneration direto).
  2. Mede tamanho ANTES (bytes reais dos parametros).
  3. Testa geracao ANTES (uma pergunta real, mesmo estilo dos testes de hoje)
     -- baseline de qualidade.
  4. Aplica torchao quantize_() com Int4WeightOnlyConfig SO' nos nn.Linear da
     torre de ENTENDIMENTO (exclui explicitamente tensores *_moe_gen*, que
     sao' a torre de geracao/difusao/acao -- nao mexer nela por enquanto).
  5. Mede tamanho DEPOIS.
  6. Testa geracao DEPOIS (mesma pergunta) -- compara se ainda faz sentido.

Nao skipa a parte dificil: se a saida pos-quantizacao virar lixo, isso sera'
reportado honestamente, nao escondido.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch

import cosmos_framework.inference.inference as inf_mod

_captured = {}
_orig_create = inf_mod.OmniInference.create.__func__


class _StopAfterLoad(SystemExit):
    pass


def _patched_create(cls, setup_args, /):
    pipe = _orig_create(cls, setup_args)
    _captured["model"] = pipe.model
    _captured["pipe"] = pipe
    raise _StopAfterLoad()


inf_mod.OmniInference.create = classmethod(_patched_create)

sys.argv = [
    "inference.py",
    "--parallelism-preset=latency",
    "-i", "inputs/reasoner/reasoner.json",
    "-o", "outputs/_quantize_scratch",
    "--checkpoint-path", "Cosmos3-Edge",
    "--seed=0",
    "--no-guardrails",
]
from cosmos_framework.scripts.inference import main  # noqa: E402

print("[1] Carregando modelo pelo caminho oficial do CLI...", flush=True)
try:
    main()
except _StopAfterLoad:
    pass
model = _captured["model"]
print(f"    classe: {type(model).__name__}", flush=True)


def model_size_bytes(m):
    total = 0
    for p in m.parameters():
        total += p.numel() * p.element_size()
    for b in m.buffers():
        total += b.numel() * b.element_size()
    return total


size_before = model_size_bytes(model)
print(f"[2] Tamanho ANTES: {size_before / 1e9:.3f} GB", flush=True)

print("\n[3] Testando geracao ANTES da quantizacao (baseline)...", flush=True)
tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
# usa o proprio metodo interno de geracao de texto do reasoner, via a mesma
# rota que test_cosmos3edge_v2/CLI usam -- aqui vamos direto no
# language_model com o tokenizer do pipe pra simplicidade
lm = model.net.language_model if hasattr(model, "net") else None
assert lm is not None, "nao achei model.net.language_model"

pipe = _captured["pipe"]
processor = pipe.processor if hasattr(pipe, "processor") else None


def quick_generate(prompt_text, max_new_tokens=40):
    # tokenizacao minima direto no tokenizer HF do language_model (bypassa
    # chat template multimodal, so' pra comparar antes/depois rapido)
    tok = lm.config._name_or_path if hasattr(lm.config, "_name_or_path") else None
    from transformers import AutoTokenizer
    global _tok_cache
    try:
        _tok_cache
    except NameError:
        _tok_cache = AutoTokenizer.from_pretrained(
            "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub/models--nvidia--Cosmos3-Edge/snapshots/6f58f6b4c91288838e60b6bcb2cc45d997e961de"
        )
    ids = _tok_cache(prompt_text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out = lm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return _tok_cache.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


PROMPT = "The capital of France is"
try:
    before_text = quick_generate(PROMPT)
    print(f"    [ANTES] '{PROMPT}' -> {before_text}", flush=True)
except Exception as e:
    print(f"    [ERRO ao gerar ANTES] {type(e).__name__}: {e}", flush=True)
    before_text = None

print("\n[4] Aplicando torchao Int4WeightOnlyConfig (so' Linear da torre de "
     "ENTENDIMENTO, excluindo *_moe_gen*)...", flush=True)
from torchao.quantization import quantize_, Int4WeightOnlyConfig

qconfig = Int4WeightOnlyConfig(group_size=128)


def _filter_fn(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return False
    if "moe_gen" in fqn:
        return False
    return True


n_quantized = 0
for name, mod in lm.named_modules():
    if _filter_fn(mod, name):
        n_quantized += 1
print(f"    {n_quantized} camadas nn.Linear elegiveis (fora _moe_gen)", flush=True)

quantize_(lm, qconfig, filter_fn=_filter_fn)
print("    quantize_() concluido", flush=True)

size_after = model_size_bytes(model)
print(f"\n[5] Tamanho DEPOIS: {size_after / 1e9:.3f} GB "
     f"(reducao: {100 * (1 - size_after / size_before):.1f}%)", flush=True)

print("\n[6] Testando geracao DEPOIS da quantizacao (mesma pergunta)...", flush=True)
try:
    after_text = quick_generate(PROMPT)
    print(f"    [DEPOIS] '{PROMPT}' -> {after_text}", flush=True)
except Exception as e:
    print(f"    [ERRO ao gerar DEPOIS] {type(e).__name__}: {e}", flush=True)
    after_text = None

print("\n=== RESUMO ===")
print(f"tamanho: {size_before/1e9:.3f} GB -> {size_after/1e9:.3f} GB")
print(f"ANTES:  {before_text}")
print(f"DEPOIS: {after_text}")
print("=== FIM ===")
