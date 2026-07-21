"""Extrai o state_dict REAL (pesos corretos) da torre de texto do Cosmos3-Edge,
interceptando o pipeline oficial (OmniInference.create) no momento exato em
que ele termina de carregar o modelo -- o MESMO caminho de codigo que ja
provamos funcionar em todos os testes de raciocinio/trajetoria de hoje.

Motivo de nao usar Cosmos3EdgeForConditionalGeneration/AutoModelForImageTextToText
direto: confirmado por inspecao que essa classe tem um mismatch de nomes com
o checkpoint real (espera 'layers.N.mixer.*' em 56 blocos alternados
attn/mlp; o checkpoint tem 'layers.N.self_attn.*'/'layers.N.mlp.*' em 28
blocos normais) -- carrega com pesos ALEATORIOS silenciosamente. So' o
caminho OmniInference (usado pelo CLI oficial) carrega certo.

Salva: {OUT_DIR}/lm_state_dict.pt (so' os tensores da torre de linguagem,
remapeados pra convencao Qwen3VL) + um txt com a lista de chaves pra
conferencia.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import torch

# monkey-patch: intercepta o pipeline logo apos o load do modelo, antes de
# gerar qualquer coisa -- reusa 100% o caminho de carregamento do CLI oficial.
import cosmos_framework.inference.inference as inf_mod

_captured = {}
_orig_create = inf_mod.OmniInference.create.__func__


class _StopAfterLoad(SystemExit):
    pass


def _patched_create(cls, setup_args, /):
    pipe = _orig_create(cls, setup_args)
    print("[patch] pipeline carregado, capturando model.state_dict()...", flush=True)
    _captured["model"] = pipe.model
    raise _StopAfterLoad()


inf_mod.OmniInference.create = classmethod(_patched_create)

# agora chama o entrypoint real do CLI, com os MESMOS argumentos que ja
# provamos funcionar (--checkpoint-path Cosmos3-Edge), apontando pra um
# input minimo qualquer (nao vai gerar nada, para antes disso).
sys.argv = [
    "inference.py",
    "--parallelism-preset=latency",
    "-i", "inputs/reasoner/reasoner.json",
    "-o", "outputs/_extract_lm_scratch",
    "--checkpoint-path", "Cosmos3-Edge",
    "--seed=0",
    "--no-guardrails",
]

from cosmos_framework.scripts.inference import main  # noqa: E402

try:
    main()
except _StopAfterLoad:
    pass

model = _captured["model"]
print(f"[ok] modelo capturado: {type(model).__name__}", flush=True)

sd = model.state_dict()
print(f"[ok] {len(sd)} tensores no state_dict completo", flush=True)

# confirma a convencao de nomes esperada por _remap_to_qwen3vl
# (cosmos_framework/scripts/convert_model_to_vlm_safetensors.py):
# prefixo 'net.language_model.' -> vira 'model.language_model.'/'lm_head.'
lm_keys = [k for k in sd if k.startswith("net.language_model.")]
print(f"[check] {len(lm_keys)} chaves com prefixo 'net.language_model.' (o que _remap_to_qwen3vl espera)")
print("  amostra:")
for k in lm_keys[:10]:
    print(f"    {k}  {tuple(sd[k].shape)}")

if not lm_keys:
    print("\n[AVISO] prefixo 'net.language_model.' NAO encontrado -- listando prefixos reais:")
    prefixes = sorted(set(k.split(".")[0] + "." + k.split(".")[1] if "." in k else k for k in list(sd.keys())[:200]))
    for p in prefixes[:30]:
        print(f"    {p}")

import os
OUT_DIR = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-extracted"
os.makedirs(OUT_DIR, exist_ok=True)
torch.save({k: v.cpu() for k, v in sd.items() if k.startswith("net.language_model.")}, f"{OUT_DIR}/lm_state_dict.pt")
with open(f"{OUT_DIR}/all_keys.txt", "w") as f:
    for k in sd:
        f.write(f"{k}\t{tuple(sd[k].shape)}\n")
print(f"\n[salvo] {OUT_DIR}/lm_state_dict.pt + all_keys.txt")
