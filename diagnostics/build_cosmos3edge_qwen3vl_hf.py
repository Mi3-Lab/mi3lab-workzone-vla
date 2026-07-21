"""Monta um diretorio HF Qwen3VLForConditionalGeneration a partir dos pesos
REAIS extraidos do Cosmos3-Edge (lm_state_dict.pt, gerado por
extract_cosmos3edge_lm_via_cli.py -- carregados pelo MESMO caminho do CLI
oficial, nao pela classe Cosmos3EdgeForConditionalGeneration que carrega
errado). Adapta a logica de remapeamento de
cosmos_framework/scripts/convert_model_to_vlm_safetensors.py (_remap_to_qwen3vl),
com o ajuste de remover o segmento '._orig_mod.' (artefato do torch.compile).

DECISAO (2026-07-21, discutida com o usuario): a torre visual do Cosmos3-Edge
(SigLIP2, posicao absoluta aprendida) e' estruturalmente incompativel com a
visao nativa do Qwen3VL (RoPE 2D + modulos deepstack/merger que o Cosmos3-Edge
nao tem -- nao e' so' nomeacao, sao subsistemas que exigiriam re-treino).
Por isso este script NAO tenta transplantar a visao: mantem a visao
PRE-TREINADA do Qwen3-VL-2B-Instruct como esta, intocada. So' a torre de TEXTO
usa os pesos reais do Cosmos3-Edge.

LIMITACAO CONHECIDA: a visao resultante NAO foi treinada junto com este texto
-- o grounding de imagem (VQA) deste export e' esperado ser pior do que o
Cosmos3-Edge original ate' que a visao seja re-treinada/adaptada. So' a
capacidade de raciocinio em TEXTO puro do Cosmos3-Edge e' fielmente preservada.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLConfig

from pathlib import Path

LM_STATE_PATH = "/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-extracted/lm_state_dict.pt"
OUT_PATH = Path("/data/wesleyferreiramaia/wokzone-alpamayo/models/cosmos3edge-lm-hf")
VLM_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"  # hidden_size=2048, bate com o Cosmos3-Edge (8B tinha 4096)

_OMNIMOT_LM_PREFIX = "net.language_model."


def _strip_orig_mod(key: str) -> str:
    return key.replace("._orig_mod.", ".")


def _remap_to_qwen3vl(key: str) -> str | None:
    key = _strip_orig_mod(key)
    if not key.startswith(_OMNIMOT_LM_PREFIX):
        return None
    inner = key[len(_OMNIMOT_LM_PREFIX):]
    if "_moe_gen" in inner or "_for_gen" in inner:
        return None
    if inner.startswith("lm_head."):
        return inner
    if inner.startswith("model."):
        return "model.language_model." + inner[len("model."):]
    return None


print("[1] Carregando lm_state_dict.pt (pesos reais, ja extraidos)...")
raw_lm_state = torch.load(LM_STATE_PATH, map_location="cpu")
print(f"    {len(raw_lm_state)} tensores brutos")

lm_state: dict[str, torch.Tensor] = {}
dropped_moe_gen = 0
for k, v in raw_lm_state.items():
    new_k = _remap_to_qwen3vl(k)
    if new_k is not None:
        lm_state[new_k] = v
    elif "_moe_gen" in k:
        dropped_moe_gen += 1
print(f"[2] Remapeados: {len(lm_state)} tensores pro formato Qwen3VL "
     f"({dropped_moe_gen} tensores _moe_gen descartados, torre de geracao/difusao)")

print(f"\n[3] Visao: NAO transplantada (incompatibilidade estrutural real -- ver docstring). "
     f"Mantendo a visao PRE-TREINADA do {VLM_MODEL_NAME} intocada.")

print(f"\n[4] Construindo config a partir de {VLM_MODEL_NAME}, com dims text_config "
     f"ajustadas pro Cosmos3-Edge real (intermediate_size=9216, vocab_size=131072)...")
config = Qwen3VLConfig.from_pretrained(VLM_MODEL_NAME)
config.text_config.intermediate_size = 9216
config.text_config.vocab_size = 131072
config.text_config.tie_word_embeddings = False
print(f"    text_config: hidden_size={config.text_config.hidden_size} "
     f"intermediate_size={config.text_config.intermediate_size} "
     f"num_hidden_layers={config.text_config.num_hidden_layers} "
     f"vocab_size={config.text_config.vocab_size}")

print(f"\n[4a] Carregando {VLM_MODEL_NAME} PRE-TREINADO completo (fonte da visao real)...")
pretrained_full = Qwen3VLForConditionalGeneration.from_pretrained(VLM_MODEL_NAME, dtype=torch.bfloat16)
vision_state = {k: v for k, v in pretrained_full.state_dict().items() if k.startswith("model.visual.")}
print(f"    {len(vision_state)} tensores de visao pre-treinados capturados")
del pretrained_full

print(f"\n[4b] Construindo modelo com a config ajustada (texto no shape certo, "
     f"visao no shape do {VLM_MODEL_NAME})...")
model = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
print(f"    modelo construido: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

lm_state.update(vision_state)
incompatible = model.load_state_dict(lm_state, strict=False)
n_overlaid = len(lm_state) - len(incompatible.unexpected_keys)
print(f"\n[5] Overlay: {n_overlaid}/{len(lm_state)} tensores do Cosmos3-Edge aplicados")
print(f"    unexpected (rejeitados, nao existem no Qwen3VL): {len(incompatible.unexpected_keys)}")
print(f"    missing (ficaram com o default do {VLM_MODEL_NAME}): {len(incompatible.missing_keys)}")
if incompatible.missing_keys:
    print(f"    amostra de missing_keys: {incompatible.missing_keys[:10]}")
if incompatible.unexpected_keys:
    print(f"\n[ERRO] tensores do Cosmos3-Edge nao reconhecidos pelo Qwen3VL:")
    print(f"    {incompatible.unexpected_keys[:10]}")
    raise RuntimeError(f"{len(incompatible.unexpected_keys)} unexpected_keys -- ver acima")

print(f"\n[6] Salvando em {OUT_PATH}...")
OUT_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_PATH, safe_serialization=True)
AutoTokenizer.from_pretrained(VLM_MODEL_NAME).save_pretrained(OUT_PATH)
AutoProcessor.from_pretrained(VLM_MODEL_NAME).save_pretrained(OUT_PATH)
print("Done.")
