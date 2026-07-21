"""Monkey-patch de load_alpamayo1_vlm pra funcionar sob DeepSpeed ZeRO-3.

O problema: em sft_base_model.py:119, load_alpamayo1_vlm faz
    model.load_state_dict(vlm_state_dict, strict=False, assign=True)
direto, sem saber que o modelo pode estar rodando sob ZeRO-3. Sob ZeRO-3 o
DeepSpeed particiona os parametros na hora da CONSTRUCAO do modelo
(deepspeed.zero.Init(), ativado automaticamente pelo HF Trainer quando
zero_stage==3): cada parametro vira um placeholder com .shape=[0] na
maioria dos ranks (so' o dono real da particao tem dado). load_state_dict
faz uma checagem de shape ANTES do assign e explode com
    "size mismatch ... shape in current model is torch.Size([0])"
mesmo o checkpoint tendo o shape certo -- porque ele compara contra o
shape LOCAL/particionado, nao o shape LOGICO completo.

A correcao padrao (como o proprio from_pretrained do HF Transformers lida
com isso) e' "regatherizar" os parametros temporariamente com
deepspeed.zero.GatheredParameters antes do load_state_dict, deixando cada
GPU ver o tensor completo por um instante; ao sair do `with`, o DeepSpeed
reparticiona automaticamente. So' fazemos isso quando ZeRO-3 esta de fato
ativo (is_deepspeed_zero3_enabled) -- sob ZeRO-2 (ou sem deepspeed) o
load_state_dict original ja funciona sem modificacao, entao nao mexemos
nesse caminho.

Import este modulo ANTES de instanciar o modelo (mesmo padrao do
alpamayo_r1_vocab_patch.py e do cotrain_model.py neste projeto): o patch
precisa estar de pe antes do Hydra chamar from_alpamayo_checkpoint.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file

from alpamayo1_5_sft.models import sft_base_model
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")

_original_load_alpamayo1_vlm = sft_base_model.load_alpamayo1_vlm


def _load_alpamayo1_vlm_zero3_aware(checkpoint_path: str, model):
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        # ZeRO-2 (ou nenhum deepspeed): o load_state_dict direto ja funciona.
        return _original_load_alpamayo1_vlm(checkpoint_path, model)

    import deepspeed

    # --- mesmo parsing de vlm_state_dict do original (sft_base_model.py) ---
    checkpoint_dir = Path(checkpoint_path)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    vlm_state_dict: dict[str, torch.Tensor] = {}

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f).get("weight_map", {})
        shard_to_keys: defaultdict[str, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            if key.startswith("vlm."):
                shard_to_keys[shard_name].append(key)
        for shard_name, keys in shard_to_keys.items():
            shard_sd = load_safetensors_file(str(checkpoint_dir / shard_name), device="cpu")
            for key in keys:
                if key in shard_sd:
                    vlm_state_dict[key] = shard_sd[key]

    if not vlm_state_dict:
        single_path = checkpoint_dir / "model.safetensors"
        if single_path.exists():
            from safetensors import safe_open
            alias_map: dict[str, str] = {}
            with safe_open(str(single_path), framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                alias_map = {k: v for k, v in meta.items() if k.startswith("vlm.")}
                for key in f.keys():
                    if key.startswith("vlm."):
                        vlm_state_dict[key] = f.get_tensor(key)
            for alias_key, canonical_key in alias_map.items():
                if alias_key not in vlm_state_dict and canonical_key in vlm_state_dict:
                    vlm_state_dict[alias_key] = vlm_state_dict[canonical_key].clone()

    if not vlm_state_dict:
        raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")

    # --- a parte nova: regatheriza SO' os parametros do submodulo vlm antes
    # de carregar, pra load_state_dict ver o shape logico completo. ---
    vlm_params = list(model.vlm.parameters())
    logger.info(
        f"[ZeRO-3] Regatherizando {len(vlm_params)} parametros de vlm.* pra "
        f"carregar {len(vlm_state_dict)} tensores do checkpoint..."
    )
    with deepspeed.zero.GatheredParameters(vlm_params, modifier_rank=0):
        if torch.distributed.get_rank() == 0:
            load_result = model.load_state_dict(vlm_state_dict, strict=False, assign=True)
            logger.info(
                f"Loaded {len(vlm_state_dict)} VLM tensors from {checkpoint_dir} "
                f"(missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)})",
            )
    # GatheredParameters com modifier_rank=0 reparticiona automaticamente ao
    # sair do `with`, propagando pra todos os ranks o que o rank 0 escreveu.

    return model


sft_base_model.load_alpamayo1_vlm = _load_alpamayo1_vlm_zero3_aware
logger.info("[wz10b_zero3_loader] patch aplicado: load_alpamayo1_vlm agora e' ZeRO-3-aware")
