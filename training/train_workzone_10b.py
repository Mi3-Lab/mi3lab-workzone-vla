"""Wrapper de train_hf.py que aplica o patch ZeRO-3-aware de carregamento de
checkpoint ANTES de qualquer instanciacao de modelo -- necessario porque
from_alpamayo_checkpoint faz um load_state_dict cru que quebra sob ZeRO-3
(ver wz10b_zero3_loader.py pro detalhe do bug e do fix).

Mesmo padrao do training/train_action_2b.py: nao modifica train_hf.py
(vendored da NVIDIA), so' reexecuta a mesma train() decorada com
@hydra.main depois do patch aplicado.
"""
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

_RECIPES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "alpamayo-recipes", "recipes")
sys.path.insert(0, os.path.join(_RECIPES, "alpamayo1_5_sft"))
sys.path.insert(0, _RECIPES)

import training.wz10b_zero3_loader  # noqa: E402  (aplica o patch on-import)
from train_hf import train  # noqa: E402

if __name__ == "__main__":
    train()
