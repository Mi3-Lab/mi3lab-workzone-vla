"""Wrapper de evaluate_hf.py que aplica o patch de vocabulario idempotente
do alpamayo_r1 antes de qualquer instanciacao de modelo (mesmo motivo do
train_action_2b.py -- ver alpamayo_r1_vocab_patch.py).
"""
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
import alpamayo_r1_vocab_patch  # noqa: E402

_RECIPES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "alpamayo-recipes", "recipes")
sys.path.insert(0, os.path.join(_RECIPES, "alpamayo1_5_sft"))
sys.path.insert(0, _RECIPES)
from evaluate_hf import evaluate  # noqa: E402

if __name__ == "__main__":
    evaluate()
