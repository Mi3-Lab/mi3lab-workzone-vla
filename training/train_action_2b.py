"""Wrapper de train_hf.py que aplica o patch de vocabulario idempotente do
alpamayo_r1 ANTES de qualquer instanciacao de modelo -- necessario porque
AlpamayoR1Config._build_processor() assume vocabulario virgem, e nosso VLM
2B ja vem com os 4000 tokens <iN> de trajetoria adicionados desde o inicio
do lineage (student_2b_clean_init).

Nao modifica train_hf.py (arquivo vendored da NVIDIA) -- so' reexecuta a
mesma funcao train() decorada com @hydra.main, depois do patch aplicado.
Mantem 100% do parsing de argumentos/config-path/config-name do hydra.
"""
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
import alpamayo_r1_vocab_patch  # noqa: E402  (aplica o patch on-import)

# `alpamayo1_5_sft` (namespace package, PEP 420, sem __init__.py) resolve
# diretamente para recipes/alpamayo1_5_sft/ -- precisa do PAI (`recipes/`)
# no path pra `alpamayo1_5_sft.trainer`/`alpamayo1_5_sft.models.*` resolverem,
# e do proprio diretorio pra achar `train_hf.py` como modulo solto.
_RECIPES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "alpamayo-recipes", "recipes")
sys.path.insert(0, os.path.join(_RECIPES, "alpamayo1_5_sft"))
sys.path.insert(0, _RECIPES)
from train_hf import train  # noqa: E402

if __name__ == "__main__":
    train()
