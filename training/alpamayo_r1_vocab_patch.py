"""Patch reaproveitavel pro pacote `alpamayo_r1` da NVIDIA.

`ReasoningVLAConfig._build_processor()` (em alpamayo_r1/models/base_model.py)
assume que o VLM de origem ainda nao tem os 4000 tokens discretos de
trajetoria (<i0>..<i3999>) e usa `assert len(discrete_tokens) == num_new_tokens`
apos `tokenizer.add_tokens(...)`.

Nosso checkpoint (lineage student_2b_clean_init -> ... -> stage7_1) ja foi
preparado desde o inicio do projeto pra ser VLA-ready: os 4000 tokens <iN>
ja existem no tokenizer, com <i0>=151669 == traj_token_start_idx original.
`add_tokens()` corretamente retorna 0 novos tokens (eles ja existem) e o
assert (que so cobre o caso "sempre sao novos") falha.

Import este modulo ANTES de qualquer uso de AlpamayoR1Config/AlpamayoR1
pra tornar add_tokens idempotente: tokens ja presentes contam como
"adicionados com sucesso" em vez de gerar 0.
"""
from transformers import PreTrainedTokenizerFast

_orig_add_tokens = PreTrainedTokenizerFast.add_tokens


def _idempotent_add_tokens(self, new_tokens, special_tokens=False):
    if isinstance(new_tokens, str):
        return _orig_add_tokens(self, new_tokens, special_tokens)
    vocab = self.get_vocab()
    already = [t for t in new_tokens if t in vocab]
    genuinely_new = [t for t in new_tokens if t not in vocab]
    n_added = _orig_add_tokens(self, genuinely_new, special_tokens) if genuinely_new else 0
    return n_added + len(already)


def apply():
    PreTrainedTokenizerFast.add_tokens = _idempotent_add_tokens


apply()
