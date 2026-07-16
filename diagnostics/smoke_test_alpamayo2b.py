"""Teste de fumaca: instanciar AlpamayoR1 (arquitetura REAL da NVIDIA, pacote
`alpamayo_r1`) com nosso VLM 2B (workzone-2b-stage7-1-hf) como backbone, em
vez do Cosmos-Reason2-8B/10B original.

Objetivo: confirmar que a classe e' genuinamente agnostica a tamanho de
backbone (como o codigo sugere: `Qwen3VLConfig.from_pretrained(vlm_name_or_path)`
sem nenhum hardcode), e que a interface action_in_proj (VLM hidden_size ->
expert hidden_size) fecha sem erro de shape. Nao treina nada -- so
instancia + 1 forward pass dummy.

Sub-configs (action_space_cfg, action_in_proj_cfg, expert_cfg, diffusion_cfg,
traj_tokenizer_cfg) copiados do config.json real do Alpamayo-1.5-10B-A1-format
-- sao independentes do tamanho do VLM.
"""
import os, sys, json

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/alpamayo-recipes/recipes/alpamayo1_5_sft")
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})

import torch
from transformers import PreTrainedTokenizerFast

# Nosso checkpoint 2B (lineage student_2b_clean_init) ja foi preparado desde
# o inicio pra ser VLA-ready: os 4000 tokens <i0>..<i3999> ja existem no
# tokenizer (confirmado: <i0>=151669, bate exato com traj_token_start_idx do
# config original), junto com <|traj_history|>/<|traj_future|>/<|route_*|>.
# O ReasoningVLAConfig._build_processor() do pacote alpamayo_r1 assume
# vocabulario "virgem" e usa `assert len(discrete_tokens) == num_new_tokens`
# -- falha quando os tokens ja foram adicionados de proposito. Patch pontual:
# add_tokens() vira idempotente (retorna len(tokens) mesmo se ja existirem).
_orig_add_tokens = PreTrainedTokenizerFast.add_tokens
def _idempotent_add_tokens(self, new_tokens, special_tokens=False):
    if isinstance(new_tokens, str):
        return _orig_add_tokens(self, new_tokens, special_tokens)
    vocab = self.get_vocab()
    already = [t for t in new_tokens if t in vocab]
    genuinely_new = [t for t in new_tokens if t not in vocab]
    n_added = _orig_add_tokens(self, genuinely_new, special_tokens) if genuinely_new else 0
    return n_added + len(already)
PreTrainedTokenizerFast.add_tokens = _idempotent_add_tokens

from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

VLM_2B = f"{BASE}/models/workzone-2b-stage7-1-hf"
REF_10B = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"

ref = json.load(open(f"{REF_10B}/config.json"))

print("[1] Construindo AlpamayoR1Config com nosso VLM 2B como backbone...")
config = AlpamayoR1Config(
    vlm_name_or_path=VLM_2B,
    vlm_backend="qwenvl3",
    traj_tokenizer_cfg=ref["traj_tokenizer_cfg"],
    hist_traj_tokenizer_cfg=ref["hist_traj_tokenizer_cfg"],
    traj_vocab_size=ref["traj_vocab_size"],
    tokens_per_history_traj=ref["tokens_per_history_traj"],
    tokens_per_future_traj=ref["tokens_per_future_traj"],
    action_space_cfg=ref["action_space_cfg"],
    action_in_proj_cfg=ref["action_in_proj_cfg"],
    action_out_proj_cfg=ref["action_out_proj_cfg"],
    expert_cfg=ref["expert_cfg"],
    diffusion_cfg=ref["diffusion_cfg"],
    model_dtype="bfloat16",
    attn_implementation="sdpa",  # evita depender de flash-attn instalado
)
print("    OK -- config construido sem erro")

print("\n[2] Instanciando AlpamayoR1 (VLM 2B do zero + expert de acao)...")
model = AlpamayoR1(config)
print("    OK -- modelo instanciado sem erro")

vlm_hidden = model.vlm.config.text_config.hidden_size
expert_hidden = config.expert_cfg["hidden_size"]
print(f"\n[3] Dimensoes: VLM hidden_size={vlm_hidden} | expert hidden_size={expert_hidden}")
print(f"    action_in_proj hidden_size={config.action_in_proj_cfg['hidden_size']}")

total = sum(p.numel() for p in model.parameters())
vlm_params = sum(p.numel() for p in model.vlm.parameters())
expert_params = total - vlm_params
print(f"\n[4] Parametros: total={total:,} | vlm={vlm_params:,} | expert(+resto)={expert_params:,}")

print("\n[5] Movendo pro CUDA + forward pass dummy...")
model = model.to(torch.bfloat16).cuda().eval()

# Monta um batch minimo: 1 imagem dummy + historico de trajetoria dummy.
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

processor = model.processor
messages = [{"role": "user", "content": [
    {"type": "image", "image": dummy_img},
    {"type": "text", "text": "Predict the future trajectory."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[dummy_img], return_tensors="pt", padding=True)
inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

ego_history_xyz = torch.zeros(1, 1, 16, 3, device="cuda")
ego_history_rot = torch.eye(3, device="cuda").reshape(1, 1, 1, 3, 3).repeat(1, 1, 16, 1, 1)

try:
    with torch.no_grad():
        out = model.vlm(**{k: v for k, v in inputs.items()}, output_hidden_states=True)
    print(f"    OK -- forward pass do VLM sozinho funcionou, "
          f"hidden_states[-1].shape={out.hidden_states[-1].shape}")
except Exception as e:
    print(f"    forward pass do VLM: {type(e).__name__}: {e}")

print("\n=== TESTE DE FUMACA CONCLUIDO ===")
