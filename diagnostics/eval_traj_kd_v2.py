"""Eval v2 do student KD: decodificação RESTRITA ao vocabulário de trajetória.

Diagnóstico do eval v1 (job 171982): em geração livre o student emite ~750/768
tokens FORA da faixa de trajetória (texto comum) → extract_traj_tokens clampa
tudo pra 0 → minADE ~31 m. Sob teacher forcing a loss era ~3.0, ou seja, o
ranking DENTRO do vocab de trajetória pode ser decente — a massa é que vaza
pro vocab de texto.

Aqui: LogitsProcessor mascara tudo exceto os 4000 traj tokens durante os 128
passos de geração (após <traj_future_start>, só traj tokens são válidos por
construção). Também imprime, pro primeiro clipe, o que o modelo gera SEM
restrição, pra diagnóstico.

Comparar com: teacher 10B (sem restrição, job 171982) minADE média 0.557 m,
mediana 0.402 m; student sem restrição: 31.2 m.
"""
import os, sys
import numpy as np
import torch

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from transformers import LogitsProcessor, LogitsProcessorList
from alpamayo.data.pai import PAIDataset
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

VAL_CHUNKS = [2868, 3126]
N_CLIPS    = 24
N_SAMPLES  = 6
TOP_P, TEMP = 0.98, 0.6

PREPROC = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": True,
}


class TrajOnlyLogits(LogitsProcessor):
    """Permite apenas tokens do vocabulário de trajetória."""
    def __init__(self, start_idx: int, traj_vocab: int, full_vocab: int):
        allowed = torch.zeros(full_vocab, dtype=torch.bool)
        allowed[start_idx : start_idx + traj_vocab] = True
        self.block_mask = ~allowed  # True = proibido

    def __call__(self, input_ids, scores):
        scores[:, self.block_mask.to(scores.device)] = float("-inf")
        return scores


def batch_to_cuda(batch):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cuda()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


print("[1] Carregando student (kd_general_driving)...")
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=f"{BASE}/checkpoints/kd_general_driving",
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-2B",
).to(torch.bfloat16).cuda().eval()

start_idx  = model.future_token_start_idx
traj_vocab = model.traj_tokenizer.vocab_size
full_vocab = model.vlm.get_input_embeddings().num_embeddings
print(f"    future_token_start_idx={start_idx}  traj_vocab={traj_vocab}  full_vocab={full_vocab}")

processor = TrajOnlyLogits(start_idx, traj_vocab, full_vocab)

# Injeta o LogitsProcessor em toda chamada generate() do VLM
_orig_generate = model.vlm.generate
def _constrained_generate(*args, **kwargs):
    kwargs["logits_processor"] = LogitsProcessorList([processor])
    return _orig_generate(*args, **kwargs)

ds = PAIDataset(
    local_dir=f"{BASE}/data/PhysicalAI-AV",
    chunk_ids=VAL_CHUNKS,
    use_default_keyframe=True,
    vla_preprocess_args=dict(PREPROC),
    model_config=model.config,
)
idxs = np.linspace(0, len(ds) - 1, N_CLIPS).astype(int).tolist()
print(f"[2] {len(ds)} clipes de val, avaliando {len(idxs)}")

# ── Diagnóstico: o que o student gera SEM restrição (1 clipe) ────────────────
print("\n[3] DIAGNOSTICO — geração livre no primeiro clipe (o que ele emite?):")
sample = ds[idxs[0]]
batch = batch_to_cuda(collate_fn_from_model_config(
    [sample], model_config=model.config, chat_template_version="r1_5"))
td = dict(batch["tokenized_data"])
input_ids = td.pop("input_ids")
traj_data = {"ego_history_xyz": batch["ego_history_xyz"],
             "ego_history_rot": batch["ego_history_rot"]}
fused = model.fuse_traj_tokens(input_ids.clone(), traj_data)
gc = model.vlm.generation_config
gc.do_sample, gc.temperature, gc.top_p = True, TEMP, TOP_P
gc.max_new_tokens, gc.num_return_sequences = 40, 1
gc.pad_token_id = model.tokenizer.pad_token_id
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.vlm.generate(input_ids=fused, **td, generation_config=gc)
gen = out[:, fused.shape[1]:][0]
print(f"    ids gerados : {gen[:25].tolist()}")
print(f"    decodificado: {model.tokenizer.decode(gen)[:300]!r}")
n_traj = ((gen >= start_idx) & (gen < start_idx + traj_vocab)).sum().item()
print(f"    tokens na faixa de trajetória: {n_traj}/{len(gen)}")

# ── Eval com decodificação restrita ──────────────────────────────────────────
print("\n[4] EVAL com decodificação RESTRITA (traj tokens apenas):")
model.vlm.generate = _constrained_generate

min_ades, min_fdes = [], []
for i, idx in enumerate(idxs):
    sample = ds[idx]
    batch = batch_to_cuda(collate_fn_from_model_config(
        [sample], model_config=model.config, chat_template_version="r1_5"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data(
            data=batch,
            top_p=TOP_P, temperature=TEMP,
            num_traj_samples=N_SAMPLES, num_traj_sets=1,
            max_generation_length=model.config.tokens_per_future_traj,
        )
    pred = pred_xyz[0, 0].float().cpu().numpy()
    gt = batch["ego_future_xyz"][0, -1].float().cpu().numpy()
    T = min(pred.shape[1], gt.shape[0])
    err = np.linalg.norm(pred[:, :T, :2] - gt[None, :T, :2], axis=-1)
    min_ade, min_fde = float(err.mean(axis=1).min()), float(err[:, -1].min())
    min_ades.append(min_ade)
    min_fdes.append(min_fde)
    print(f"  [{i+1:2d}/{len(idxs)}] clip {sample['clip_id'][:8]}  "
          f"minADE={min_ade:6.2f} m  minFDE={min_fde:6.2f} m", flush=True)

a, f = np.array(min_ades), np.array(min_fdes)
print(f"\nRESUMO STUDENT RESTRITO ({len(a)} clipes, K={N_SAMPLES}):")
print(f"  minADE  media={a.mean():.3f} m  mediana={np.median(a):.3f} m  "
      f"p90={np.percentile(a,90):.3f} m  max={a.max():.3f} m")
print(f"  minFDE  media={f.mean():.3f} m  mediana={np.median(f):.3f} m")
print(f"\nREFERENCIAS: teacher 0.557/0.402 m (media/mediana) | student livre 31.2/20.4 m")
print("\n=== EVAL V2 CONCLUIDO ===")
