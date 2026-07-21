"""Diagnostico do loss=0.0 constante no fine-tune do 10B em workzone.

Roda 1 forward pass real com o MESMO modelo/config do job de producao, mas
inspeciona diretamente o tensor de `labels` (quantos tokens sao IGNORE_INDEX
vs supervisionados) e a loss crua -- em vez de confiar no numero que o
Trainer loga. Testa DOIS datasets:
  1. lingoqa_combined (o novo, usado no job que deu loss=0.0)
  2. lingoqa_stage7_1 (o mesmo que treinou com sucesso o stage7_1 -- controle)
com o MESMO checkpoint (Alpamayo-1.5-10B-A1-format), pra isolar se o
problema e' o DADO ou o CHECKPOINT/caminho de carregamento.
"""
import sys, os
BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
_RECIPES = f"{BASE}/alpamayo-recipes/recipes"
sys.path.insert(0, f"{_RECIPES}/alpamayo1_5_sft")
sys.path.insert(0, _RECIPES)
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": f"{BASE}/.hf_cache", "WANDB_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
})

import torch
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA, IGNORE_INDEX
from alpamayo.data.lingoqa import LingoQADataset
import functools

print("[1] Carregando modelo (Alpamayo-1.5-10B-A1-format + Cosmos-Reason2-8B)...", flush=True)
model = TrainableReasoningVLA.from_alpamayo_checkpoint(
    checkpoint_path=f"{BASE}/models/Alpamayo-1.5-10B-A1-format",
    vlm_name_or_path=f"{BASE}/models/Cosmos-Reason2-8B",
)
model = model.cuda().eval()
print("    OK\n", flush=True)

VLA_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "question", "answer"],
    "components_prompt": ["answer"],
    "label_components": ["answer"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": False,
}


def probe(name, data_root):
    print(f"=== {name} ({data_root}) ===", flush=True)
    ds = LingoQADataset(local_dir=None, data_root=data_root, parquet_name="train.parquet",
                        model_config=model.config, vla_preprocess_args=VLA_ARGS)
    sample = ds[0]
    td = sample["tokenized_data"]
    print(f"  chaves de tokenized_data: {sorted(td.keys())}")
    print(f"  chaves de sample (nivel superior): {sorted(sample.keys())}")
    input_ids = td.get("input_ids")
    if input_ids is None:
        print("  !! 'input_ids' NAO esta em tokenized_data -- abortando esta probe")
        return
    labels_mask = td.get("labels_mask")
    print(f"  input_ids shape: {tuple(input_ids.shape)}")
    if labels_mask is not None:
        n_total = labels_mask.numel()
        n_true = labels_mask.sum().item()
        print(f"  labels_mask: {n_true}/{n_total} tokens supervisionados "
              f"({'!!! ZERO SUPERVISIONADO !!!' if n_true == 0 else 'ok'})")
    else:
        print("  labels_mask: AUSENTE no sample (chave nao existe)")

    batch = {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in td.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(tokenized_data=batch,
                    ego_history_xyz=None, ego_history_rot=None,
                    ego_future_xyz=None, ego_future_rot=None,
                    labels_mask=batch.get("labels_mask"))
    print(f"  loss real (forward direto): {out.loss.item()}")
    print(f"  loss e' NaN? {torch.isnan(out.loss).item()} | Inf? {torch.isinf(out.loss).item()}")
    print(flush=True)


probe("CONTROLE (dataset que ja funcionou no stage7_1)",
     f"{BASE}/data/roadwork/lingoqa_stage7_1")
probe("NOVO (dataset usado no job que deu loss=0.0)",
     f"{BASE}/data/roadwork/lingoqa_combined")

print("=== FIM DO DIAGNOSTICO ===")
