"""Avaliacao real de qualidade da cabeca de acao treinada: ADE/minADE da
trajetoria prevista vs GT (egomotion real), no chunk de validacao (3126,
nunca visto em treino).

Reusa sample_trajectories_from_data (TrainableAlpamayoR1, ja treinado) +
distance_metrics.compute_ade/compute_minade (funcoes reais da NVIDIA).
"""
import os, sys, json

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
_RECIPES = f"{BASE}/alpamayo-recipes/recipes"
sys.path.insert(0, f"{_RECIPES}/alpamayo1_5_sft")
sys.path.insert(0, _RECIPES)
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": f"{BASE}/.hf_cache",
})

sys.path.insert(0, f"{BASE}/mi3lab-workzone-vla")
import training.alpamayo_r1_vocab_patch  # noqa: E402  (idempotencia de vocab)

import torch  # noqa: E402
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1  # noqa: E402
from alpamayo.data.pai import PAIDataset  # noqa: E402
from alpamayo.metrics import distance_metrics  # noqa: E402

CKPT = f"{BASE}/checkpoints/sft_action_2b_egomotion/checkpoint-684"
PAI_DIR = f"{BASE}/data/PhysicalAI-AV"
VAL_CHUNKS = [3126]  # nunca usado em treino
N_SAMPLES = 6         # amostras de trajetoria por clip (flow matching e' estocastico)

print("[1] Carregando checkpoint treinado...")
model = TrainableAlpamayoR1.from_pretrained(CKPT, dtype=torch.bfloat16)
model = model.cuda().eval()
print("    OK")

print("\n[2] Carregando dataset de validacao (chunk 3126, nunca visto em treino)...")
vla_preprocess_args = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": False,
    "include_frame_nums": False,
    "generation_mode": True,
}
ds = PAIDataset(
    local_dir=PAI_DIR,
    chunk_ids=VAL_CHUNKS,
    num_history_steps=16,
    num_future_steps=64,
    time_step=0.1,
    use_default_keyframe=True,
    model_config=model.config,
    vla_preprocess_args=vla_preprocess_args,
)
print(f"    OK -- {len(ds)} clipes")

all_min_ade = []
all_ade = []

MAX_CLIPS = int(os.environ.get("EVAL_MAX_CLIPS", len(ds)))
n_eval = min(MAX_CLIPS, len(ds))
print(f"\n[3] Rodando inferencia + ADE em {n_eval}/{len(ds)} clipes...")
for i in range(n_eval):
    sample = ds[i]
    # monta um "batch" de tamanho 1
    data_batch = {}
    for k, v in sample.items():
        if hasattr(v, "unsqueeze"):
            data_batch[k] = v.unsqueeze(0).cuda()
        else:
            data_batch[k] = v

    with torch.no_grad():
        pred_xyz, pred_rot, logprob = model.sample_trajectories_from_data(
            data=data_batch,
            num_traj_samples=N_SAMPLES,
            num_traj_sets=1,
            top_p=0.98,
            temperature=0.6,
        )

    gt_xyz = data_batch["ego_future_xyz"][:, -1] if data_batch["ego_future_xyz"].dim() == 4 \
        else data_batch["ego_future_xyz"]

    min_ade_out = distance_metrics.compute_minade(
        pred_xyz, gt_xyz, disable_summary=True,
        timestep_horizons=[5, 10, 30, 50], time_step=0.1,
    )
    ade_out = distance_metrics.compute_ade(pred_xyz, gt_xyz)

    min_ade_val = min_ade_out["min_ade"].mean().item()
    ade_val = ade_out.mean().item()
    all_min_ade.append(min_ade_val)
    all_ade.append(ade_val)
    print(f"  clip {i+1}/{len(ds)} (id={sample['clip_id']}): "
          f"min_ade={min_ade_val:.3f}m  ade={ade_val:.3f}m")

import statistics
print("\n=== RESUMO ===")
print(f"n clipes avaliados: {len(all_ade)}")
print(f"ADE  -- media: {statistics.mean(all_ade):.3f}m | mediana: {statistics.median(all_ade):.3f}m")
print(f"minADE (melhor de {N_SAMPLES} amostras) -- media: {statistics.mean(all_min_ade):.3f}m | "
      f"mediana: {statistics.median(all_min_ade):.3f}m")
print("\nReferencia (documentada no fork kimsunguk0, Alpamayo 1.5 10B real, FP8, "
      "veiculo real): ADE 0.17-0.19m")
print("Nosso resultado e' do backbone 2B, treino de 3 epocas, GT direto (sem KD do 10B) "
      "-- comparar com essa referencia da' o contexto de qualidade relativa.")
