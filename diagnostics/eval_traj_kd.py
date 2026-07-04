"""Avaliação head-to-head de trajetória: student KD (2B) vs teacher (10B puro).

Mesmos clipes de validação (chunks 2868, 3126 do PhysicalAI-AV), mesmas
métricas: minADE/minFDE (mínimo sobre K=6 amostras, distância XY em metros)
contra a trajetória real do motorista (ego_future_xyz, 64 steps @ 0.1s = 6.4s).

Referência anterior do teacher (job 170613): minADE 0.35–1.97 m.
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

from alpamayo.data.pai import PAIDataset
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

VAL_CHUNKS = [2868, 3126]
N_CLIPS    = 24          # subamostra uniforme dos clipes de val
N_SAMPLES  = 6           # K trajetórias por clipe (mesma config do MetricRunner)
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


def eval_model(name, ckpt_path, vlm_path):
    print(f"\n{'='*70}\n[{name}] carregando {ckpt_path}")
    model = TrainableReasoningVLA.from_alpamayo_checkpoint(
        checkpoint_path=ckpt_path, vlm_name_or_path=vlm_path,
    ).to(torch.bfloat16).cuda().eval()
    print(f"[{name}] VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    ds = PAIDataset(
        local_dir=f"{BASE}/data/PhysicalAI-AV",
        chunk_ids=VAL_CHUNKS,
        use_default_keyframe=True,
        vla_preprocess_args=dict(PREPROC),
        model_config=model.config,
    )
    idxs = np.linspace(0, len(ds) - 1, N_CLIPS).astype(int).tolist()
    print(f"[{name}] {len(ds)} clipes de val, avaliando {len(idxs)}")

    min_ades, min_fdes, clip_rows = [], [], []
    for i, idx in enumerate(idxs):
        sample = ds[idx]
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config, chat_template_version="r1_5"
        )
        batch = batch_to_cuda(batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = model.sample_trajectories_from_data(
                data=batch,
                top_p=TOP_P, temperature=TEMP,
                num_traj_samples=N_SAMPLES, num_traj_sets=1,
                max_generation_length=model.config.tokens_per_future_traj,
            )
        # pred_xyz: [1, 1, K, Tf, 3] — GT: [1, 1, 64, 3]
        pred = pred_xyz[0, 0].float().cpu().numpy()          # [K, Tf, 3]
        gt = batch["ego_future_xyz"][0, -1].float().cpu().numpy()  # [64, 3]
        T = min(pred.shape[1], gt.shape[0])
        err = np.linalg.norm(pred[:, :T, :2] - gt[None, :T, :2], axis=-1)  # [K, T]
        ade_k = err.mean(axis=1)      # [K]
        fde_k = err[:, -1]            # [K]
        min_ade, min_fde = float(ade_k.min()), float(fde_k.min())
        min_ades.append(min_ade)
        min_fdes.append(min_fde)
        clip_rows.append((sample["clip_id"][:8], min_ade, min_fde))
        print(f"  [{i+1:2d}/{len(idxs)}] clip {sample['clip_id'][:8]}  "
              f"minADE={min_ade:6.2f} m  minFDE={min_fde:6.2f} m", flush=True)

    a, f = np.array(min_ades), np.array(min_fdes)
    print(f"\n[{name}] RESUMO ({len(a)} clipes, K={N_SAMPLES}):")
    print(f"  minADE  media={a.mean():.3f} m  mediana={np.median(a):.3f} m  "
          f"p90={np.percentile(a,90):.3f} m  max={a.max():.3f} m")
    print(f"  minFDE  media={f.mean():.3f} m  mediana={np.median(f):.3f} m  "
          f"p90={np.percentile(f,90):.3f} m  max={f.max():.3f} m")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return clip_rows, a, f


student_rows, sa, sf = eval_model(
    "STUDENT 2B (kd_general_driving)",
    f"{BASE}/checkpoints/kd_general_driving",
    f"{BASE}/models/Cosmos-Reason2-2B",
)
teacher_rows, ta, tf = eval_model(
    "TEACHER 10B (Alpamayo-1.5 original)",
    f"{BASE}/models/Alpamayo-1.5-10B-A1-format",
    f"{BASE}/models/Cosmos-Reason2-8B",
)

print(f"\n{'='*70}\nCOMPARATIVO POR CLIPE (minADE m | minFDE m):")
print(f"{'clip':10s} {'student':>16s} {'teacher':>16s}")
for (cid, s_ade, s_fde), (_, t_ade, t_fde) in zip(student_rows, teacher_rows):
    print(f"{cid:10s} {s_ade:7.2f} | {s_fde:6.2f} {t_ade:7.2f} | {t_fde:6.2f}")

print(f"\nAGREGADO — student vs teacher:")
print(f"  minADE media : {sa.mean():.3f} vs {ta.mean():.3f} m "
      f"(gap {sa.mean()-ta.mean():+.3f})")
print(f"  minADE mediana: {np.median(sa):.3f} vs {np.median(ta):.3f} m")
print(f"  minFDE media : {sf.mean():.3f} vs {tf.mean():.3f} m")
print("\n=== EVAL TRAJ KD CONCLUIDO ===")
