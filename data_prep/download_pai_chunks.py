"""Baixa os chunks essenciais do PhysicalAI-AV para /data (Fase A — KD geral).

Chunks = os 19 da receita oficial stage2_nav do alpamayo-recipes.
Por chunk: 4 câmeras usadas pelo Alpamayo + egomotion + calibração (~9 GB).
Mais metadados de raiz (clip_index, features, reasoning) uma vez.

Destino: /data/wesleyferreiramaia/wokzone-alpamayo/data/PhysicalAI-AV
(layout local_dir esperado por PhysicalAIAVDatasetLocalInterface).
"""
import os

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HOME": f"{BASE}/.hf_cache",
    "HF_HUB_OFFLINE": "0",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
})

from huggingface_hub import snapshot_download

LOCAL_DIR = f"{BASE}/data/PhysicalAI-AV"
CHUNKS = [214, 224, 276, 317, 420, 727, 728, 968, 982,
          1519, 1657, 1984, 2277, 2368, 2372, 2447, 2599, 2634, 2868]

CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]

patterns = ["clip_index.parquet", "features.csv", "README.md",
            "reasoning/ood_reasoning.parquet",
            "metadata/*"]
for c in CHUNKS:
    tag = f"chunk_{c:04d}"
    for cam in CAMERAS:
        patterns.append(f"camera/{cam}/{cam}.{tag}.zip")
    patterns.append(f"labels/egomotion/egomotion.{tag}.zip")
    patterns.append(f"calibration/*/*.{tag}.parquet")

print(f"[DL] {len(CHUNKS)} chunks → {LOCAL_DIR}")
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=patterns,
    max_workers=4,
)
print("[DONE] download completo")
