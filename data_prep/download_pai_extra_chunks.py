"""Baixa 20 chunks EXTRA do PhysicalAI-AV, escolhidos por densidade de reasoning.

Cruzamento clip_index.parquet × reasoning/ood_reasoning.parquet: estes 20
chunks concentram +117 clipes com anotação de raciocínio oficial da NVIDIA
(7 → 124 no total, sobre os 19 chunks já baixados). ~106 GB.

Mesmo destino dos 19 chunks originais — snapshot_download é incremental
(pula arquivos já presentes).
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
CHUNKS = [15, 17, 34, 120, 133, 137, 139, 148, 149, 153,
          174, 270, 609, 1786, 1790, 1799, 1862, 2443, 3125, 3126]

CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]

patterns = []
for c in CHUNKS:
    tag = f"chunk_{c:04d}"
    for cam in CAMERAS:
        patterns.append(f"camera/{cam}/{cam}.{tag}.zip")
    patterns.append(f"labels/egomotion/egomotion.{tag}.zip")
    patterns.append(f"calibration/*/*.{tag}.parquet")

print(f"[DL] {len(CHUNKS)} chunks extra (densos em reasoning) → {LOCAL_DIR}")
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=patterns,
    max_workers=4,
)
print("[DONE] download completo")
