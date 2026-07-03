#!/usr/bin/env python3
"""
Fast parallel download of remaining ROADWork files.
Uses HF_HUB_DISABLE_XET=1 to bypass the XET backend that was stalling.
Downloads up to 4 files simultaneously.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Bypass XET — use plain HTTPS which is much more reliable
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/wesleyferreiramaia/wokzone-alpamayo/.hf_cache/hub"

from huggingface_hub import hf_hub_download

REPO_ID   = "anuragxel/roadwork-dataset"
RAW       = Path("/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork/raw")
WORKERS   = 4   # parallel downloads

# All files still needed (already-present files will be skipped)
PENDING = [
    "traj_images_dense.zip",        # 18.9 GB
    "videos_compressed.zip",        # 7.2 GB
    "videos_compressed.z01",        # 10 GB
    "videos_compressed.z02",        # 10 GB
    "videos_compressed.z03",        # 10 GB
    "videos_compressed.z04",        # 10 GB
    "videos_compressed.z05",        # 10 GB
    "videos_compressed.z06",        # 10 GB
    "videos_compressed.z07",        # 10 GB
    "discovered_images.zip",        # 0.5 GB
    "discovered_subsets_with_annotations.zip",  # 1.7 GB
]


def download_one(filename: str) -> tuple[str, float]:
    dest = RAW / filename
    if dest.exists():
        size = dest.stat().st_size / 1024**3
        return filename, size

    t0 = time.time()
    print(f"  ↓ START  {filename}", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(RAW),
    )
    elapsed = time.time() - t0
    size = (RAW / filename).stat().st_size / 1024**3
    speed = size / elapsed * 1024  # MB/s
    print(f"  ✓ DONE   {filename}  {size:.1f} GB  {elapsed/60:.1f}min  {speed:.0f} MB/s", flush=True)
    return filename, elapsed


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    todo = [f for f in PENDING if not (RAW / f).exists()]
    done = [f for f in PENDING if (RAW / f).exists()]

    print(f"Já completos ({len(done)}): {done}")
    print(f"Para baixar ({len(todo)}): {todo}\n")

    if not todo:
        print("Tudo já baixado!")
        return

    print(f"Iniciando {WORKERS} downloads em paralelo...\n")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, f): f for f in todo}
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                _, elapsed = fut.result()
            except Exception as e:
                print(f"  ✗ ERRO   {fname}: {e}", flush=True)

    total = (time.time() - t_start) / 60
    total_gb = sum((RAW / f).stat().st_size for f in PENDING if (RAW / f).exists()) / 1024**3
    print(f"\nConcluído em {total:.1f} min — {total_gb:.1f} GB em /raw/")


if __name__ == "__main__":
    main()
