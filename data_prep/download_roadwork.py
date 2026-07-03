#!/usr/bin/env python3
"""
Download ROADWork dataset from HuggingFace.

Paper: ROADWork — A Dataset and Benchmark for Learning to Recognize,
       Observe, Analyze and Drive Through Work Zones (ICCV 2025)
Source: https://huggingface.co/datasets/anuragxel/roadwork-dataset

Directory layout after extraction:
  raw/             downloaded zip files
  scene/           images + annotations + sem_seg
  pathways/        traj_images + traj_annotations
  pathways_dense/  traj_images_dense + traj_annotations_dense
  videos/          multi-part zip assembled and extracted
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Redirect ALL HuggingFace caches to /data/ — never touch home directory.
_HF_CACHE = str(Path(__file__).resolve().parents[2] / ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _HF_CACHE + "/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_CACHE + "/hub")
os.environ.setdefault("XDG_CACHE_HOME", _HF_CACHE)

from huggingface_hub import hf_hub_download

REPO_ID = "anuragxel/roadwork-dataset"
REPO_TYPE = "dataset"

BASE = Path(__file__).parent
RAW = BASE / "raw"

# Groups: (files_to_download, extract_destination)
CORE_SCENE = (
    ["images.zip", "annotations.zip", "sem_seg_labels.zip", "roadwork_raw_meta.zip"],
    BASE / "scene",
)

PATHWAYS = (
    ["traj_images.zip", "traj_annotations.zip"],
    BASE / "pathways",
)

PATHWAYS_DENSE = (
    ["traj_images_dense.zip", "traj_annotations_dense.zip"],
    BASE / "pathways_dense",
)

VIDEOS = (
    [
        "videos_compressed.zip",
        "videos_compressed.z01",
        "videos_compressed.z02",
        "videos_compressed.z03",
        "videos_compressed.z04",
        "videos_compressed.z05",
        "videos_compressed.z06",
        "videos_compressed.z07",
    ],
    BASE / "videos",
)

DISCOVERED = (
    ["discovered_images.zip", "discovered_subsets_with_annotations.zip"],
    BASE / "scene",
)

GROUPS = {
    "scene": CORE_SCENE,
    "pathways": PATHWAYS,
    "pathways_dense": PATHWAYS_DENSE,
    "videos": VIDEOS,
    "discovered": DISCOVERED,
}


def download_file(filename: str) -> Path:
    """Download a single file to raw/, return local path."""
    local = RAW / filename
    if local.exists():
        print(f"  [skip] {filename} already in raw/")
        return local
    print(f"  [download] {filename} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(RAW),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a simple zip file."""
    print(f"  [extract] {zip_path.name} -> {dest}/")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def assemble_and_extract_videos(raw_dir: Path, dest: Path) -> None:
    """
    Assemble the multi-part zip (videos_compressed.zip + .z01–.z07) using
    the `zip` CLI tool (handles split archives), then extract.
    """
    assembled = raw_dir / "videos_assembled.zip"
    if not assembled.exists():
        print("  [assemble] joining multi-part zip with 'zip -FF' ...")
        cmd = [
            "zip", "-FF",
            str(raw_dir / "videos_compressed.zip"),
            "--out", str(assembled),
        ]
        subprocess.run(cmd, check=True)

    print(f"  [extract] videos_assembled.zip -> {dest}/")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        extract_zip(assembled, dest)
    except zipfile.BadZipFile:
        # Fallback: use unzip CLI which handles split archives natively
        print("  [fallback] using unzip CLI ...")
        subprocess.run(
            ["unzip", "-o", str(raw_dir / "videos_compressed.zip"), "-d", str(dest)],
            check=True,
        )


def run_group(name: str, skip_extract: bool) -> None:
    files, dest = GROUPS[name]
    print(f"\n=== {name.upper()} ===")
    local_paths = [download_file(f) for f in files]

    if skip_extract:
        print("  [skip extract] --no-extract flag set")
        return

    if name == "videos":
        assemble_and_extract_videos(RAW, dest)
    else:
        for p in local_paths:
            if p.suffix == ".zip":
                extract_zip(p, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ROADWork dataset")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(GROUPS.keys()) + ["all"],
        default=["all"],
        help="Which groups to download (default: all)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download zip files only, skip extraction",
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    targets = list(GROUPS.keys()) if "all" in args.groups else args.groups
    for name in targets:
        run_group(name, skip_extract=args.no_extract)

    print("\nDone.")
    print(f"  raw/             -> {RAW}")
    print(f"  scene/           -> {BASE / 'scene'}")
    print(f"  pathways/        -> {BASE / 'pathways'}")
    print(f"  pathways_dense/  -> {BASE / 'pathways_dense'}")
    print(f"  videos/          -> {BASE / 'videos'}")


if __name__ == "__main__":
    main()
