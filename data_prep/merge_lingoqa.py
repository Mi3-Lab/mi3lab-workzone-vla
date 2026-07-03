"""Merge lingoqa_pathways + lingoqa_scene into lingoqa_combined.

Image layout in lingoqa_combined/:
  images/pathways/ -> ../../pathways/images/   (3,117 images)
  images/scene/    -> ../../scene/images/       (7,416 images)
"""

import os
import numpy as np
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
OUT_DIR = os.path.join(BASE, "lingoqa_combined")
os.makedirs(OUT_DIR, exist_ok=True)

# Create image symlinks
img_dir = os.path.join(OUT_DIR, "images")
os.makedirs(img_dir, exist_ok=True)

for name, src in [
    ("pathways", os.path.join(BASE, "pathways", "images")),
    ("scene",    os.path.join(BASE, "scene",    "images")),
]:
    link = os.path.join(img_dir, name)
    if not os.path.exists(link):
        os.symlink(src, link)
        print(f"Symlink: {link} -> {src}")


def prefix_images(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rewrite image paths to use images/{prefix}/filename."""
    df = df.copy()
    df["images"] = df["images"].apply(
        lambda imgs: np.array([
            f"images/{prefix}/{os.path.basename(p)}" for p in imgs
        ])
    )
    return df


for split in ("train", "val"):
    pathways = pd.read_parquet(os.path.join(BASE, "lingoqa_pathways", f"{split}.parquet"))
    scene    = pd.read_parquet(os.path.join(BASE, "lingoqa_scene",    f"{split}.parquet"))

    pathways = prefix_images(pathways, "pathways")
    scene    = prefix_images(scene,    "scene")

    combined = pd.concat([pathways, scene], ignore_index=True)
    combined["question_id"] = combined["question_id"].astype(str)

    out = os.path.join(OUT_DIR, f"{split}.parquet")
    combined.to_parquet(out, index=False)

    print(f"\n{split}:")
    print(f"  pathways: {len(pathways):,} pares ({pathways['segment_id'].nunique():,} imgs)")
    print(f"  scene:    {len(scene):,} pares ({scene['segment_id'].nunique():,} imgs)")
    print(f"  combined: {len(combined):,} pares ({combined['segment_id'].nunique():,} imgs)")
    print(f"  -> {out}")

print("\nDataset combinado pronto.")
