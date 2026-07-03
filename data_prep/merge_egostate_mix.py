"""Monta lingoqa_egomix: egostate (novo, upsampled) + amostra do lingoqa_roadwork.

Mix para o fine-tune stage5 continuar do checkpoint-8622 sem esquecer as
habilidades Q&A existentes (traffic/active/description) que o v9 usa.

Estrutura de imagens (um único data_root com symlinks):
  images/       → lingoqa_roadwork/images   (linhas roadwork mantêm caminho)
  scene_images/ → scene/images              (linhas egostate INSIDE/APPROACHING)
  bdd/          → bdd100k                   (linhas egostate OUTSIDE)
"""
import os
import random
import uuid

import pandas as pd

BASE    = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
OUT_DIR = os.path.join(BASE, "lingoqa_egomix")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

N_ROADWORK_TRAIN = 8000
EGOSTATE_UPSAMPLE = 4

for name, target in [("images", "../lingoqa_roadwork/images"),
                     ("scene_images", "../scene/images"),
                     ("bdd", "../bdd100k")]:
    link = os.path.join(OUT_DIR, name)
    if not os.path.exists(link):
        os.symlink(target, link)
        print(f"Symlink: {name} → {target}")


def rewrite_ego_path(p):
    if p.startswith("../bdd100k/"):
        return p.replace("../bdd100k/", "bdd/")
    if p.startswith("images/"):
        return p.replace("images/", "scene_images/", 1)
    return p


def norm_ego(df, upsample):
    df = df.copy()
    df["images"] = df["images"].apply(lambda arr: [rewrite_ego_path(p) for p in arr])
    df["question_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["segment_id"]  = df["question_id"]
    df = df[["question_id", "segment_id", "images", "question", "answer"]]
    return pd.concat([df] * upsample, ignore_index=True)


for split, n_road, ego_up in [("train", N_ROADWORK_TRAIN, EGOSTATE_UPSAMPLE),
                              ("val",   500,              1)]:
    road = pd.read_parquet(os.path.join(BASE, "lingoqa_roadwork", f"{split}.parquet"))
    if len(road) > n_road:
        road = road.sample(n=n_road, random_state=42)
    ego = norm_ego(pd.read_parquet(os.path.join(BASE, "lingoqa_egostate", f"{split}.parquet")),
                   ego_up)
    mix = pd.concat([road[["question_id", "segment_id", "images", "question", "answer"]], ego],
                    ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    mix.to_parquet(out, index=False)
    print(f"{split}: roadwork={len(road)} egostate={len(ego)} total={len(mix)} → {out}")
