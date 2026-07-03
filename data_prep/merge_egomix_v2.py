"""lingoqa_egomix_v2: egomix (retenção + egostate) + leitura de placas (signtext).

Mix para o Stage 5.1, continuando do checkpoint-1700 (Stage 5):
  - roadwork Q&A (8k amostra): retenção das habilidades de descrição/cena
  - egostate ×4: mantém o estado ego-cêntrico aprendido no Stage 5
  - signtext ×3: NOVA habilidade — ler texto das placas TTC

Symlinks (um único data_root):
  images/       → lingoqa_roadwork/images
  scene_images/ → scene/images   (egostate + signtext usam scene)
  bdd/          → bdd100k
"""
import os
import random
import uuid

import pandas as pd

BASE    = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
OUT_DIR = os.path.join(BASE, "lingoqa_egomix_v2")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

N_ROADWORK_TRAIN  = 8000
EGOSTATE_UPSAMPLE = 4
SIGNTEXT_UPSAMPLE = 3

for name, target in [("images", "../lingoqa_roadwork/images"),
                     ("scene_images", "../scene/images"),
                     ("bdd", "../bdd100k")]:
    link = os.path.join(OUT_DIR, name)
    if not os.path.exists(link):
        os.symlink(target, link)


def rewrite_scene_path(p):
    if p.startswith("../bdd100k/"):
        return p.replace("../bdd100k/", "bdd/")
    if p.startswith("images/"):
        return p.replace("images/", "scene_images/", 1)
    return p


def norm(df, upsample):
    df = df.copy()
    df["images"] = df["images"].apply(lambda arr: [rewrite_scene_path(p) for p in arr])
    df["question_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["segment_id"]  = df["question_id"]
    df = df[["question_id", "segment_id", "images", "question", "answer"]]
    return pd.concat([df] * upsample, ignore_index=True)


for split, n_road, ego_up, sign_up in [("train", N_ROADWORK_TRAIN, EGOSTATE_UPSAMPLE, SIGNTEXT_UPSAMPLE),
                                       ("val",   500,              1,                 1)]:
    road = pd.read_parquet(os.path.join(BASE, "lingoqa_roadwork", f"{split}.parquet"))
    if len(road) > n_road:
        road = road.sample(n=n_road, random_state=42)
    ego  = norm(pd.read_parquet(os.path.join(BASE, "lingoqa_egostate", f"{split}.parquet")), ego_up)
    sign = norm(pd.read_parquet(os.path.join(BASE, "lingoqa_signtext", f"{split}.parquet")), sign_up)
    mix = pd.concat(
        [road[["question_id", "segment_id", "images", "question", "answer"]], ego, sign],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    mix.to_parquet(out, index=False)
    print(f"{split}: roadwork={len(road)} egostate={len(ego)} signtext={len(sign)} total={len(mix)} → {out}")
