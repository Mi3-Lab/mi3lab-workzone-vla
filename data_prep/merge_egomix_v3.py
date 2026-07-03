"""lingoqa_egomix_v3: retenção de direção GERAL + especialização work zone.

Stage 6 — o rebalanceamento. Motivação (usuário): work zone é um long tail;
o modelo deve ser um VLA de direção geral que TAMBÉM domina work zones.
Teste 170550: nosso 2B colapsou em templates em cenas normais.

Mix:
  - general (self-distill do Cosmos-2B base): raciocínio de direção geral —
    a âncora anti-esquecimento. SEM upsample (diversidade > repetição).
  - roadwork Q&A (6k amostra): habilidades de cena work zone
  - egostate ×3: estado ego-cêntrico (Stage 5)
  - signtext ×2: leitura de placas (Stage 5.1)
"""
import os
import random
import uuid

import pandas as pd

BASE    = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
OUT_DIR = os.path.join(BASE, "lingoqa_egomix_v3")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

for name, target in [("images", "../lingoqa_roadwork/images"),
                     ("scene_images", "../scene/images"),
                     ("bdd", "../bdd100k"),
                     ("general_frames", "../lingoqa_general/video_frames")]:
    link = os.path.join(OUT_DIR, name)
    if not os.path.exists(link):
        os.symlink(target, link)


def rewrite(p):
    if p.startswith("../bdd100k/"):
        return p.replace("../bdd100k/", "bdd/")
    if p.startswith("video_frames/"):
        return p.replace("video_frames/", "general_frames/", 1)
    if p.startswith("images/") and not p.startswith("images/train") and not p.startswith("images/val"):
        return p.replace("images/", "scene_images/", 1)
    return p


def norm(df, upsample):
    df = df.copy()
    df["images"] = df["images"].apply(lambda arr: [rewrite(p) for p in arr])
    df["question_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["segment_id"]  = df["question_id"]
    df = df[["question_id", "segment_id", "images", "question", "answer"]]
    return pd.concat([df] * upsample, ignore_index=True)


def rewrite_general(df):
    df = df.copy()
    df["images"] = df["images"].apply(
        lambda arr: [p if p.startswith("bdd/") else "general_frames/" + os.path.basename(p) for p in arr])
    return df


for split, n_road, ego_up, sign_up in [("train", 6000, 3, 2), ("val", 400, 1, 1)]:
    road = pd.read_parquet(os.path.join(BASE, "lingoqa_roadwork", f"{split}.parquet"))
    if len(road) > n_road:
        road = road.sample(n=n_road, random_state=42)
    ego  = norm(pd.read_parquet(os.path.join(BASE, "lingoqa_egostate", f"{split}.parquet")), ego_up)
    sign = norm(pd.read_parquet(os.path.join(BASE, "lingoqa_signtext", f"{split}.parquet")), sign_up)
    gen  = norm(rewrite_general(pd.read_parquet(os.path.join(BASE, "lingoqa_general", f"{split}.parquet"))), 1)
    mix = pd.concat(
        [road[["question_id", "segment_id", "images", "question", "answer"]], ego, sign, gen],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    mix.to_parquet(out, index=False)
    print(f"{split}: roadwork={len(road)} egostate={len(ego)} signtext={len(sign)} "
          f"general={len(gen)} total={len(mix)} → {out}")
