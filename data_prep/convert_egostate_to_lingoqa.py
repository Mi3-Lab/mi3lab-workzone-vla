"""Gera Q&A de estado EGO-CÊNTRICO a partir da geometria das anotações ROADWork.

Princípio (feedback do usuário): ver workers != estar na zona. O estado é a
posição do EGO em relação ao início da zona:
  - APPROACHING: elementos de work zone visíveis mas ainda longe (todos os
    objetos pequenos / perto do horizonte)
  - INSIDE: elementos imediatamente ao lado do veículo (objeto grande, borda
    inferior baixa na imagem)
  - OUTSIDE: sem work zone (negativos bdd100k)
  - EXITING: não rotulável em imagem única — emerge da dinâmica temporal (HMM)
    ou dos snippets de vídeo (futuro).

Proxy geométrico (weak supervision, só no treino):
  proximity = max sobre objetos de (y_bottom_bbox / img_height)
  INSIDE       se proximity >= 0.82  (objeto chega à parte de baixo da imagem)
  APPROACHING  se proximity <= 0.60  (tudo ainda perto do horizonte)
  faixa 0.60-0.82: ambígua → descartada (mantém só labels confiáveis)
"""
import json
import os
import random
import glob

import pandas as pd

BASE      = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
SCENE_DIR = os.path.join(BASE, "scene")
BDD_DIR   = os.path.join(BASE, "bdd100k")
OUT_DIR   = os.path.join(BASE, "lingoqa_egostate")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)

INSIDE_THR      = 0.82
APPROACHING_THR = 0.60

QUESTION = (
    "Regarding the road work zone, what is this vehicle's current position status: "
    "OUTSIDE, APPROACHING, or INSIDE?"
)
ANSWERS = {
    "OUTSIDE": (
        "OUTSIDE. No work zone elements near the vehicle. "
        "The road ahead is clear of construction activity."
    ),
    "APPROACHING": (
        "APPROACHING. Work zone elements are visible ahead but the vehicle has not "
        "reached them yet. Prepare to slow down and follow work zone guidance."
    ),
    "INSIDE": (
        "INSIDE. Work zone elements are immediately beside the vehicle; it is "
        "currently passing through the work zone. Maintain reduced speed and "
        "maximum caution."
    ),
}


def load_split(name):
    path = os.path.join(SCENE_DIR, "annotations", f"instances_{name}_gps_split.json")
    with open(path) as f:
        return json.load(f)


def label_from_geometry(img, anns):
    """Retorna (label, proximity) ou (None, prox) se ambíguo."""
    H = img["height"]
    if not anns:
        return None, 0.0
    prox = 0.0
    for a in anns:
        bbox = a.get("bbox")
        if not bbox:
            continue
        _, y, _, h = bbox
        prox = max(prox, (y + h) / H)
    if prox >= INSIDE_THR:
        return "INSIDE", prox
    if prox <= APPROACHING_THR and prox > 0:
        return "APPROACHING", prox
    return None, prox


rows = {"train": [], "val": []}
stats = {"train": {}, "val": {}}

for split, out_split in [("train", "train"), ("val", "val")]:
    try:
        data = load_split(split)
    except FileNotFoundError:
        print(f"[WARN] split {split} não encontrado, pulando")
        continue
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    n_ins, n_app, n_amb = 0, 0, 0
    for img in data["images"]:
        anns = anns_by_img.get(img["id"], [])
        label, prox = label_from_geometry(img, anns)
        if label is None:
            n_amb += 1
            continue
        if label == "INSIDE":
            n_ins += 1
        else:
            n_app += 1
        rows[out_split].append({
            "images":       [os.path.join("images", img["file_name"])],
            "question":     QUESTION,
            "answer":       ANSWERS[label],
            "question_type": "egostate",
            "proximity":    round(prox, 3),
        })
    stats[out_split] = {"INSIDE": n_ins, "APPROACHING": n_app, "ambiguous": n_amb}

# Negativos OUTSIDE do bdd100k (balanceados com a menor classe do treino)
bdd_imgs = sorted(glob.glob(os.path.join(BDD_DIR, "*.jpg")))
random.shuffle(bdd_imgs)
n_out_train = min(len(bdd_imgs) * 4 // 5,
                  min(stats["train"].get("INSIDE", 0), stats["train"].get("APPROACHING", 0)) or 500)
n_out_val   = min(len(bdd_imgs) - n_out_train, max(50, n_out_train // 8))

for i, path in enumerate(bdd_imgs[: n_out_train + n_out_val]):
    split = "train" if i < n_out_train else "val"
    rows[split].append({
        "images":       [os.path.relpath(path, OUT_DIR)],
        "question":     QUESTION,
        "answer":       ANSWERS["OUTSIDE"],
        "question_type": "egostate",
        "proximity":    0.0,
    })
    if split == "train":
        stats["train"]["OUTSIDE"] = stats["train"].get("OUTSIDE", 0) + 1
    else:
        stats["val"]["OUTSIDE"] = stats["val"].get("OUTSIDE", 0) + 1

# Symlink de imagens do scene (mesmo padrão do lingoqa_scene)
img_link = os.path.join(OUT_DIR, "images")
if not os.path.exists(img_link):
    os.symlink(os.path.join(SCENE_DIR, "images"), img_link)
    print(f"Symlink: {img_link}")

# Balancear: INSIDE domina (fotos tiradas na zona) — cap em 2x APPROACHING
for split in ["train", "val"]:
    ins  = [r for r in rows[split] if r["answer"].startswith("INSIDE")]
    rest = [r for r in rows[split] if not r["answer"].startswith("INSIDE")]
    n_app = sum(1 for r in rest if r["answer"].startswith("APPROACHING"))
    cap = max(2 * n_app, 100)
    if len(ins) > cap:
        random.shuffle(ins)
        ins = ins[:cap]
        stats[split]["INSIDE"] = f"{cap} (capped)"
    rows[split] = ins + rest

for split in ["train", "val"]:
    random.shuffle(rows[split])
    df = pd.DataFrame(rows[split])
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    df.to_parquet(out, index=False)
    print(f"{split}: {len(df)} exemplos → {out}")
    print(f"   {stats[split]}")

print("\nDistribuição de proximity (train INSIDE/APPROACHING):")
df_t = pd.DataFrame(rows["train"])
wz = df_t[df_t["proximity"] > 0]
print(wz["proximity"].describe())
