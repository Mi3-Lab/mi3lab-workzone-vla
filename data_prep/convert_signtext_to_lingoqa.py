"""Gera Q&A de LEITURA DE PLACAS a partir das anotações with_signs do ROADWork.

As anotações instances_*_gps_split_with_signs.json têm attributes.sign_text
com a transcrição humana ("ROAD WORK AHEAD", "DETOUR", ...). O fine-tune
Stage 5 destruiu o OCR zero-shot do Cosmos-Reason2 base (toda pergunta
fora do domínio cai nos templates ROADWork) — então leitura de placas
precisa ser ENSINADA com o vocabulário de perguntas do próprio pipeline.

Pergunta fixa (mesma string usada na inferência):
  "Read the text on the temporary traffic control signs in this scene."
Resposta:
  "Signs read: ROAD WORK AHEAD; DETOUR."  (textos legíveis, sem duplicatas)
  "No legible sign text in this scene."   (negativos: sem texto legível)
"""
import json
import os
import random

import pandas as pd

BASE    = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
OUT_DIR = os.path.join(BASE, "lingoqa_signtext")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

QUESTION = "Read the text on the temporary traffic control signs in this scene."
NEG_ANSWER = "No legible sign text in this scene."
MAX_NEGATIVES_FRAC = 0.35   # proporção de negativos no split final

img_link = os.path.join(OUT_DIR, "images")
if not os.path.exists(img_link):
    os.symlink(os.path.join(BASE, "scene", "images"), img_link)
    print(f"Symlink: {img_link}")

for split in ["train", "val"]:
    path = os.path.join(BASE, "scene", "annotations",
                        f"instances_{split}_gps_split_with_signs.json")
    d = json.load(open(path))
    imgs = {i["id"]: i for i in d["images"]}

    texts_by_img = {}
    has_sign_by_img = set()
    for a in d["annotations"]:
        at = a.get("attributes", {})
        if not isinstance(at, dict):
            continue
        if "sign_text" in at:
            has_sign_by_img.add(a["image_id"])
            t = (at.get("sign_text") or "").strip()
            if t and not at.get("text_occluded", False):
                texts_by_img.setdefault(a["image_id"], []).append(t.upper())

    rows = []
    # Positivos: imagens com pelo menos um texto legível
    for img_id, texts in texts_by_img.items():
        uniq = sorted(set(texts))
        answer = "Signs read: " + "; ".join(uniq) + "."
        rows.append({
            "images":   [os.path.join("images", imgs[img_id]["file_name"])],
            "question": QUESTION,
            "answer":   answer,
        })
    n_pos = len(rows)

    # Negativos: imagens COM placas mas SEM texto legível (occluded/vazio)
    neg_ids = [i for i in has_sign_by_img if i not in texts_by_img]
    random.shuffle(neg_ids)
    n_neg = min(len(neg_ids), int(n_pos * MAX_NEGATIVES_FRAC / (1 - MAX_NEGATIVES_FRAC)))
    for img_id in neg_ids[:n_neg]:
        rows.append({
            "images":   [os.path.join("images", imgs[img_id]["file_name"])],
            "question": QUESTION,
            "answer":   NEG_ANSWER,
        })

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    df.to_parquet(out, index=False)
    print(f"{split}: {n_pos} positivos + {n_neg} negativos = {len(df)} → {out}")
