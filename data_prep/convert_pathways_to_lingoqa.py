"""Convert ROADWork pathways annotations to LingoQA parquet format.

Uses the OFFICIAL human-written descriptions from the ROADWork paper (ICCV 2025)
instead of synthetic Q&A generated from bounding boxes.

Each annotation has:
  - description: human-written scene description (or "No Description")
  - objects: detected objects with category + bbox + confidence
  - trajectory: 20 (x,y) waypoints in pixel space

Generates 4-6 Q&A pairs per image:
  1. Scene description (core label from paper)
  2. Object inventory (from detection list)
  3. Active/passive zone classification
  4. Navigation advice
  5. For "No Description" samples: explicit negative examples
"""

import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
PATHWAYS_DIR = os.path.join(BASE, "pathways")
OUT_DIR = os.path.join(BASE, "lingoqa_pathways")
os.makedirs(OUT_DIR, exist_ok=True)

# Symlink images so LingoQADataset can find them at {data_root}/images/...
IMG_LINK = os.path.join(OUT_DIR, "images")
if not os.path.exists(IMG_LINK):
    os.symlink(os.path.join(PATHWAYS_DIR, "images"), IMG_LINK)
    print(f"Created symlink: {IMG_LINK}")


# ── Question templates ────────────────────────────────────────────────────────
Q_DESCRIBE = [
    "Describe the work zone elements visible in this scene.",
    "What construction or road work activity is present in this scene?",
    "Provide a scene description for autonomous vehicle navigation.",
]

Q_OBJECTS = [
    "What objects are present in this scene and where are they positioned?",
    "Count and identify all work zone objects visible in this scene.",
    "List the road work objects detected and their positions relative to the vehicle.",
]

Q_ACTIVE = [
    "Is this an active work zone with workers present, or a passive zone?",
    "Are there workers present in this scene? What does this mean for vehicle speed?",
]

Q_SAFE = [
    "Is there active road work or construction in this scene?",
    "Describe the road conditions and any work zone activity in this scene.",
    "What does this scene indicate for autonomous vehicle navigation?",
]

WORKER_CATEGORIES = {"Worker", "Police Officer"}
HIGH_RISK_CATEGORIES = {"Worker", "Police Officer", "Work Vehicle"}


def build_object_answer(objects: list[dict]):
    """Build a natural object inventory from detection list."""
    if not objects:
        return None

    # Count by category, keep score >= 0.5
    counts: Counter = Counter()
    spatial: dict[str, list[str]] = {}

    for obj in objects:
        if obj.get("score", 1.0) < 0.45:
            continue
        cat = obj["category_id"]
        counts[cat] += 1

        # Rough spatial position from bbox
        bbox = obj.get("bbox", None)
        if bbox:
            x, y, w, h = bbox
            cx = x + w / 2
            if cx < 640:
                pos = "left"
            elif cx > 1280:
                pos = "right"
            else:
                pos = "center"
            spatial.setdefault(cat, []).append(pos)

    if not counts:
        return None

    parts = []
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        positions = spatial.get(cat, [])
        if positions:
            dominant = Counter(positions).most_common(1)[0][0]
            parts.append(f"{cnt} {cat}{'s' if cnt > 1 else ''} ({dominant} side)")
        else:
            parts.append(f"{cnt} {cat}{'s' if cnt > 1 else ''}")

    total = sum(counts.values())
    has_workers = any(c in WORKER_CATEGORIES for c in counts)
    zone_type = "active" if has_workers else "passive"

    return (
        f"Total objects detected: {total}. "
        f"Breakdown: {'; '.join(parts)}. "
        f"This is a {zone_type} work zone"
        + (" with workers present — reduce speed and be prepared to stop."
           if has_workers else " with devices only — reduce speed and stay alert.")
    )


def build_active_answer(objects: list[dict], description: str) -> str:
    """Determine active vs passive zone."""
    has_workers = any(
        obj.get("score", 1.0) >= 0.45 and obj["category_id"] in WORKER_CATEGORIES
        for obj in objects
    )
    if has_workers:
        return (
            "Active work zone: workers are present. "
            "Reduce speed significantly, be prepared to stop, "
            "watch for flagger instructions and unexpected worker movement."
        )
    elif description and description != "No Description":
        return (
            "Passive work zone: no workers visible, only control devices. "
            "Reduce speed and maintain heightened awareness, "
            "but no flagger interaction expected."
        )
    else:
        return (
            "Normal road conditions: no workers or work zone devices detected. "
            "Proceed at normal posted speed with standard attention."
        )


def build_negative_answers(sample_id: str) -> list[tuple[str, str]]:
    """For 'No Description' samples — explicit negative Q&A."""
    return [
        (random.choice(Q_SAFE),
         "No work zone activity detected in this scene. "
         "Road appears clear of construction equipment, workers, and temporary traffic control devices. "
         "Proceed at normal posted speed limit with standard attention."),
        (random.choice(Q_ACTIVE),
         "No workers or work zone devices detected. "
         "This appears to be a normal road segment with no active construction. "
         "Standard driving behavior applies."),
    ]


def process_split(annotations: list[dict], split: str) -> pd.DataFrame:
    rows = []
    rng = random.Random(42)

    for item in annotations:
        sample_id = item["id"]
        img_rel = item["image"]         # e.g. "images/boston_xxx_0050.jpg"
        description = item["description"].strip()
        objects = item.get("objects", [])
        is_negative = (description == "No Description")

        # LingoQADataset expects images as a list of relative paths
        images = [img_rel]

        if is_negative:
            pairs = build_negative_answers(sample_id)
        else:
            pairs = []

            # Q1: scene description (primary label from ROADWork paper)
            q = rng.choice(Q_DESCRIBE)
            pairs.append((q, description))

            # Q2: object inventory (from detection results)
            obj_ans = build_object_answer(objects)
            if obj_ans:
                q = rng.choice(Q_OBJECTS)
                pairs.append((q, obj_ans))

            # Q3: active vs passive
            q = rng.choice(Q_ACTIVE)
            pairs.append((q, build_active_answer(objects, description)))

        for i, (question, answer) in enumerate(pairs):
            rows.append({
                "question_id": f"{sample_id}_{i}",
                "segment_id": sample_id,
                "images": np.array(images),
                "question": question,
                "answer": answer,
            })

    df = pd.DataFrame(rows)
    print(f"\n{split}: {len(annotations)} images → {len(df)} Q&A pairs")
    neg = sum(1 for x in annotations if x["description"] == "No Description")
    print(f"  Negative (no description): {neg} ({100*neg/len(annotations):.1f}%)")
    print(f"  Pairs per image: {len(df)/len(annotations):.1f}")
    return df


def main():
    train_path = os.path.join(PATHWAYS_DIR, "annotations", "trajectories_train_equidistant.json")
    val_path   = os.path.join(PATHWAYS_DIR, "annotations", "trajectories_val_equidistant.json")

    with open(train_path) as f:
        train_ann = json.load(f)
    with open(val_path) as f:
        val_ann = json.load(f)

    train_df = process_split(train_ann, "train")
    val_df   = process_split(val_ann,   "val")

    train_out = os.path.join(OUT_DIR, "train.parquet")
    val_out   = os.path.join(OUT_DIR, "val.parquet")
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"\nSalvo em: {OUT_DIR}")
    print(f"  train.parquet: {len(train_df)} pares")
    print(f"  val.parquet:   {len(val_df)} pares")

    # Show sample Q&A
    print("\n── Exemplos de Q&A (positivos) ──")
    for _, row in train_df[~train_df["segment_id"].str.endswith("_0") == False].head(3).iterrows():
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
        print()

    print("── Exemplos de Q&A (negativos) ──")
    neg_rows = train_df[train_df["answer"].str.startswith("No work zone")]
    for _, row in neg_rows.head(2).iterrows():
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
        print()


if __name__ == "__main__":
    main()
