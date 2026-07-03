"""Convert ROADWork scene/ annotations to LingoQA parquet format.

Uses:
  - scene_description: real human-written scene description
  - scene_level_tags.travel_alteration: ground-truth lane status
  - COCO annotations: object counts/positions for inventory Q&A
  - Negatives: images where travel_alteration == ["None"]
"""

import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork"
SCENE_DIR = os.path.join(BASE, "scene")
OUT_DIR = os.path.join(BASE, "lingoqa_scene")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_LINK = os.path.join(OUT_DIR, "images")
if not os.path.exists(IMG_LINK):
    os.symlink(os.path.join(SCENE_DIR, "images"), IMG_LINK)
    print(f"Created symlink: {IMG_LINK}")

WORKER_CATEGORIES = {1, 14}   # Police Officer, Worker
HIGH_RISK = {1, 2, 8, 14}     # Police Officer, Police Vehicle, Work Vehicle, Worker

Q_DESCRIBE = [
    "Describe the work zone elements visible in this scene.",
    "What construction or road work activity is present in this scene?",
    "Provide a scene description for autonomous vehicle navigation.",
    "What road work or traffic control devices do you see in this scene?",
]

Q_ALTERATION = [
    "How is traffic flow affected in this scene?",
    "What is the lane or road alteration present in this scene?",
    "Describe the traffic alteration caused by the work zone.",
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

ALTERATION_DESCRIPTIONS = {
    "Partially Blocked": (
        "Lane partially blocked. One or more lanes are obstructed by work zone devices or activity. "
        "Reduce speed, merge away from blocked lane, and proceed with caution."
    ),
    "Fully Blocked": (
        "Lane or road fully blocked. Traffic cannot proceed in the affected lane or direction. "
        "Prepare to stop, follow detour signage, and watch for flagger instructions."
    ),
    "Lane Shift": (
        "Lane shift in effect. Traffic is redirected to an alternate path through the work zone. "
        "Follow temporary lane markings and reduce speed through the shifted section."
    ),
    "None": (
        "No lane alteration detected. Road appears clear of active work zone restrictions. "
        "Proceed at normal posted speed with standard attention."
    ),
    "Other": (
        "Non-standard traffic alteration present. Exercise caution and follow posted signage."
    ),
}


def build_alteration_answer(alterations: list[str]) -> str:
    if not alterations or alterations == ["None"]:
        return ALTERATION_DESCRIPTIONS["None"]
    parts = [ALTERATION_DESCRIPTIONS.get(a, ALTERATION_DESCRIPTIONS["Other"]) for a in alterations]
    if len(parts) == 1:
        return parts[0]
    return " Additionally: ".join(parts)


def build_object_answer(anns: list[dict], cat_map: dict[int, str]) -> str | None:
    counts: Counter = Counter()
    spatial: dict[str, list[str]] = {}

    for ann in anns:
        cat = cat_map.get(ann["category_id"], "Unknown")
        counts[cat] += 1
        bbox = ann.get("bbox")
        if bbox:
            x, y, w, h = bbox
            cx = x + w / 2
            pos = "left" if cx < 640 else ("right" if cx > 1280 else "center")
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
    cat_ids = {ann["category_id"] for ann in anns}
    has_workers = bool(cat_ids & WORKER_CATEGORIES)
    zone_type = "active" if has_workers else "passive"

    return (
        f"Total objects detected: {total}. "
        f"Breakdown: {'; '.join(parts)}. "
        f"This is a {zone_type} work zone"
        + (" with workers present — reduce speed and be prepared to stop."
           if has_workers else " with devices only — reduce speed and stay alert.")
    )


def build_active_answer(anns: list[dict], is_negative: bool) -> str:
    if is_negative:
        return (
            "Normal road conditions: no workers or work zone devices detected. "
            "Proceed at normal posted speed with standard attention."
        )
    cat_ids = {ann["category_id"] for ann in anns}
    has_workers = bool(cat_ids & WORKER_CATEGORIES)
    if has_workers:
        return (
            "Active work zone: workers are present. "
            "Reduce speed significantly, be prepared to stop, "
            "watch for flagger instructions and unexpected worker movement."
        )
    return (
        "Passive work zone: no workers visible, only control devices. "
        "Reduce speed and maintain heightened awareness, "
        "but no flagger interaction expected."
    )


def process_split(ann_file: str, split: str) -> pd.DataFrame:
    with open(ann_file) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    rows = []
    rng = random.Random(42)

    for img in data["images"]:
        img_id = img["id"]
        fname = img["file_name"]
        desc = img.get("scene_description", "").strip()
        tags = img.get("scene_level_tags", {})
        alterations = tags.get("travel_alteration", ["None"])
        anns = anns_by_img[img_id]

        is_negative = (alterations == ["None"] and not desc)

        images = [f"images/{fname}"]
        pairs = []

        if desc:
            pairs.append((rng.choice(Q_DESCRIBE), desc))

        alt_ans = build_alteration_answer(alterations)
        pairs.append((rng.choice(Q_ALTERATION), alt_ans))

        pairs.append((rng.choice(Q_ACTIVE), build_active_answer(anns, is_negative)))

        obj_ans = build_object_answer(anns, cat_map)
        if obj_ans:
            pairs.append((
                "What objects are present in this scene and where are they positioned?",
                obj_ans,
            ))

        for i, (question, answer) in enumerate(pairs):
            rows.append({
                "question_id": f"scene_{img_id}_{i}",
                "segment_id": f"scene_{img_id}",
                "images": np.array(images),
                "question": question,
                "answer": answer,
            })

    df = pd.DataFrame(rows)
    neg = sum(1 for img in data["images"]
              if img.get("scene_level_tags", {}).get("travel_alteration", []) == ["None"])
    print(f"\n{split}: {len(data['images'])} images → {len(df)} Q&A pairs")
    print(f"  Negatives (None): {neg} ({100*neg/len(data['images']):.1f}%)")
    print(f"  Pairs per image: {len(df)/len(data['images']):.1f}")
    return df


def main():
    train_file = os.path.join(SCENE_DIR, "annotations", "instances_train_gps_split.json")
    val_file   = os.path.join(SCENE_DIR, "annotations", "instances_val_gps_split.json")

    train_df = process_split(train_file, "train")
    val_df   = process_split(val_file,   "val")

    train_df.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(OUT_DIR, "val.parquet"), index=False)

    print(f"\nSalvo em: {OUT_DIR}")
    print(f"  train.parquet: {len(train_df)} pares")
    print(f"  val.parquet:   {len(val_df)} pares")

    print("\n── Exemplos ──")
    for _, row in train_df.head(3).iterrows():
        print(f"Q: {row['question']}")
        print(f"A: {row['answer'][:120]}...")
        print()


if __name__ == "__main__":
    main()
