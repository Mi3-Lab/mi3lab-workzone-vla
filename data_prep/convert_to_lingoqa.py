#!/usr/bin/env python3
"""
ROADWork COCO → LingoQA parquet — SOTA work zone detection annotations.

Generates rich, diverse Q&A pairs per image using:
  - COCO bounding boxes → object positions (left/center/right, near/mid/far)
  - Object counts per category
  - Severity scoring (based on Worker proximity and lane obstruction)
  - Causal reasoning chains (IF...THEN)
  - 10 question types per image → ~53k train / 21k val pairs

Output layout:
  {output_dir}/train.parquet, val.parquet
  {output_dir}/images/{split}/{segment_id}/0.jpg  (symlinks)
"""

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCENE_DIR   = Path("/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork/scene")
OUTPUT_DIR  = Path("/data/wesleyferreiramaia/wokzone-alpamayo/data/roadwork/lingoqa_roadwork")
TRAIN_ANN   = SCENE_DIR / "annotations/instances_train_gps_split.json"
VAL_ANN     = SCENE_DIR / "annotations/instances_val_gps_split.json"

# Risk weight per category (higher = more dangerous to the AV)
RISK_WEIGHT = {
    "Worker":                              10,
    "Police Officer":                       8,
    "Work Vehicle":                         6,
    "Arrow Board":                          5,
    "Temporary Traffic Control Sign":       5,
    "Temporary Traffic Control Message Board": 4,
    "Cone":                                 3,
    "Drum":                                 3,
    "Barricade":                            3,
    "Barrier":                              3,
    "Fence":                                2,
    "Vertical Panel":                       2,
    "Tubular Marker":                       2,
    "Bike Lane":                            1,
    "Work Equipment":                       1,
    "Other Roadwork Objects":               1,
    "Police Vehicle":                       4,
}

IMG_W = 1920  # standard width in ROADWork


def x_to_zone(cx: float, img_w: float = IMG_W) -> str:
    rel = cx / img_w
    if rel < 0.33:
        return "left"
    elif rel < 0.67:
        return "center"
    else:
        return "right"


def area_to_distance(area: float, img_area: float = IMG_W * 1080) -> str:
    rel = area / img_area
    if rel > 0.05:
        return "close (≤20m)"
    elif rel > 0.01:
        return "mid-range (20–60m)"
    else:
        return "distant (>60m)"


def severity_score(cats_with_meta: list[dict]) -> tuple[int, str]:
    """Returns (score 1-5, label)."""
    score = 0
    for obj in cats_with_meta:
        cat = obj["category"]
        zone = obj["zone"]
        dist = obj["distance"]
        w = RISK_WEIGHT.get(cat, 1)
        # multiply by proximity
        if "close" in dist:
            w *= 3
        elif "mid" in dist:
            w *= 2
        # center lane is worst
        if zone == "center":
            w *= 2
        elif zone == "left":
            w = int(w * 1.2)
        score += w
    # normalize to 1-5
    if score == 0:
        return 1, "minimal"
    elif score < 15:
        return 2, "low"
    elif score < 40:
        return 3, "moderate"
    elif score < 80:
        return 4, "high"
    else:
        return 5, "critical"


def speed_recommendation(sev: int, travel_alt: list[str]) -> str:
    blocked = any("Blocked" in t for t in travel_alt)
    if sev >= 5:
        return "stop or ≤5 mph"
    elif sev == 4:
        return "≤10 mph" if blocked else "≤15 mph"
    elif sev == 3:
        return "≤25 mph"
    elif sev == 2:
        return "≤35 mph"
    else:
        return "normal posted limit with increased alertness"


def build_qa_pairs(img_meta: dict, objs: list[dict], cat_name: dict[int, str]) -> list[tuple[str, str]]:
    """Generate up to 10 diverse QA pairs for one image."""
    fname     = img_meta["file_name"]
    tags      = img_meta.get("scene_level_tags", {})
    desc      = img_meta.get("scene_description", "Work zone scene.")
    city      = img_meta.get("city_name", "unknown city")
    weather   = ", ".join(tags.get("weather", ["unknown"]))
    daytime   = tags.get("daytime", "unknown")
    env       = tags.get("scene_environment", "unknown")
    travel_alt = tags.get("travel_alteration", [])

    # Enrich objects with spatial metadata
    enriched = []
    for ann in objs:
        cx = ann["bbox"][0] + ann["bbox"][2] / 2
        enriched.append({
            "category": cat_name[ann["category_id"]],
            "zone":     x_to_zone(cx, img_meta.get("width", IMG_W)),
            "distance": area_to_distance(ann["area"], img_meta.get("width", IMG_W) * img_meta.get("height", 1080)),
            "area":     ann["area"],
        })

    # Counts per category
    counts: dict[str, int] = defaultdict(int)
    for e in enriched:
        counts[e["category"]] += 1

    sev_score, sev_label = severity_score(enriched)
    speed = speed_recommendation(sev_score, travel_alt)
    lane_status = "partially blocked" if any("Partial" in t for t in travel_alt) else \
                  "fully blocked" if any("Block" in t for t in travel_alt) else "open but affected"

    # Unique categories present
    cats_present = sorted(set(e["category"] for e in enriched))

    # Workers specifically
    workers = [e for e in enriched if e["category"] in ("Worker", "Police Officer")]
    worker_summary = ""
    if workers:
        locs = set(w["zone"] for w in workers)
        dists = set(w["distance"] for w in workers)
        worker_summary = (
            f"{len(workers)} worker(s) detected in the {'/'.join(locs)} lane area, "
            f"{', '.join(dists)}."
        )

    pairs: list[tuple[str, str]] = []

    # Q1 — full scene description
    pairs.append((
        "Provide a comprehensive description of all work zone elements in this scene for autonomous vehicle navigation.",
        f"{desc} "
        f"Scene: {env} environment in {city}, {weather} conditions, {daytime} lighting. "
        f"Lane status: {lane_status}. "
        f"Detected objects: {', '.join(f'{v} {k}(s)' for k, v in sorted(counts.items())) if counts else 'none identified'}. "
        f"Overall hazard severity: {sev_score}/5 ({sev_label}). "
        f"Recommended speed: {speed}."
    ))

    # Q2 — severity and recommended action
    pairs.append((
        "What is the danger level of this work zone for an autonomous vehicle, and what immediate actions are required?",
        f"Danger level: {sev_score}/5 ({sev_label}). "
        f"{worker_summary + ' ' if worker_summary else ''}"
        f"Lane obstruction: {lane_status}. "
        f"Required actions: reduce speed to {speed}; "
        f"{'prepare to stop or change lanes; ' if sev_score >= 4 else ''}"
        f"increase following distance; scan for workers and equipment at all times."
    ))

    # Q3 — object counting
    if counts:
        count_str = "; ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(counts.items()))
        pairs.append((
            "Count and list all traffic control devices and work zone objects visible in this scene.",
            f"Total objects detected: {sum(counts.values())}. "
            f"Breakdown: {count_str}. "
            f"These indicate an active work zone requiring reduced speed and heightened awareness."
        ))

    # Q4 — spatial localization
    if enriched:
        spatial_desc = []
        for cat in cats_present:
            objs_cat = [e for e in enriched if e["category"] == cat]
            zones = sorted(set(e["zone"] for e in objs_cat))
            spatial_desc.append(f"{len(objs_cat)} {cat}(s) on the {'/'.join(zones)}")
        pairs.append((
            "Where are the work zone elements positioned relative to the vehicle's path? "
            "Which elements are in the direct path of travel?",
            f"Spatial layout: {'; '.join(spatial_desc)}. "
            f"{'DIRECT PATH HAZARD: elements detected in center lane — evasive action required. ' if any(e['zone'] == 'center' for e in enriched) else ''}"
            f"Lane status: {lane_status}."
        ))

    # Q5 — worker safety (only if workers present)
    if workers:
        pairs.append((
            "Are there any workers or pedestrians in or near the active travel lane? "
            "What is the risk to human safety and what must the vehicle do?",
            f"{worker_summary} "
            f"{'CRITICAL: worker(s) in or near active lane. Reduce speed to ≤10 mph immediately and be prepared to stop.' if any(w['zone'] in ('center', 'left') for w in workers) else 'Workers detected near the road. Reduce speed and maintain safe distance.'} "
            f"Do not proceed if workers are within 5m of the vehicle path."
        ))

    # Q6 — causal reasoning chain
    if sev_score >= 3:
        cause = f"{'Workers in active lane' if workers else 'Multiple traffic control devices blocking lane'}"
        pairs.append((
            "Using causal reasoning, explain what could happen if the autonomous vehicle does not respond "
            "correctly to this work zone, and what the correct response is.",
            f"CAUSE: {cause} detected at {sev_label} severity. "
            f"IF vehicle maintains speed: risk of collision with {'worker(s)' if workers else 'equipment/barriers'}, "
            f"potential injury/fatality and vehicle damage. "
            f"CORRECT RESPONSE: reduce speed to {speed}; "
            f"{'change to unobstructed lane if available; ' if 'Blocked' in str(travel_alt) else ''}"
            f"increase following distance to ≥50m; activate hazard lights if speed drops below 20 mph; "
            f"yield to all workers and flaggers."
        ))

    # Q7 — traffic control sign specific
    signs = [e for e in enriched if "Sign" in e["category"] or "Board" in e["category"]]
    if signs:
        pairs.append((
            "What temporary traffic control signs or message boards are visible, and what do they indicate for vehicle speed and lane use?",
            f"{len(signs)} temporary traffic control device(s) detected "
            f"({'Arrow Board' if any('Arrow' in s['category'] for s in signs) else ''}"
            f"{'Traffic Control Sign' if any('Sign' in s['category'] for s in signs) else ''}"
            f"{'Message Board' if any('Board' in s['category'] for s in signs) else ''}). "
            f"These indicate: active work zone ahead; mandatory speed reduction to {speed}; "
            f"follow temporary lane markings; be prepared for lane closures."
        ))

    # Q8 — weather/visibility impact
    pairs.append((
        "How do the current weather and lighting conditions affect the safety risks of this work zone for an autonomous vehicle?",
        f"Conditions: {weather}, {daytime} lighting in {env} environment. "
        f"{'Reduced visibility increases stopping distance — add extra margin. ' if daytime in ('Night', 'Dawn', 'Dusk') else ''}"
        f"{'Wet/slippery road surface reduces braking performance. ' if any(w in weather.lower() for w in ('rain', 'wet', 'snow', 'fog')) else ''}"
        f"Base hazard severity {sev_score}/5 ({sev_label}). "
        f"Adjusted recommended speed given conditions: {speed}."
    ))

    # Q9 — active vs completed zone
    active_markers = [c for c in cats_present if c in ("Worker", "Arrow Board", "Work Vehicle", "Police Officer")]
    pairs.append((
        "Is this an active work zone with workers present, or a passive zone with only devices? "
        "How does this change the vehicle's required response?",
        f"{'ACTIVE work zone: ' + str(len(workers)) + ' worker(s) present.' if workers else 'Passive zone: no workers visible, only control devices.'} "
        f"{'Active zones require maximum caution: slow to ≤15 mph, be prepared to stop, follow flagger instructions.' if workers else 'Passive zones still require reduced speed and attention, but no flagger interaction.'} "
        f"Detected markers: {', '.join(active_markers) if active_markers else 'static devices only'}."
    ))

    # Q10 — binary detection
    has_cones    = "Cone" in counts
    has_workers  = bool(workers)
    has_signs    = bool(signs)
    has_barriers = any(c in counts for c in ("Barrier", "Barricade", "Fence", "Drum"))
    pairs.append((
        "Answer yes/no for each: (1) Are there cones? (2) Are there workers? "
        "(3) Are there temporary signs? (4) Are there barriers? "
        "Then provide the overall safety assessment.",
        f"(1) Cones: {'YES — ' + str(counts['Cone']) + ' detected' if has_cones else 'NO'}. "
        f"(2) Workers: {'YES — ' + str(len(workers)) + ' detected' if has_workers else 'NO'}. "
        f"(3) Temporary signs: {'YES — ' + str(len(signs)) + ' detected' if has_signs else 'NO'}. "
        f"(4) Barriers: {'YES — barriers/drums/fence detected' if has_barriers else 'NO'}. "
        f"Overall assessment: severity {sev_score}/5 ({sev_label}), recommend {speed}."
    ))

    return pairs


def load_coco(ann_path: Path):
    with open(ann_path) as f:
        coco = json.load(f)
    id_to_image  = {img["id"]: img for img in coco["images"]}
    cat_name     = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    return id_to_image, img_to_anns, cat_name


def build_records(id_to_image, img_to_anns, cat_name, split: str) -> list[dict]:
    records = []
    img_out_dir = OUTPUT_DIR / "images" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_meta in id_to_image.items():
        fname = img_meta["file_name"]
        stem  = Path(fname).stem
        src   = SCENE_DIR / "images" / fname
        if not src.exists():
            continue

        seg_dir = img_out_dir / stem
        seg_dir.mkdir(exist_ok=True)
        dest = seg_dir / "0.jpg"
        if not dest.exists():
            os.symlink(src.resolve(), dest)

        anns = img_to_anns.get(img_id, [])
        pairs = build_qa_pairs(img_meta, anns, cat_name)

        for question, answer in pairs:
            q_id = hashlib.md5(f"{stem}:{question}".encode()).hexdigest()
            records.append({
                "question_id": q_id,
                "segment_id":  stem,
                "images":      np.array([f"images/{split}/{stem}/0.jpg"]),
                "question":    question,
                "answer":      answer,
            })
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train annotations...")
    id_to_img_tr, anns_tr, cat_name = load_coco(TRAIN_ANN)
    print(f"  {len(id_to_img_tr)} images, {sum(len(v) for v in anns_tr.values())} annotations")

    print("Loading val annotations...")
    id_to_img_val, anns_val, _ = load_coco(VAL_ANN)
    print(f"  {len(id_to_img_val)} images")

    print("Building train records...")
    train_recs = build_records(id_to_img_tr, anns_tr, cat_name, "train")
    print(f"  {len(train_recs)} QA pairs")

    print("Building val records...")
    val_recs = build_records(id_to_img_val, anns_val, cat_name, "val")
    print(f"  {len(val_recs)} QA pairs")

    # Show a sample
    import random; random.seed(42)
    sample = random.choice(train_recs)
    print(f"\n=== Amostra ===\nSEGMENT: {sample['segment_id']}\nQ: {sample['question']}\nA: {sample['answer'][:400]}\n")

    train_df = pd.DataFrame(train_recs)
    val_df   = pd.DataFrame(val_recs)
    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR   / "val.parquet",   index=False)

    print(f"Saved:")
    print(f"  train.parquet — {len(train_df)} rows")
    print(f"  val.parquet   — {len(val_df)} rows")


if __name__ == "__main__":
    main()
