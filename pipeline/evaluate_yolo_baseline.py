#!/usr/bin/env python3
"""YOLO-champion baseline over the SAME eval protocol as evaluate_cascade.py.

Faithful headless port of the state logic actually shipped in
~/workzone/scripts/jetson_app.py (the ESV 2026 winning system, `make
workzone`): YOLO detections -> per-category counts -> weighted score ->
adaptive EMA -> 4-state machine with hysteresis. Ported pieces and their
sources, verified line-by-line against jetson_app.py:

  - yolo_frame_score / weights   (jetson_app.py:604, configs/jetson_config.yaml weights_yolo)
  - evidence + adaptive_alpha    (jetson_app.py:1273-1275)
  - update_state                 (jetson_app.py:644)
  - thresholds                   (configs/jetson_config.yaml: enter 0.5 / exit 0.3 /
                                  approach 0.25 / min_inside 6 / min_out 20 / ema_alpha 0.1)
  - model params                 (imgsz 960, conf 0.25, iou 0.7)

Also ports, when --clip is passed, the two optional fusion branches: global
CLIP fusion + per-cue CLIP verification (jetson_app.py:1279-1284, 444-543,
CUE_PROMPTS at :389) and the orange-pixel context boost (jetson_app.py:1287-
1295). Without --clip this is the YOLO-only configuration (the app's
--disable-clip mode).

Runs over the same calibration/validation splits, same human ground truth,
and the same simulated camera clock as the VLM harness: each frame's real
processing latency advances the video position, so each system samples as
densely as its own speed allows — the YOLO system's speed advantage is part
of the honest comparison.

Must run with the workzone venv (numpy<2 for ultralytics/torch):
    ~/workzone/venv/bin/python pipeline/evaluate_yolo_baseline.py --split validation

NOTE: do not run while the VLM evaluation is running — both use the GPU and
would distort each other's latency-driven sampling.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO

from paper_metrics import (EventLevelAccumulator, TimingOffsetAccumulator,
                            TransitionToleranceAccumulator, print_paper_report)

# ── CLIP fusion config (jetson_config.yaml `fusion:`) ────────────────────────
CLIP_WEIGHT = 0.35
CLIP_TRIGGER_TH = 0.2
CLIP_INTERVAL = 3
PER_CUE_TH = 0.05
PER_CUE_INTERVAL = 3
ORANGE_WEIGHT = 0.25
CONTEXT_TRIGGER_BELOW = 0.5
ORANGE_H_LOW, ORANGE_H_HIGH = 5, 25
CLIP_POS_TEXT = ("driving through a road construction work zone with orange barrels, "
                  "traffic cones, concrete barriers, lane closure signs, and construction workers")
CLIP_NEG_TEXT = ("driving on a clear normal road or highway with regular traffic, no "
                  "construction, no orange cones, no work zone barriers")

# Per-cue CLIP prompts (jetson_app.py:389 CUE_PROMPTS, verbatim)
CUE_PROMPTS = {
    "channelization": {
        "pos": ["traffic cone on road", "orange construction barrel on asphalt",
                "striped barricade on road", "road barrier", "vertical panel marker"],
        "neg": ["tree trunk", "street light pole", "mailbox", "pedestrian", "car wheel",
                "fire hydrant", "electricity pole", "bush"],
        "inactive": ["traffic cones stacked on a truck bed", "cones stored in a pile",
                     "construction barrels on a trailer", "equipment in storage yard"],
    },
    "workers": {
        "pos": ["construction worker in high-visibility safety vest",
                "person wearing hard hat and safety gear", "road worker flagging traffic"],
        "neg": ["pedestrian in casual clothes", "business person in suit", "runner",
                "cyclist", "mannequin", "statue"],
    },
    "vehicles": {
        "pos": ["yellow construction excavator", "dump truck on road",
                "pickup truck with flashing amber lights", "road roller", "utility work truck"],
        "neg": ["sedan car", "family suv", "sports car", "motorcycle", "city bus", "taxi"],
    },
    "ttc_signs": {
        "pos": ["orange diamond construction sign facing camera", "road work ahead sign",
                "speed limit sign facing camera", "white rectangular regulatory sign"],
        "neg": ["commercial billboard advertisement", "shop sign", "street name sign",
                "parking sign", "restaurant sign"],
        "inactive": ["back of a road sign", "grey metal sign back", "sign facing away",
                     "oblique sign edge"],
    },
    "message_board": {
        "pos": ["electronic arrow board trailer with lights on",
                "variable message sign displaying text", "digital traffic sign"],
        "neg": ["parked cargo trailer", "billboard", "back of a truck", "container"],
        "inactive": ["message board turned off", "black screen message board",
                     "folded arrow board"],
    },
}

# ── Category sets (workzone/src/workzone/apps/streamlit_utils.py) ───────────
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}

WEIGHTS = {"bias": -0.35, "channelization": 0.9, "workers": 0.8,
           "vehicles": 0.5, "ttc_signs": 0.7, "message_board": 0.6}
F_CONF = {"enter_th": 0.5, "exit_th": 0.3, "approach_th": 0.25,
          "min_inside_frames": 6, "min_out_frames": 20, "ema_alpha": 0.1}

STATE_MAP = {"OUT": "outside", "APPROACHING": "approaching",
             "INSIDE": "inside", "EXITING": "exiting"}
STATE_ORDER = ["outside", "approaching", "inside", "exiting"]


def clamp01(x):
    return max(0.0, min(1.0, x))


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def ema(prev, x, alpha):
    return x if prev is None else alpha * x + (1.0 - alpha) * prev


def adaptive_alpha(evidence, alpha_min, alpha_max):
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


class ClipFusion:
    """Faithful port of jetson_app.py's global CLIP fusion + PerCueVerifier
    (lines 444-543, 1279-1295, 1520-1534). Loads open_clip ViT-B-32 (openai),
    the global pos/neg text embeddings, and per-category CUE_PROMPTS
    embeddings used to re-verify individual YOLO detections."""

    def __init__(self, device="cuda"):
        import open_clip
        import torch
        self.torch = torch
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", cache_dir=os.path.expanduser("~/workzone/weights/clip"))
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        toks = tokenizer([CLIP_POS_TEXT, CLIP_NEG_TEXT]).to(device)
        with torch.no_grad():
            txt = self.model.encode_text(toks)
        self.pos_emb, self.neg_emb = txt[0], txt[1]

        self.cue_embeddings = {}
        for category, prompts in CUE_PROMPTS.items():
            pos_toks = tokenizer(prompts["pos"]).to(device)
            neg_toks = tokenizer(prompts["neg"]).to(device)
            with torch.no_grad():
                pos_e = self.model.encode_text(pos_toks)
                pos_e = pos_e / (pos_e.norm(dim=-1, keepdim=True) + 1e-8)
                pos_mean = pos_e.mean(dim=0)
                pos_mean = pos_mean / (pos_mean.norm() + 1e-8)

                neg_e = self.model.encode_text(neg_toks)
                neg_e = neg_e / (neg_e.norm(dim=-1, keepdim=True) + 1e-8)
                neg_mean = neg_e.mean(dim=0)
                neg_mean = neg_mean / (neg_mean.norm() + 1e-8)

                inactive_mean = None
                if "inactive" in prompts:
                    inact_toks = tokenizer(prompts["inactive"]).to(device)
                    inact_e = self.model.encode_text(inact_toks)
                    inact_e = inact_e / (inact_e.norm(dim=-1, keepdim=True) + 1e-8)
                    inactive_mean = inact_e.mean(dim=0)
                    inactive_mean = inactive_mean / (inactive_mean.norm() + 1e-8)
            self.cue_embeddings[category] = (pos_mean, neg_mean, inactive_mean)

    def global_score(self, frame_bgr):
        """jetson_app.py:634 clip_frame_score."""
        small = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            img = self.model.encode_image(x)
            img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
            return float((img @ self.pos_emb.unsqueeze(-1)).squeeze().item()
                         - (img @ self.neg_emb.unsqueeze(-1)).squeeze().item())

    def verify_batch(self, crops_bgr, categories):
        """jetson_app.py:492 PerCueVerifier.verify_batch."""
        if not crops_bgr:
            return []
        inputs, valid_indices = [], []
        for i, (crop, cat) in enumerate(zip(crops_bgr, categories)):
            if cat in self.cue_embeddings and crop.size > 0:
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                inputs.append(self.preprocess(Image.fromarray(rgb)))
                valid_indices.append(i)
        if not inputs:
            return [0.0] * len(crops_bgr)

        img_batch = self.torch.stack(inputs).to(self.device)
        with self.torch.no_grad(), self.torch.autocast(device_type="cuda", enabled=True):
            img_embs = self.model.encode_image(img_batch)
            img_embs = img_embs / (img_embs.norm(dim=-1, keepdim=True) + 1e-8)

        scores = [0.0] * len(crops_bgr)
        for i, idx in enumerate(valid_indices):
            cat = categories[idx]
            pos_emb, neg_emb, inactive_emb = self.cue_embeddings[cat]
            emb = img_embs[i]
            sim_pos = float(self.torch.dot(emb, pos_emb))
            sim_neg = float(self.torch.dot(emb, neg_emb))
            reject_score = sim_neg
            if inactive_emb is not None:
                sim_inactive = float(self.torch.dot(emb, inactive_emb))
                if sim_inactive > sim_pos:
                    scores[idx] = -1.0
                    continue
                reject_score = max(sim_neg, sim_inactive)
            scores[idx] = sim_pos - reject_score
        return scores


def orange_context_score(frame_bgr):
    """jetson_app.py:1287-1295 context boost (orange-pixel ratio)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([ORANGE_H_LOW, 80, 50]), np.array([ORANGE_H_HIGH, 255, 255]))
    ratio = np.count_nonzero(mask) / mask.size
    return clamp01(float(logistic(30.0 * (ratio - 0.08))))


def get_cue_category(name):
    if name in CHANNELIZATION:
        return "channelization"
    if name in WORKERS:
        return "workers"
    if name in VEHICLES:
        return "vehicles"
    if name.startswith("Temporary Traffic Control Sign"):
        return "ttc_signs"
    if name in MESSAGE_BOARD:
        return "message_board"
    return None


def yolo_frame_score(counts, weights=WEIGHTS):
    score = float(weights["bias"])
    score += weights["channelization"] * safe_div(counts.get("channelization", 0), 5.0)
    score += weights["workers"] * safe_div(counts.get("workers", 0), 3.0)
    score += weights["vehicles"] * safe_div(counts.get("vehicles", 0), 2.0)
    score += weights["ttc_signs"] * safe_div(counts.get("ttc_signs", 0), 4.0)
    score += weights["message_board"] * safe_div(counts.get("message_board", 0), 1.0)
    total = sum(counts.get(k, 0) for k in
                ("channelization", "workers", "vehicles", "ttc_signs", "message_board"))
    return clamp01(score), total


def update_state(prev, score, state_dur, out_f, f_conf=F_CONF):
    """Verbatim port of jetson_app.py:644 update_state."""
    enter_th = f_conf["enter_th"]
    exit_th = f_conf["exit_th"]
    approach_th = f_conf["approach_th"]
    min_out = f_conf["min_out_frames"]
    MAX_APPROACH_DUR = 150

    if prev == "OUT":
        if score >= approach_th:
            return "APPROACHING", 0, 0
        return "OUT", 0, out_f + 1
    elif prev == "APPROACHING":
        if state_dur > MAX_APPROACH_DUR:
            return "OUT", 0, 0
        if score >= enter_th:
            return "INSIDE", 0, 0
        elif score <= (approach_th - 0.05):
            if out_f >= (min_out * 2):
                return "OUT", 0, 0
            return "APPROACHING", state_dur + 1, out_f + 1
        else:
            return "APPROACHING", state_dur + 1, 0
    elif prev == "INSIDE":
        if score < exit_th:
            return "EXITING", 0, 0
        return "INSIDE", state_dur + 1, 0
    elif prev == "EXITING":
        if score >= enter_th:
            return "INSIDE", state_dur, 0
        elif out_f >= min_out:
            return "OUT", 0, 0
        return "EXITING", state_dur, out_f + 1
    return prev, state_dur, out_f


def load_ground_truth(annotations, video_name, n_frames):
    gt = np.full(n_frames, "outside", dtype=object)
    for state_str, intervals in annotations[video_name].items():
        for start, end in intervals:
            gt[start:min(end + 1, n_frames)] = state_str
    return gt


def run_yolo_on_video(model, video_path, clip=None, verbose=False, dense=False):
    """clip: optional ClipFusion instance. None => YOLO-only (--disable-clip
    equivalent). When present, ports jetson_app.py's should_verify per-cue
    gating (every PER_CUE_INTERVAL frames, top-4 candidates by confidence),
    global CLIP fusion (every CLIP_INTERVAL frames once y_ema clears
    CLIP_TRIGGER_TH), and the orange-pixel context boost."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    predicted = np.full(n_frames, "outside", dtype=object)
    transitions = []
    state, y_ema_v, f_ema_v, state_dur, out_f = "OUT", None, None, 0, 999
    last_clip_score = 0.5
    cycle_idx = 0  # drives CLIP_INTERVAL / PER_CUE_INTERVAL like jetson_app.py's f_idx

    frame_idx = -1
    sim_clock = 0.0
    cycle_ms_acc = []

    while True:
        target_idx = int(sim_clock * fps)
        if target_idx >= n_frames:
            break
        while frame_idx < target_idx:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.retrieve()
        if not ok:
            break

        t0 = time.monotonic()
        res = model.predict(frame, conf=0.25, iou=0.7, imgsz=960, verbose=False)[0]

        h_img, w_img = frame.shape[:2]
        candidates = []
        for box, cid, conf in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(),
                                   res.boxes.conf.cpu().tolist()):
            cat = get_cue_category(model.names[int(cid)])
            if cat:
                x1, y1, x2, y2 = map(int, box)
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                candidates.append({"cat": cat, "conf": conf, "crop": frame[y1:y2, x1:x2]})

        counts = {}
        should_verify = clip is not None and (cycle_idx % PER_CUE_INTERVAL == 0)
        if should_verify and candidates:
            candidates.sort(key=lambda c: c["conf"], reverse=True)
            to_verify, remaining = candidates[:4], candidates[4:]
            scores = clip.verify_batch([c["crop"] for c in to_verify], [c["cat"] for c in to_verify])
            for c, s in zip(to_verify, scores):
                if s > PER_CUE_TH:
                    counts[c["cat"]] = counts.get(c["cat"], 0) + 1
            for c in remaining:
                counts[c["cat"]] = counts.get(c["cat"], 0) + 1
        else:
            for c in candidates:
                counts[c["cat"]] = counts.get(c["cat"], 0) + 1

        y_s, total_objs = yolo_frame_score(counts)
        evidence = clamp01(0.5 * clamp01(total_objs / 8.0) + 0.5 * clamp01(y_s))
        alpha = adaptive_alpha(evidence, F_CONF["ema_alpha"] * 0.4, F_CONF["ema_alpha"] * 1.2)
        y_ema_v = ema(y_ema_v, y_s, alpha)

        fused = y_s
        if clip is not None and y_ema_v >= CLIP_TRIGGER_TH:
            if cycle_idx % CLIP_INTERVAL == 0:
                last_clip_score = logistic(clip.global_score(frame) * 3.0)
            fused = (1.0 - CLIP_WEIGHT) * fused + CLIP_WEIGHT * last_clip_score
        if clip is not None and y_ema_v < CONTEXT_TRIGGER_BELOW:
            ctx = orange_context_score(frame)
            fused = (1.0 - ORANGE_WEIGHT) * fused + ORANGE_WEIGHT * ctx
        f_ema_v = ema(f_ema_v, clamp01(fused), alpha)

        prev_state = state
        state, state_dur, out_f = update_state(state, f_ema_v, state_dur, out_f)
        cycle_idx += 1

        elapsed_s = time.monotonic() - t0
        cycle_ms_acc.append(elapsed_s * 1000)

        mapped = STATE_MAP[state]
        if state != prev_state:
            transitions.append((frame_idx, STATE_MAP[prev_state], mapped))
        predicted[max(0, frame_idx):] = mapped

        if verbose:
            print(f"  frame={frame_idx:4d} t={sim_clock:6.1f}s state={mapped:11s} "
                  f"score={y_s:.2f} f_ema={f_ema_v:.2f} counts={counts} "
                  f"({elapsed_s * 1000:.0f}ms)", flush=True)
        # dense=True advances by one frame period instead of the measured
        # latency: the offline convention (every frame available), used only
        # to decompose how much of the published-vs-reproduced gap is protocol.
        sim_clock += (1.0 / fps) if dense else elapsed_s

    cap.release()
    return predicted, transitions, (sum(cycle_ms_acc) / len(cycle_ms_acc) if cycle_ms_acc else 0)


def evaluate(predicted, ground_truth):
    n = min(len(predicted), len(ground_truth))
    correct = sum(predicted[i] == ground_truth[i] for i in range(n))
    confusion = {}
    for i in range(n):
        key = (ground_truth[i], predicted[i])
        confusion[key] = confusion.get(key, 0) + 1
    return correct / n, confusion


def transition_lag(transitions, ground_truth):
    n = len(ground_truth)
    onsets = {}
    for i in range(1, n):
        if ground_truth[i] != ground_truth[i - 1]:
            onsets.setdefault(ground_truth[i], []).append(i)
    results = []
    for state, idxs in onsets.items():
        for onset in idxs:
            after = [f for f, _, to in transitions if to == state and f >= onset]
            results.append((state, onset, min(after) - onset if after else None))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default=os.path.expanduser("~/workzone/data/videos"))
    parser.add_argument("--annotations",
                         default=os.path.expanduser("~/workzone/data/workzone_annotations_full.json"))
    parser.add_argument("--weights",
                         default=os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280_960.engine"))
    parser.add_argument("--split-file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_split.json"))
    parser.add_argument("--split", choices=["calibration", "validation"])
    parser.add_argument("--videos", nargs="*")
    parser.add_argument("--clip", action="store_true",
                         help="Enable CLIP fusion + per-cue verification + context boost "
                              "(the full champion pipeline, matching `make workzone` defaults)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dense", action="store_true",
                         help="advance the replay clock by one frame period instead of the "
                              "measured latency (offline convention); used only to decompose "
                              "how much of the published-vs-reproduced gap is protocol")
    parser.add_argument("--cache-dir", default=None,
                         help="If set, save each video's per-frame predicted/gt state arrays "
                              "here (.npz) so paper-exact metrics can be recomputed later "
                              "without re-running inference")
    args = parser.parse_args()

    annotations = json.load(open(args.annotations))
    if args.videos:
        video_list = args.videos
    else:
        video_list = json.load(open(args.split_file))[args.split or "calibration"]

    print(f"Loading YOLO: {args.weights}")
    try:
        model = YOLO(args.weights, task="detect")
        # Warm-up + smoke test (engine may be built for another device).
        model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=960, verbose=False)
    except Exception as e:
        print(f"Engine failed ({e}); falling back to .pt")
        model = YOLO(os.path.expanduser("~/workzone/weights/yolo12s_hardneg_1280.pt"))

    clip = None
    if args.clip:
        print("Loading CLIP (open_clip ViT-B-32, openai) for global fusion + per-cue verification...")
        clip = ClipFusion()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    mode = "YOLO+CLIP (full champion)" if clip else "YOLO-only"
    print(f"Evaluating {len(video_list)} video(s) with the {mode} pipeline...")
    all_confusion = {}
    total_correct, total_frames = 0, 0
    all_lags = []
    cycle_means = []
    event_acc = EventLevelAccumulator(STATE_ORDER)
    timing_acc = TimingOffsetAccumulator(STATE_ORDER)
    transition_acc = TransitionToleranceAccumulator()

    for i, video_name in enumerate(video_list):
        video_path = os.path.join(args.videos_dir, video_name)
        if not os.path.exists(video_path):
            print(f"[{i + 1}/{len(video_list)}] SKIP: {video_name}")
            continue
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        gt = load_ground_truth(annotations, video_name, n_frames)

        t0 = time.monotonic()
        predicted, transitions, mean_cycle = run_yolo_on_video(model, video_path, clip=clip, verbose=args.verbose, dense=args.dense)
        acc, confusion = evaluate(predicted, gt)
        all_lags.extend(transition_lag(transitions, gt))
        for k, v in confusion.items():
            all_confusion[k] = all_confusion.get(k, 0) + v
        n = min(len(predicted), len(gt))

        pred_str = predicted[:n].tolist()
        gt_str = gt[:n].tolist()
        event_acc.add_video(pred_str, gt_str)
        timing_acc.add_video(pred_str, gt_str)
        transition_acc.add_video(pred_str, gt_str)
        if args.cache_dir:
            np.savez_compressed(os.path.join(args.cache_dir, f"{video_name}.npz"),
                                 predicted=np.array(pred_str), gt=np.array(gt_str))

        total_correct += acc * n
        total_frames += n
        cycle_means.append(mean_cycle)
        print(f"[{i + 1}/{len(video_list)}] {video_name}  acc={acc:.1%}  "
              f"transitions={len(transitions)}  cycle={mean_cycle:.0f}ms  "
              f"({time.monotonic() - t0:.1f}s)")

    print(f"\n=== RESUMO (YOLO champion, {mode}) ===")
    print(f"Acuracia por-frame agregada: {total_correct / total_frames:.1%}  "
          f"({total_frames} frames em {len(cycle_means)} videos)")
    print(f"Ciclo medio de inferencia: {sum(cycle_means) / len(cycle_means):.0f}ms")

    print("\n--- Matriz de confusao (linha=verdade, coluna=previsto) ---")
    header = "GT\\PRED".ljust(14) + "".join(s[:8].rjust(10) for s in STATE_ORDER)
    print(header)
    for gt_s in STATE_ORDER:
        row = gt_s.ljust(14)
        for pred_s in STATE_ORDER:
            row += str(all_confusion.get((gt_s, pred_s), 0)).rjust(10)
        print(row)

    print("\n--- Atraso de deteccao de transicao (frames a 30fps) ---")
    by_state = {}
    for state, onset, lag in all_lags:
        by_state.setdefault(state, []).append(lag)
    for state, lags in by_state.items():
        found = [l for l in lags if l is not None]
        missed = len(lags) - len(found)
        if found:
            print(f"{state:14s} media={sum(found) / len(found):.1f}f "
                  f"({sum(found) / len(found) / 30:.2f}s)  min={min(found)}  max={max(found)}  "
                  f"perdidas={missed}/{len(lags)}")
        else:
            print(f"{state:14s} TODAS perdidas ({missed}/{len(lags)})")

    print_paper_report(all_confusion, event_acc, timing_acc, transition_acc, STATE_ORDER)


if __name__ == "__main__":
    main()
