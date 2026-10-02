#!/usr/bin/env python3
"""Convert draft_labels.DRAFT to the benchmark format (frame intervals per state,
as in workzone_annotations_full.json), plus the visibility label and notes.

Usage:  python3 build_draft.py   ->  california_draft.json
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.expanduser("~/jetson-deploy/pipeline"))
import eval_external_cosmos3 as E  # noqa: E402
from draft_labels import DRAFT  # noqa: E402

out = {"_meta": {"annotator": "claude-draft", "verified": False,
                 "guideline": "annotation/GUIDELINE.md", "grid_s": 2}}
for key, d in DRAFT.items():
    cap = cv2.VideoCapture(E.find_video(key))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    to_f = lambda s: min(int(round(s * fps)), n - 1)  # noqa: E731
    rec = {s: [] for s in ("outside", "approaching", "inside", "exiting")}
    for st, a, b in d["states"]:
        rec[st].append([to_f(a), to_f(b) - 1 if to_f(b) < n - 1 else n - 1])
    rec["visible"] = [[to_f(a), to_f(b) - 1] for a, b in d["visible"]]
    rec["notes"] = [[to_f(a), to_f(b) - 1, txt] for a, b, txt in d["notes"]]
    out[key] = rec
json.dump(out, open("california_draft.json", "w"), indent=1)
print("wrote california_draft.json:", [k for k in out if not k.startswith("_")])
