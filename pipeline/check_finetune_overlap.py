#!/usr/bin/env python3
"""Reviewer question: do the 2B's fine-tuning images come from the benchmark's
drives or videos?  Run where the fine-tuning manifest lives (A100 side).

ROADWork names non-Pittsburgh images <city>_<sequence_id>_<video_id>_<frame_id>.jpg
and benchmark snippets <city>_<sequence_id>_<video_id>_<frame_id>_snippet.mp4,
so overlap can be checked at drive (sequence) level, video level, and as frame
distance to the snippet's start frame.

    python3 check_finetune_overlap.py --images manifest.txt [--split eval_split_full.json]

manifest.txt: one image path per line (or a JSON list / JSON-lines with an
"image" field).  Prints counts; writes nothing.
"""
import argparse
import json
import os
import re

PAT = re.compile(r"(boston|seattle|[a-z]+)_([0-9a-f]{32})_(\d{6})_(\d{5})")


def read_manifest(path):
    txt = open(path).read().strip()
    if txt.startswith("["):
        items = json.loads(txt)
    elif txt.startswith("{"):
        items = [json.loads(l) for l in txt.splitlines() if l.strip()]
    else:
        items = txt.splitlines()
    out = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("image") or it.get("images") or it.get("file_name") or ""
            if isinstance(it, list):
                out.extend(it)
                continue
        out.append(str(it))
    return out


def key(name):
    m = PAT.search(os.path.basename(name).lower())
    return (m.group(1), m.group(2), m.group(3), int(m.group(4))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_split_full.json"))
    args = ap.parse_args()
    split = json.load(open(args.split))
    bench = {s: [key(v) for v in vids] for s, vids in split.items()}
    imgs = [k for k in (key(p) for p in read_manifest(args.images)) if k]
    print(f"{len(imgs)} fine-tuning images with a ROADWork sequence name")
    for s, keys in bench.items():
        drives = {(c, q) for c, q, _, _ in keys}
        videos = {(c, q, v) for c, q, v, _ in keys}
        starts = {}
        for c, q, v, f in keys:
            starts.setdefault((c, q, v), []).append(f)
        same_drive = [k for k in imgs if (k[0], k[1]) in drives]
        same_video = [k for k in imgs if (k[0], k[1], k[2]) in videos]
        near = [k for k in same_video if any(abs(k[3] - f) <= 900 for f in starts[(k[0], k[1], k[2])])]
        print(f"{s:12s} drives {len(drives):3d} | images from those drives {len(same_drive):6d} | "
              f"from the same videos {len(same_video):6d} | within 30 s of a snippet start {len(near):6d}")


if __name__ == "__main__":
    main()
