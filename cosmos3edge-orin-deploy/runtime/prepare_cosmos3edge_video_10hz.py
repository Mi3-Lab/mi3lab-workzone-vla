#!/usr/bin/env python3
"""Extract a deterministic video timeline and build Edge-LLM requests.

One request is created for every target timestamp.  The manifest is the source
of truth shared by the reasoner, inverse-dynamics runner and renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


DEFAULT_PROMPT = (
    "Analyze this driving frame. Return exactly two concise lines: "
    "SCENE: describe the road-work evidence and traffic state. "
    "ACTION: state the safest immediate ego-vehicle path. Do not speculate."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--native-input-dir", type=Path)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-generate-length", type=int, default=96)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.hz <= 0:
        raise SystemExit("--hz must be positive")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: fps={source_fps}, frames={frame_count}")

    duration_s = frame_count / source_fps
    # Include every tick whose timestamp is covered by an actual source frame.
    sample_count = int(math.floor(((frame_count - 1) / source_fps) * args.hz + 1e-9)) + 1
    source_indexes = [min(frame_count - 1, int(round(i * source_fps / args.hz))) for i in range(sample_count)]
    if len(set(source_indexes)) != sample_count:
        raise RuntimeError("target cadence is higher than the source cadence")

    args.frame_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.requests.parent.mkdir(parents=True, exist_ok=True)
    wanted = {source_idx: sample_idx for sample_idx, source_idx in enumerate(source_indexes)}
    samples: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sample_idx = wanted.get(frame_idx)
        if sample_idx is not None:
            timestamp_s = sample_idx / args.hz
            image_path = (args.frame_dir / f"sample_{sample_idx:04d}_t{round(timestamp_s * 1000):06d}ms.jpg").resolve()
            ok_write = cv2.imwrite(
                str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]
            )
            if not ok_write:
                raise RuntimeError(f"failed to write {image_path}")
            samples.append(
                {
                    "sample_idx": sample_idx,
                    "timestamp_s": timestamp_s,
                    "source_frame_idx": frame_idx,
                    "image": str(image_path),
                }
            )
        frame_idx += 1
    cap.release()
    samples.sort(key=lambda item: item["sample_idx"])
    if len(samples) != sample_count:
        raise RuntimeError(f"extracted {len(samples)} of {sample_count} scheduled samples")

    manifest = {
        "schema": "cosmos3edge.video_timeline.v1",
        "video": str(args.video.resolve()),
        "video_sha256": sha256(args.video),
        "source_fps": source_fps,
        "source_frames": frame_count,
        "source_width": width,
        "source_height": height,
        "duration_s": duration_s,
        "target_hz": args.hz,
        "sample_count": sample_count,
        "prompt": args.prompt,
        "samples": samples,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    requests = {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": args.max_generate_length,
        "requests": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": sample["image"]},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ]
            }
            for sample in samples
        ],
    }
    args.requests.write_text(json.dumps(requests, ensure_ascii=False, indent=2) + "\n")
    if args.native_input_dir is not None:
        args.native_input_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            native_input = {
                "model_mode": "reasoner",
                "name": f"t{sample['sample_idx']:04d}",
                "prompt": args.prompt,
                "vision_path": sample["image"],
                "max_new_tokens": args.max_generate_length,
                "do_sample": False,
                "seed": 0,
            }
            path = args.native_input_dir / f"t{sample['sample_idx']:04d}.json"
            path.write_text(json.dumps(native_input, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in manifest if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
