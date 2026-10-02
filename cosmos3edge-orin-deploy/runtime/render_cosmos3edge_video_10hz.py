#!/usr/bin/env python3
"""Render exact 10 Hz reasoner and AV trajectory updates beside the video."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


BG = (13, 17, 23)
PANEL = (24, 31, 41)
GRID = (55, 65, 77)
TEXT = (235, 240, 245)
MUTED = (150, 162, 175)
CYAN = (72, 220, 235)
GREEN = (110, 225, 140)
YELLOW = (70, 215, 255)
RED = (80, 95, 245)


def put(canvas, text, xy, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def wrap(text: str, width_px: int, scale: float = 0.48) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width_px:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_trajectory(canvas, rect, update, trajectory_mode):
    x0, y0, width, height = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), PANEL, -1)
    is_future = trajectory_mode == "policy"
    title = "AV ROLLOUT EXPERIMENTAL | FUTURO" if is_future else "AV INVERSE | NOVA INFERENCIA"
    put(canvas, title, (x0 + 22, y0 + 34), 0.66, CYAN, 2)
    if update is None:
        put(canvas, "INDISPONIVEL PARA ESTE TIMESTAMP", (x0 + 22, y0 + 84), 0.58, RED, 2)
        return
    points = np.asarray(update["absolute_pose_translation_61x3"], dtype=np.float64)
    xs, zs = points[:, 0], points[:, 2]
    px, py, pw, ph = x0 + 48, y0 + 76, width - 86, height - 156
    xpad = max(float(np.ptp(xs)) * 0.15, 0.2)
    zpad = max(float(np.ptp(zs)) * 0.08, 0.4)
    xmin, xmax = float(xs.min() - xpad), float(xs.max() + xpad)
    zmin, zmax = float(zs.min() - zpad), float(zs.max() + zpad)

    def project(x, z):
        return (
            px + int((x - xmin) / max(xmax - xmin, 1e-8) * pw),
            py + ph - int((z - zmin) / max(zmax - zmin, 1e-8) * ph),
        )

    for idx in range(6):
        xx = px + idx * pw // 5
        yy = py + idx * ph // 5
        cv2.line(canvas, (xx, py), (xx, py + ph), GRID, 1)
        cv2.line(canvas, (px, yy), (px + pw, yy), GRID, 1)
    polyline = np.asarray([project(x, z) for x, z in zip(xs, zs)], dtype=np.int32)
    cv2.polylines(canvas, [polyline], False, CYAN, 3, cv2.LINE_AA)
    cv2.circle(canvas, tuple(polyline[0]), 7, GREEN, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(polyline[-1]), 8, YELLOW, -1, cv2.LINE_AA)
    latency = float(update["latency_s"])
    deadline_met = bool(update["meets_realtime_deadline"])
    meaning = "SE(3) futuro experimental, 6.0 s" if is_future else "movimento observado, nao futuro"
    put(canvas, f"{meaning} | 60x9 | latencia {latency:.3f}s", (x0 + 22, y0 + height - 54), 0.45)
    put(
        canvas,
        "deadline 100 ms: PASS" if deadline_met else "deadline 100 ms: FAIL",
        (x0 + 22, y0 + height - 24),
        0.52,
        GREEN if deadline_met else RED,
        2,
    )


def draw_text(canvas, rect, update, average_latency):
    x0, y0, width, height = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), PANEL, -1)
    put(canvas, "REASONER INT4 + VISUAL FP16", (x0 + 22, y0 + 34), 0.66, YELLOW, 2)
    if update is None:
        put(canvas, "INDISPONIVEL PARA ESTE TIMESTAMP", (x0 + 22, y0 + 82), 0.58, RED, 2)
        return
    deadline_met = average_latency <= 0.1
    put(canvas, f"batch media/request: {average_latency:.3f}s", (x0 + 22, y0 + 61), 0.43, MUTED)
    put(
        canvas,
        "deadline 100 ms: PASS" if deadline_met else "deadline 100 ms: FAIL",
        (x0 + width - 300, y0 + 61),
        0.46,
        GREEN if deadline_met else RED,
        2,
    )
    lines = wrap(update["display_text"], width - 44, 0.48)
    max_lines = max(1, (height - 88) // 23)
    for line_idx, line in enumerate(lines[:max_lines]):
        put(canvas, line, (x0 + 22, y0 + 91 + line_idx * 23), 0.48, TEXT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reasoner", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    reasoner = json.loads(args.reasoner.read_text())
    trajectory = json.loads(args.trajectory.read_text())
    if reasoner["video_sha256"] != manifest["video_sha256"] or trajectory["video_sha256"] != manifest["video_sha256"]:
        raise RuntimeError("input provenance mismatch")
    reasoner_by_idx = {int(item["sample_idx"]): item for item in reasoner["updates"]}
    trajectory_by_idx = {int(item["sample_idx"]): item for item in trajectory["updates"]}
    hz = float(manifest["target_hz"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = 1920, 1080
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_idx / fps
        sample_idx = min(int(math.floor(timestamp_s * hz + 1e-9)), int(manifest["sample_count"]) - 1)
        canvas = np.full((out_h, out_w, 3), BG, dtype=np.uint8)
        put(canvas, "COSMOS3-EDGE | BOSTON | INFERENCIA TEMPORAL 10 Hz", (28, 47), 0.92, TEXT, 2)
        put(canvas, f"t={timestamp_s:05.2f}s  amostra={sample_idx:03d}  fonte={frame_idx:03d}", (1390, 44), 0.50, MUTED)
        video_w, video_h = 1110, 624
        resized = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)
        canvas[78 : 78 + video_h, 28 : 28 + video_w] = resized
        cv2.rectangle(canvas, (28, 78), (28 + video_w, 78 + video_h), GRID, 1)
        draw_trajectory(canvas, (1162, 78, 730, 624), trajectory_by_idx.get(sample_idx), trajectory.get("mode"))
        draw_text(
            canvas,
            (28, 728, 1864, 324),
            reasoner_by_idx.get(sample_idx),
            float(reasoner["average_wall_s_per_request"]),
        )
        assert encoder.stdin is not None
        encoder.stdin.write(canvas.tobytes())
        frame_idx += 1
    cap.release()
    assert encoder.stdin is not None
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    if frame_idx != source_frames:
        raise RuntimeError(f"rendered {frame_idx} of {source_frames} frames")

    metadata = {
        "schema": "cosmos3edge.overlay_10hz.v1",
        "source_video": str(args.video.resolve()),
        "video_sha256": manifest["video_sha256"],
        "output": str(args.output.resolve()),
        "source_frames": frame_idx,
        "source_fps": fps,
        "target_hz": hz,
        "reasoner_updates": len(reasoner_by_idx),
        "trajectory_updates": len(trajectory_by_idx),
        "trajectory_mode": trajectory.get("mode"),
        "trajectory_meaning": trajectory.get("meaning"),
        "no_interpolated_or_fabricated_inferences": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
