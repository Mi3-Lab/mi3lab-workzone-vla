#!/usr/bin/env python3
"""Fail-closed validation of the final Boston 10 Hz evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def exact_timeline(name: str, data: dict, expected: int, video_sha: str) -> None:
    updates = data["updates"]
    if len(updates) != expected:
        raise RuntimeError(f"{name}: {len(updates)} != {expected}")
    if [int(item["sample_idx"]) for item in updates] != list(range(expected)):
        raise RuntimeError(f"{name}: indexes are not contiguous")
    if data["video_sha256"] != video_sha:
        raise RuntimeError(f"{name}: source video hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--reasoner", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--native-bf16", type=Path, required=True)
    parser.add_argument("--native-int4", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    timeline = load(args.timeline)
    expected = int(timeline["sample_count"])
    if expected != 151 or float(timeline["target_hz"]) != 10.0:
        raise RuntimeError("acceptance requires exactly 151 samples at 10 Hz")
    samples = timeline["samples"]
    if [int(item["sample_idx"]) for item in samples] != list(range(expected)):
        raise RuntimeError("manifest indexes are not contiguous")
    timestamps = np.asarray([item["timestamp_s"] for item in samples], dtype=np.float64)
    if not np.allclose(timestamps, np.arange(expected) / 10.0, atol=1e-9, rtol=0):
        raise RuntimeError("manifest timestamps are not exact 100 ms ticks")

    video_sha = timeline["video_sha256"]
    reasoner, trajectory, bf16, int4 = map(
        load, (args.reasoner, args.trajectory, args.native_bf16, args.native_int4)
    )
    for name, data in (
        ("reasoner", reasoner),
        ("trajectory", trajectory),
        ("native_bf16", bf16),
        ("native_int4", int4),
    ):
        exact_timeline(name, data, expected, video_sha)
    if trajectory.get("mode") != "policy":
        raise RuntimeError("trajectory evidence is not an AV policy-mode rollout")
    expected_meaning = "experimental base-model AV future rollout; not a post-trained or control-validated AV policy"
    if trajectory.get("meaning") != expected_meaning:
        raise RuntimeError("trajectory evidence is missing the experimental/non-control qualification")
    if int(trajectory.get("future_steps", 0)) != 60:
        raise RuntimeError("AV policy must predict exactly 60 future transitions")
    if float(trajectory.get("future_horizon_s", 0.0)) != 6.0:
        raise RuntimeError("AV policy horizon must be exactly 6 seconds")
    if bf16.get("precision") != "bf16":
        raise RuntimeError("native BF16 evidence has the wrong precision label")
    if int4.get("precision") != "int4":
        raise RuntimeError("native INT4 evidence has the wrong precision label")
    int4_load = int4.get("load", {})
    int4_state_count = int(int4_load.get("torchao_int4_state_values", 0))
    int4_loaded_count = int(int4_load.get("torchao_int4_loaded_values", 0))
    if int4_state_count <= 0:
        raise RuntimeError("native INT4 evidence did not verify quantized state tensors")
    if int4_loaded_count <= 0:
        raise RuntimeError("native INT4 evidence did not verify quantized tensors in the loaded model")
    if int4_loaded_count != int4_state_count:
        raise RuntimeError("native INT4 tensor count changed while loading the model")
    for name, data in (("reasoner", reasoner), ("native_bf16", bf16), ("native_int4", int4)):
        if any(not item["exact_text"].strip() for item in data["updates"]):
            raise RuntimeError(f"{name} contains empty text")
    for item in trajectory["updates"]:
        action = np.asarray(item["action_60x9"], dtype=np.float64)
        if action.shape != (60, 9) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid trajectory sample {item['sample_idx']}")
        absolute = np.asarray(item["absolute_pose_translation_61x3"], dtype=np.float64)
        if absolute.shape != (61, 3) or not np.isfinite(absolute).all():
            raise RuntimeError(f"invalid SE(3)-integrated trajectory sample {item['sample_idx']}")
        if not item.get("pose_integration", "").startswith("SE(3) backward_framewise"):
            raise RuntimeError(f"wrong pose convention at sample {item['sample_idx']}")
        if abs(float(item["prediction_end_s"]) - float(item["timestamp_s"]) - 6.0) > 1e-9:
            raise RuntimeError(f"invalid prediction horizon at sample {item['sample_idx']}")

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
                "-of", "json", str(args.video),
            ],
            text=True,
        )
    )["streams"][0]
    if int(probe["width"]) != 1920 or int(probe["height"]) != 1080 or int(probe["nb_frames"]) != 451:
        raise RuntimeError(f"invalid rendered video: {probe}")

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (
            args.timeline, args.reasoner, args.trajectory,
            args.native_bf16, args.native_int4, args.video,
        )
    }
    result = {
        "schema": "cosmos3edge.complete_delivery_acceptance.v1",
        "status": "PASS",
        "samples": expected,
        "target_hz": 10.0,
        "timestamps_exact": True,
        "reasoner_texts_nonempty": True,
        "trajectory_shapes_and_finiteness": "151 x [60,9] PASS",
        "trajectory_mode": "experimental base-model AV rollout",
        "control_validated": False,
        "future_horizon": "60 transitions / 6.0 seconds per 100 ms tick",
        "video": probe,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
