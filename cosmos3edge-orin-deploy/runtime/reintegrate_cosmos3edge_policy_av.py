#!/usr/bin/env python3
"""Replace an invalid XYZ cumulative sum with backward-framewise SE(3)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cosmos3edge_av_pose import integrate_av_relative_poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload.get("mode") != "policy" or payload.get("domain") != "av":
        raise RuntimeError("expected an AV policy timeline")
    for update in payload["updates"]:
        poses = integrate_av_relative_poses(np.asarray(update["action_60x9"], dtype=np.float32))
        update.pop("cumulative_translation_61x3", None)
        update["absolute_pose_translation_61x3"] = poses[:, :3, 3].tolist()
        update["pose_integration"] = "SE(3) backward_framewise: T[i+1] = T[i] @ delta_T[i]"
    payload["schema"] = "cosmos3edge.policy_av_onnx_timeline_se3.v2"
    payload["meaning"] = "experimental base-model AV future rollout; not a post-trained or control-validated AV policy"
    payload["trajectory_representation"] = (
        "raw relative translation+rot6d integrated as backward-framewise SE(3); model-native units"
    )
    payload["supersedes_invalid_xyz_cumsum"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
