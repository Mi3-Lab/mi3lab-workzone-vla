#!/usr/bin/env python3
"""Merge independently computed inverse-dynamics timeline shards."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    shards = [json.loads(path.read_text()) for path in args.shard]
    provenance = (manifest["video_sha256"], float(manifest["target_hz"]))
    for path, shard in zip(args.shard, shards):
        if (shard["video_sha256"], float(shard["target_hz"])) != provenance:
            raise RuntimeError(f"provenance mismatch in {path}")
        if not shard.get("complete"):
            raise RuntimeError(f"incomplete shard: {path}")

    updates = sorted(
        [update for shard in shards for update in shard["updates"]],
        key=lambda item: int(item["sample_idx"]),
    )
    indexes = [int(update["sample_idx"]) for update in updates]
    expected = list(range(int(manifest["sample_count"])))
    if indexes != expected:
        raise RuntimeError(f"timeline is not exact/contiguous: got {indexes}, expected {expected}")
    latencies = [float(update["latency_s"]) for update in updates]
    payload = {key: value for key, value in shards[0].items() if key not in {"updates", "progress", "sample_range"}}
    payload.update(
        schema="cosmos3edge.inverse_onnx_timeline_merged.v1",
        manifest=str(args.manifest.resolve()),
        sample_range=[0, len(updates)],
        updates=updates,
        shards=[str(path.resolve()) for path in args.shard],
        progress={
            "completed": len(updates),
            "requested": len(updates),
            "last_sample_idx": indexes[-1],
            "latency_mean_s": statistics.mean(latencies),
            "latency_p50_s": float(np.percentile(latencies, 50)),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "latency_max_s": max(latencies),
            "realtime_deadlines_met": sum(bool(item["meets_realtime_deadline"]) for item in updates),
        },
        complete=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()

