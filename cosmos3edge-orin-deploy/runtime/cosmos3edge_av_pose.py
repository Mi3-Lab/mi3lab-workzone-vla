#!/usr/bin/env python3
"""Integrate Cosmos3 AV relative 9D actions with NVIDIA's SE(3) convention."""

from __future__ import annotations

import numpy as np


def _rot6d_to_so3(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float32)
    if value.shape != (6,):
        raise ValueError(f"rot6d must have shape (6,), got {value.shape}")
    col0, col1 = value[:3], value[3:]
    approximate = np.stack((col0, col1, np.cross(col0, col1)), axis=-1)
    u, _, vt = np.linalg.svd(approximate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation.astype(np.float32, copy=False)


def integrate_av_relative_poses(actions: np.ndarray) -> np.ndarray:
    """Convert [T,9] backward-framewise deltas into [T+1,4,4] poses."""
    relative = np.asarray(actions, dtype=np.float32)
    if relative.ndim != 2 or relative.shape[1] != 9:
        raise ValueError(f"AV actions must have shape (T,9), got {relative.shape}")
    if not np.isfinite(relative).all():
        raise ValueError("AV actions contain non-finite values")
    current = np.eye(4, dtype=np.float32)
    absolute = [current.copy()]
    for row in relative:
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = _rot6d_to_so3(row[3:9])
        delta[:3, 3] = row[:3]
        current = current @ delta
        absolute.append(current.copy())
    result = np.stack(absolute)
    if not np.isfinite(result).all():
        raise ValueError("integrated AV poses contain non-finite values")
    return result
