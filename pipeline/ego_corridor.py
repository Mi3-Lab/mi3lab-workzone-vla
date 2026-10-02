#!/usr/bin/env python3
"""Geometric ego-relevance: does a detected work-zone object lie in the corridor
the ego vehicle is about to drive through?

Motivation (measured, see COSMOS3_REFORMULATION.md): asking the VLM directly
("are the cones on the ego lane?") does NOT discriminate — a physical-AI world
model that perceives the scene correctly still answers almost independently of
ground truth.  Ego-relevance needs geometry, not language.

Model: a forward-facing dashcam.  The ego path projects to a trapezoidal
corridor that narrows toward the vanishing row and widens toward the bottom of
the frame.  An object is ego-relevant iff its box overlaps that corridor.

Optionally the corridor is shifted/steered by the ego trajectory (lateral
displacement over the horizon), so a lane change bends the corridor toward the
objects it will actually pass.
"""
from dataclasses import dataclass


@dataclass
class Corridor:
    """Trapezoid in normalised image coords (0..1), x to the right, y down."""
    y_horizon: float = 0.52     # rows above this are too far to matter
    half_w_bottom: float = 0.30  # half-width of the lane at the image bottom
    half_w_horizon: float = 0.045
    x_bottom: float = 0.50      # where the lane sits at the bottom (ego centre)
    x_horizon: float = 0.50     # where it converges (shifted by steering)

    def half_width_at(self, y: float) -> float:
        if y <= self.y_horizon:
            return self.half_w_horizon
        t = (y - self.y_horizon) / max(1e-6, 1.0 - self.y_horizon)
        return self.half_w_horizon + t * (self.half_w_bottom - self.half_w_horizon)

    def centre_at(self, y: float) -> float:
        if y <= self.y_horizon:
            return self.x_horizon
        t = (y - self.y_horizon) / max(1e-6, 1.0 - self.y_horizon)
        return self.x_horizon + t * (self.x_bottom - self.x_horizon)

    def contains_box(self, box, img_w, img_h, min_overlap=0.15, coord_scale=None):
        """box = [x1,y1,x2,y2].  Returns (is_relevant, overlap_frac).

        The Cosmos3-Edge reasoner emits coordinates normalised to 0..1000
        (Qwen-VL convention), not pixels.  When `coord_scale` is None we detect
        it: any coordinate beyond the frame implies the 0..1000 convention.
        """
        x1, y1, x2, y2 = [float(v) for v in box]
        if coord_scale is None:
            coord_scale = 1000.0 if max(x1, y1, x2, y2) > max(img_w, img_h) else None
        if coord_scale:
            sx = sy = coord_scale
        else:
            sx, sy = img_w, img_h
        x1, x2 = min(x1, x2) / sx, max(x1, x2) / sx
        y1, y2 = min(y1, y2) / sy, max(y1, y2) / sy
        # sample the box vertically; measure horizontal overlap with corridor
        rows, inside = 7, 0
        for i in range(rows):
            y = y1 + (y2 - y1) * (i / max(1, rows - 1))
            if y < self.y_horizon:
                continue
            c, hw = self.centre_at(y), self.half_width_at(y)
            lo, hi = c - hw, c + hw
            ov = max(0.0, min(x2, hi) - max(x1, lo))
            if ov > 0 and (x2 - x1) > 0 and ov / (x2 - x1) >= min_overlap:
                inside += 1
        frac = inside / rows
        return frac > 0, frac

    def steered(self, lateral_shift_norm: float):
        """Bend the corridor by the ego's lateral motion over the horizon.
        lateral_shift_norm > 0 => vehicle drifts right."""
        c = Corridor(self.y_horizon, self.half_w_bottom, self.half_w_horizon,
                     self.x_bottom, self.x_horizon)
        c.x_horizon = min(0.95, max(0.05, self.x_horizon + lateral_shift_norm))
        return c


def lateral_shift_from_poses(poses, forward_axis=2, lateral_axis=0,
                             max_norm=0.35):
    """From integrated SE(3) poses (N,4,4), estimate how far the ego drifts
    sideways relative to how far it travels forward, normalised for use as a
    corridor shift.  Returns a value in [-max_norm, max_norm]."""
    import numpy as np
    P = np.asarray(poses)
    if P.ndim != 3 or len(P) < 2:
        return 0.0
    t0, t1 = P[0][:3, 3], P[-1][:3, 3]
    d = t1 - t0
    fwd = abs(float(d[forward_axis]))
    lat = float(d[lateral_axis])
    if fwd < 1e-3:
        return 0.0
    ratio = lat / fwd
    return max(-max_norm, min(max_norm, ratio))


def parse_boxes(text):
    """Pull [{'bbox_2d':[...], 'label':...}] out of the reasoner's JSON-ish
    reply, tolerating truncation (the reply is capped by the token budget)."""
    import json
    import re
    out = []
    for m in re.finditer(r'\{\s*"bbox_2d"\s*:\s*\[([^\]]*)\]\s*,\s*"label"\s*:\s*"([^"]*)"', text):
        try:
            nums = [float(v) for v in m.group(1).split(",")]
            if len(nums) == 4:
                out.append({"bbox": nums, "label": m.group(2)})
        except ValueError:
            continue
    if not out:  # last resort: any 4-number array
        for m in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text):
            out.append({"bbox": [float(g) for g in m.groups()], "label": "?"})
    return out
