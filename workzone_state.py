"""Work zone temporal state machine with hysteresis.

Two signals feed the state machine:
  1. keyword_score  — keyword rules on the VLM description (fast, every frame)
  2. model_state    — direct model question every MODEL_QUERY_INTERVAL frames (slower, more accurate)

State transitions require sustained evidence, not a single frame.
"""

from collections import deque
from enum import Enum

import numpy as np


class WZState(Enum):
    OUTSIDE     = "outside"
    APPROACHING = "approaching"
    INSIDE      = "inside"
    EXITING     = "exiting"


# ── Keyword classifiers ────────────────────────────────────────────────────────

# Strong INSIDE: heavy equipment and workers — only visible when vehicle is inside the zone
_INSIDE_STRONG = [
    "work vehicle", "construction vehicle", "excavator", "bulldozer",
    "worker", "workers", "crew",
    "jersey barrier", "drum", "arrow board",
    "barricade on right", "barricade on left",
    "barrier on right", "barrier on left",
    "fenced-off", "surrounded", "surrounding",
    "narrowed", "lane closure", "partially blocking",
]

# APPROACHING: warning signs and light markers seen from a distance
_APPROACH_STRONG = [
    "road work ahead", "work ahead", "construction ahead",
    "slow down", "reduce speed", "warning", "caution",
    "traffic cone", "orange cone", "tubular marker",
    "temporary traffic control", "traffic control sign",
    "distant", "far ahead", "visible in the distance",
]

_EXITING_KW = [
    "end of", "end work", "end of the work zone",
    "resuming", "resume speed",
    "clearing", "clear road", "road clear", "open road",
    "past the", "beyond the work",
]

_NEGATIVE_KW = [
    "no work zone", "no construction", "no cones", "no workers",
    "typical urban", "road is clear", "clear of any", "no visible",
    "no objects", "no relevant", "not a work zone",
]

# Keep these as module-level aliases for parse_model_state compatibility
NEGATIVE_KW = _NEGATIVE_KW
EXITING_KW  = _EXITING_KW


def keyword_score(text: str) -> dict:
    """Contextual per-state scores — not pure keyword counting.

    Rules (in priority order):
    1. Negative keywords → OUTSIDE
    2. Heavy equipment / workers → INSIDE (they only appear when vehicle is inside)
    3. Elements on both left AND right side of road → INSIDE (bilateral = inside)
    4. Light markers (cones, tubular) on single side, no heavy elements → APPROACHING
    5. Only exiting keywords → EXITING
    """
    t = text.lower()

    if any(k in t for k in _NEGATIVE_KW):
        return {s: 0.0 for s in WZState}

    inside_score = sum(1 for k in _INSIDE_STRONG if k in t)

    # Bilateral detection: elements explicitly on both sides = vehicle is inside
    bilateral = ("left side of road" in t and "right side of road" in t)
    if bilateral:
        inside_score += 2

    approach_score = sum(1 for k in _APPROACH_STRONG if k in t)

    # Single-side light markers WITHOUT any heavy elements → approaching or exiting
    single_side = ("left side of road" in t) != ("right side of road" in t)  # XOR
    if single_side and inside_score == 0:
        approach_score += 1

    exiting_score = sum(1 for k in _EXITING_KW if k in t)

    # OUTSIDE gets a positive signal when only light markers are visible (no heavy equipment).
    # This allows the EMA to build toward OUTSIDE from INSIDE state as the vehicle exits,
    # since INSIDE→APPROACHING is not a valid transition.
    outside_score = (approach_score * 0.5) if inside_score == 0 else 0.0

    total = inside_score + approach_score + exiting_score + outside_score + 1e-9
    return {
        WZState.INSIDE:      inside_score   / total,
        WZState.APPROACHING: approach_score / total,
        WZState.EXITING:     exiting_score  / total,
        WZState.OUTSIDE:     outside_score  / total,
    }


def parse_model_state(answer: str):
    """Parse direct model answer into a WZState.

    The model uses its own vocabulary: ACTIVE=INSIDE, TTC=APPROACHING, PASSIVE=OUTSIDE.
    We also handle the user-requested labels as fallback.
    """
    a = answer.lower().strip()
    # INSIDE: explicit label or model's "active" (actively inside work zone)
    if "inside" in a or "active" in a:
        return WZState.INSIDE
    # APPROACHING: explicit label or model's "ttc" (Temporary Traffic Control setup visible)
    if "approach" in a or "ttc" in a:
        return WZState.APPROACHING
    # EXITING: explicit label
    if "exit" in a:
        return WZState.EXITING
    # OUTSIDE: explicit label, negation, or model's "passive" (normal/passive driving)
    if "outside" in a or "no work" in a or "passive" in a or "normal" in a:
        return WZState.OUTSIDE
    return None


# ── State machine ──────────────────────────────────────────────────────────────

# Minimum duration (seconds) of sustained signal to transition INTO each state
ENTER_THRESHOLD_S = {
    WZState.OUTSIDE:     0.5,   # reduced: 0.5s of no-wz evidence enough to exit
    WZState.APPROACHING: 0.17,  # ~5 frames at 30fps — fast entry, critical for driver warning
    WZState.INSIDE:      0.33,  # ~10 frames at 30fps
    WZState.EXITING:     0.5,   # ~15 frames at 30fps
}

# Valid state transitions
VALID_TRANSITIONS = {
    WZState.OUTSIDE:     {WZState.APPROACHING, WZState.INSIDE},
    WZState.APPROACHING: {WZState.INSIDE, WZState.OUTSIDE},
    WZState.INSIDE:      {WZState.EXITING, WZState.OUTSIDE},  # allow direct INSIDE→OUTSIDE
    WZState.EXITING:     {WZState.OUTSIDE, WZState.INSIDE},
}

# EMA alpha for smoothing per-state scores — higher = more responsive to current frame
EMA_ALPHA = 0.4


class WorkZoneStateMachine:
    def __init__(self, infer_fps: float = 30.0):
        self.state = WZState.OUTSIDE
        self.ema_scores = {s: 0.0 for s in WZState}
        self.ema_scores[WZState.OUTSIDE] = 1.0
        self.candidate = WZState.OUTSIDE
        self.candidate_count = 0
        self.frames_in_state = 0
        # Convert second-based thresholds to inference-count thresholds
        self.enter_threshold = {
            s: max(1, round(t * infer_fps))
            for s, t in ENTER_THRESHOLD_S.items()
        }

    def update(self, kw_scores: dict) -> WZState:
        # 1. EMA smooth keyword scores
        for s in WZState:
            self.ema_scores[s] = (
                EMA_ALPHA * kw_scores.get(s, 0.0)
                + (1 - EMA_ALPHA) * self.ema_scores[s]
            )

        # 2. Mild stickiness — avoids single-frame flicker but doesn't trap state
        smoothed = dict(self.ema_scores)
        smoothed[self.state] = max(smoothed[self.state], 0.20)

        # 3. Pick best candidate among VALID transitions only
        allowed = VALID_TRANSITIONS[self.state] | {self.state}
        best = max(allowed, key=lambda s: smoothed[s])

        if best == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = best
            self.candidate_count = 1

        # 4. Transition if candidate sustained long enough
        if (self.candidate != self.state
                and self.candidate_count >= self.enter_threshold[self.candidate]):
            self.state = self.candidate
            self.frames_in_state = 0
            self.candidate_count = 0

        self.frames_in_state += 1
        return self.state

    @property
    def confidence(self) -> float:
        return self.ema_scores[self.state]
