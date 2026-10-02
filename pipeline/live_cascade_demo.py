#!/usr/bin/env python3
"""Live cascade demo: video at native fps + evidence-first state machine.

The video window plays untouched at its native frame rate. A separate
inference thread drives the CascadeStateMachine from cascade_state_machine.py
over the persistent llm_stream_video process (engines stay loaded):

  - GATE  (Yes/No, ~90ms @480px)  -> every cycle, on the newest frame
  - SIGN  (reads sign text, ~200ms) -> every cycle in OUTSIDE, alongside
                                     GATE: a Yes/No rephrasing of "is there a
                                     sign" was tested and is NOT reliable (it
                                     missed a sign the full text-read caught
                                     consistently) so the full read is what
                                     runs every cycle, not a cheap gate
  - DESC  (rich text, ~350ms)     -> only when candidate evidence fires
                                     (GATE or SIGN) or, in active mode,
                                     refreshed periodically
  - EGO   (OUTSIDE/APPROACHING/INSIDE, ~160ms) -> only in active mode, feeds
                                     the Bayesian filter

Overlay shows: state (color-coded), per-channel latency, cycle Hz and the
latest description. The terminal logs every inference and state transition.

Usage:
    python3 pipeline/live_cascade_demo.py --video ../demo/boston.mp4 [--loop]
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from collections import deque

import cv2

from cascade_state_machine import CascadeStateMachine, WZState, has_corroboration, parse_gate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE_DIR = os.path.join(REPO, "engines", "llm")
DEFAULT_MM_ENGINE_DIR = os.path.join(REPO, "engines", "visual")
DEFAULT_PLUGIN = os.path.expanduser("~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so")
DEFAULT_BINARY = os.path.expanduser("~/TensorRT-Edge-LLM/build/examples/llm/llm_stream_video")

GATE_PROMPT = (
    "Are there any road work indicators in this scene (cones, barriers, "
    "temporary signs, workers, work vehicles)? Answer Yes or No."
)
DESC_PROMPT = "Describe the work zone elements visible in this scene."
EGO_PROMPT = (
    "Regarding the road work zone, is the ego vehicle OUTSIDE, APPROACHING, "
    "or INSIDE the work zone? Answer with one word."
)
# Both trained into the checkpoint (stage5_1_signs / ROADWork worker labels):
# SIGN reads temporary traffic signs ("ROAD WORK AHEAD", "DETOUR", ...) which
# appear BEFORE cones — earlier detection and a corroboration channel the
# state machine already understands. WORKERS reports presence + distance with
# a CRITICAL flag for close-by workers.
#
# SIGN must stay at this full-text phrasing and token budget (15) even in
# OUTSIDE state / --state-only mode: a cheap Yes/No rephrasing ("is there a
# sign?") was measured to miss signs the full read catches every time (e.g.
# boston.mp4 frame 30, "SHOULDER WORK" — the Yes/No version answered "No").
# Forcing the model to transcribe the sign text is what makes it reliable.
SIGN_PROMPT = "Read any temporary traffic control signs visible in this scene. What do they say?"
WORKERS_PROMPT = "Are there any workers visible in the work zone? Where are they?"

SHM_FRAME = "/dev/shm/cascade_frame.jpg"
SHM_REQUEST = "/dev/shm/cascade_request.json"

# The fine-tuned DESC answers embed structured fields — parse them instead of
# discarding: "Overall hazard severity: 2/5 (low). Recommended speed: ≤35 mph."
SEVERITY_RE = re.compile(r"severity:\s*(\d)\s*/\s*5", re.I)
SPEED_RE = re.compile(r"speed:\s*(?:stop or\s*)?[≤<=]*\s*(\d+)\s*mph", re.I)

STATE_COLORS = {  # BGR
    WZState.OUTSIDE: (96, 168, 48),      # green
    WZState.APPROACHING: (0, 200, 255),  # yellow
    WZState.INSIDE: (48, 48, 230),       # red
    WZState.EXITING: (0, 140, 255),      # orange
}


class PersistentEngine:
    """Synchronous request/response wrapper over the llm_stream_video process."""

    def __init__(self, binary, plugin_path, engine_dir, mm_engine_dir,
                 temperature=0.4, top_p=0.9, top_k=40):
        self.temperature = temperature
        # temperature=0 (greedy) makes the C++ runtime emit a WARNING on
        # EVERY request if top_p/top_k aren't already 1.0/1 ("numerical
        # instability"). WARNING-level lines go to stderr (see logger.h:
        # severity<=WARNING routes to stderr, INFO to stdout). Setting these
        # here avoids the warning at the source; the drain thread below is
        # the general-purpose fix for any other sustained stderr chatter.
        if temperature == 0:
            top_p, top_k = 1.0, 1
        self.top_p = top_p
        self.top_k = top_k
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = plugin_path
        self.proc = subprocess.Popen(
            [binary, "--engineDir", engine_dir, "--multimodalEngineDir", mm_engine_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        for _ in range(200):
            line = self.proc.stderr.readline()
            if not line:
                raise RuntimeError("llm_stream_video exited before signaling READY")
            if "READY" in line:
                break
        else:
            raise RuntimeError("llm_stream_video did not signal READY in time")

        # Drain stderr continuously for the life of the process. Without
        # this, any sustained WARNING/ERROR chatter (e.g. the temp=0 case
        # above, before the fix) fills the OS pipe buffer (~64KB) and the
        # child blocks forever on its next stderr write -- a silent full
        # deadlock with zero progress and zero error message. Discovered
        # after an 8-hour hang with 0/208 videos processed.
        self._stderr_tail = deque(maxlen=200)

        def _drain():
            for line in self.proc.stderr:
                self._stderr_tail.append(line.rstrip())
        threading.Thread(target=_drain, daemon=True).start()

    def infer(self, image_path, prompt, max_gen_len):
        """Returns (text, latency_ms) or (None, latency_ms) on failure."""
        request = {
            "batch_size": 1,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_generate_length": max_gen_len,
            "requests": [{
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }]
            }],
        }
        with open(SHM_REQUEST, "w") as f:
            json.dump(request, f)
        self.proc.stdin.write(SHM_REQUEST + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None, 0.0
            line = line.strip()
            if line.startswith("{"):
                break
        data = json.loads(line)
        if not data.get("ok"):
            return None, data.get("latency_ms", 0.0)
        return data.get("output_text", "").strip(), data.get("latency_ms", 0.0)

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.terminate()


class CascadeWorker:
    """Runs the cascade over the newest frame available; shares status with the UI."""

    def __init__(self, engine, infer_width=480, desc_refresh_s=1.5, desc_max_len=40,
                 state_only=False):
        self.engine = engine
        self.infer_width = infer_width
        self.desc_refresh_s = desc_refresh_s
        self.desc_max_len = desc_max_len
        # state_only: minimum tokens for a state decision — gate "Yes"/"No"
        # (2 tokens; the 1st generated token is the invisible answer_start),
        # EGO "INSIDE" (3 tokens), and DESC only fired for entry corroboration
        # at 12 tokens (objects are cited at the start of the answer). No
        # periodic DESC refresh: the output of the system is just the state.
        self.state_only = state_only
        self.gate_len = 2 if state_only else 3
        self.ego_len = 3 if state_only else 8
        if state_only:
            self.desc_max_len = 12
        self.csm = CascadeStateMachine()

        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop = False
        self._last_desc_time = 0.0

        # Shared status for the overlay.
        self.status = {
            "state": self.csm.state,
            "gate": None,
            "gate_ms": 0.0,
            "cycle_ms": 0.0,
            "desc": "",
            "desc_ms": 0.0,
            "ego": "",
            "sign": "",
            "workers": "",
            "workers_critical": False,
            "severity": None,
            "speed_mph": None,
            "cycles": 0,
        }
        self._aux_slot = 0  # rotates DESC -> SIGN -> WORKERS in active mode
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=8)

    def submit_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def snapshot(self):
        with self._lock:
            return dict(self.status)

    def _log(self, tag, ms, text):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{tag:4s}] {ms:6.0f}ms  {text}", flush=True)

    def _run(self):
        while not self._stop:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            t_cycle = time.monotonic()
            if self.infer_width > 0 and frame.shape[1] > self.infer_width:
                h = int(frame.shape[0] * self.infer_width / frame.shape[1])
                frame = cv2.resize(frame, (self.infer_width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(SHM_FRAME, frame)

            prev_state = self.csm.state

            # ── Layer 1: neutral binary gate, every cycle ────────────────
            gate_txt, gate_ms = self.engine.infer(SHM_FRAME, GATE_PROMPT, self.gate_len)
            if gate_txt is None:
                break
            gate = parse_gate(gate_txt)
            if not self.state_only:
                self._log("GATE", gate_ms, gate_txt)

            desc_txt = ""
            desc_ms = 0.0
            ego_txt = ""
            now = time.monotonic()

            sign_txt = ""
            sign_detected = False
            fast_entry = False
            candidate = bool(gate)  # overridden below in OUTSIDE with SIGN OR'd in
            if self.csm.state == WZState.OUTSIDE:
                # SIGN runs every cycle here, NOT only when gate fires. Signs
                # ("ROAD WORK AHEAD") often appear before cones are visible,
                # and the combined GATE question can miss an isolated distant
                # sign (measured false negative on boston.mp4 frame 30) — so
                # SIGN needs to be able to raise candidate evidence on its
                # own, not just confirm what GATE already saw.
                sign_txt, sign_ms = self.engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
                sign_txt = sign_txt or ""
                sign_detected = has_corroboration("", sign_txt)
                if not self.state_only:
                    self._log("SIGN", sign_ms, sign_txt[:100])

                candidate = bool(gate) or sign_detected
                # Candidate evidence -> corroborate with DESC too (catches the
                # cones/barriers-without-a-sign case that SIGN alone misses).
                if candidate:
                    desc_txt, desc_ms = self.engine.infer(SHM_FRAME, DESC_PROMPT, self.desc_max_len)
                    desc_txt = desc_txt or ""
                    self._last_desc_time = now
                    if not self.state_only:
                        self._log("DESC", desc_ms, desc_txt[:100])
                    # gate=Yes AND DESC citing a specific object on the SAME
                    # cycle is two channels agreeing synchronously — as
                    # trustworthy as the sign read, so it gets the same
                    # fast-path. Only a bare gate=Yes with no corroborating
                    # object (a thin/possible false-positive) still waits for
                    # the smoothed K_ENTER window.
                    fast_entry = sign_detected or (bool(gate) and has_corroboration(desc_txt, ""))
                else:
                    fast_entry = False
            else:
                # Active mode: EGO drives the Bayesian filter every cycle.
                ego_txt, ego_ms = self.engine.infer(SHM_FRAME, EGO_PROMPT, self.ego_len)
                ego_txt = ego_txt or ""
                if not self.state_only:
                    self._log("EGO", ego_ms, ego_txt)
                # Periodic auxiliary slot, rotating DESC -> SIGN -> WORKERS
                # (skipped in state_only mode: exit needs no corroboration).
                if not self.state_only and now - self._last_desc_time >= self.desc_refresh_s:
                    slot = self._aux_slot % 3
                    self._aux_slot += 1
                    self._last_desc_time = time.monotonic()
                    if slot == 0:
                        desc_txt, desc_ms = self.engine.infer(SHM_FRAME, DESC_PROMPT, self.desc_max_len)
                        desc_txt = desc_txt or ""
                        self._log("DESC", desc_ms, desc_txt[:100])
                    elif slot == 1:
                        sign_txt, sign_ms = self.engine.infer(SHM_FRAME, SIGN_PROMPT, 15)
                        sign_txt = sign_txt or ""
                        self._log("SIGN", sign_ms, sign_txt[:100])
                    else:
                        workers_txt, workers_ms = self.engine.infer(SHM_FRAME, WORKERS_PROMPT, 25)
                        workers_txt = workers_txt or ""
                        self._log("WORK", workers_ms, workers_txt[:100])
                        with self._lock:
                            self.status["workers"] = workers_txt
                            self.status["workers_critical"] = "CRITICAL" in workers_txt

            state = self.csm.update(candidate, ego_txt, desc_txt, sign_txt, fast_entry=fast_entry)
            cycle_ms = (time.monotonic() - t_cycle) * 1000.0

            if self.state_only:
                hz = 1000.0 / cycle_ms if cycle_ms else 0.0
                self._log("STATE", cycle_ms, f"{state.value.upper()}  ({hz:.1f}Hz)")
            if state != prev_state:
                print(f"\n########  STATE: {prev_state.value.upper()} -> {state.value.upper()}  ########\n",
                      flush=True)

            with self._lock:
                self.status["state"] = state
                self.status["gate"] = gate
                self.status["gate_ms"] = gate_ms
                self.status["cycle_ms"] = cycle_ms
                if desc_txt:
                    self.status["desc"] = desc_txt
                    self.status["desc_ms"] = desc_ms
                    m = SEVERITY_RE.search(desc_txt)
                    if m:
                        self.status["severity"] = int(m.group(1))
                    m = SPEED_RE.search(desc_txt)
                    if m:
                        self.status["speed_mph"] = int(m.group(1))
                if ego_txt:
                    self.status["ego"] = ego_txt
                if sign_txt and has_corroboration("", sign_txt):
                    # Only surface an actual sign reading, not the "no sign
                    # visible" filler text SIGN returns every OUTSIDE cycle.
                    self.status["sign"] = sign_txt
                if state == WZState.OUTSIDE and not candidate:
                    # Stale alerts must not survive leaving the zone — but keep
                    # them while a candidate is being corroborated (gate or
                    # sign fired), so an anticipatory "ROAD WORK AHEAD" stays
                    # visible even when it was SIGN (not GATE) that caught it.
                    self.status["workers"] = ""
                    self.status["workers_critical"] = False
                    self.status["severity"] = None
                    self.status["speed_mph"] = None
                    self.status["sign"] = ""
                self.status["cycles"] += 1


def draw_overlay(frame, st):
    h, w = frame.shape[:2]
    state = st["state"]
    color = STATE_COLORS[state]

    # Top banner: state + latencies.
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 64), (0, 0, 0), -1)
    frame = cv2.addWeighted(banner, 0.55, frame, 0.45, 0)
    cv2.rectangle(frame, (8, 10), (250, 54), color, -1)
    cv2.putText(frame, state.value.upper(), (18, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (255, 255, 255), 2)

    cycle_ms = st["cycle_ms"]
    hz = 1000.0 / cycle_ms if cycle_ms else 0.0
    gate_lbl = {True: "YES", False: "NO", None: "--"}[st["gate"]]
    info = f"gate {st['gate_ms']:.0f}ms [{gate_lbl}]  cycle {cycle_ms:.0f}ms ({hz:.1f}Hz)"
    if "dropped_total" in st:
        info += f"  dropped {st['dropped_total']}"
    cv2.putText(frame, info, (265, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    line2 = []
    if st["ego"]:
        line2.append(f"ego: {st['ego'][:24]}")
    if st.get("severity") is not None:
        line2.append(f"hazard {st['severity']}/5")
    if st.get("speed_mph") is not None:
        line2.append(f"speed <={st['speed_mph']}mph")
    if line2:
        cv2.putText(frame, "  ".join(line2), (265, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 255), 1)

    # Anticipatory sign reading + worker alert, right under the banner.
    y = 88
    if st.get("sign"):
        cv2.putText(frame, f"SIGN: {st['sign'][:70]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        y += 28
    if st.get("workers"):
        color = (0, 0, 255) if st.get("workers_critical") else (0, 215, 255)
        cv2.putText(frame, f"WORKERS: {st['workers'][:65]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Bottom: latest description, wrapped.
    desc = st["desc"]
    if desc:
        wrap = 78
        lines = [desc[i:i + wrap] for i in range(0, len(desc), wrap)][:3]
        bh = 16 + 24 * len(lines)
        banner = frame.copy()
        cv2.rectangle(banner, (0, h - bh), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(banner, 0.6, frame, 0.4, 0)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - bh + 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"desc {st['desc_ms']:.0f}ms", (w - 130, h - bh + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--multimodal-engine-dir", default=DEFAULT_MM_ENGINE_DIR)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--infer-width", type=int, default=480)
    parser.add_argument("--desc-refresh", type=float, default=1.5,
                         help="Seconds between DESC refreshes while in active mode")
    parser.add_argument("--desc-max-len", type=int, default=40)
    parser.add_argument("--state-only", action="store_true",
                         help="Output only the state (OUTSIDE/APPROACHING/INSIDE/EXITING): "
                              "minimum tokens per channel, no descriptions, lowest latency")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    engine = PersistentEngine(args.binary, args.plugin_path, args.engine_dir,
                              args.multimodal_engine_dir)
    worker = CascadeWorker(engine, infer_width=args.infer_width,
                           desc_refresh_s=args.desc_refresh, desc_max_len=args.desc_max_len,
                           state_only=args.state_only)
    worker.start()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / native_fps

    window = "Work Zone VLA - cascade live (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_FULLSCREEN)
    print(f"Playing {args.video} at {native_fps:.1f} fps (camera-sim: wall-clock advance, "
          f"late frames are DROPPED like a real camera). Cascade: GATE every cycle, "
          f"EGO in active mode{', state-only' if args.state_only else ''}.")

    # Camera simulation: the video position advances with the wall clock, no
    # matter how long display/inference take. If we fall behind, frames are
    # skipped (grab without decode) — exactly like a live camera, so outputs
    # can never accumulate playback delay.
    t0 = time.monotonic()
    frame_idx = -1
    dropped_total = 0
    try:
        while True:
            target_idx = int((time.monotonic() - t0) * native_fps)
            if target_idx <= frame_idx:
                # Ahead of schedule: sleep until the next frame is due.
                due = (frame_idx + 1) / native_fps - (time.monotonic() - t0)
                if due > 0:
                    time.sleep(due)
                target_idx = frame_idx + 1

            # Skip everything older than "now"; decode only the newest.
            prev_idx = frame_idx
            ended = False
            while frame_idx < target_idx:
                if not cap.grab():
                    ended = True
                    break
                frame_idx += 1
            if ended:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t0 = time.monotonic()
                    frame_idx = -1
                    continue
                break
            # All grabbed frames except the one we display were dropped.
            dropped_total += max(0, frame_idx - prev_idx - 1)

            ok, frame = cap.retrieve()
            if not ok:
                continue

            worker.submit_frame(frame)
            st = worker.snapshot()
            st["dropped_total"] = dropped_total
            frame = draw_overlay(frame, st)
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        worker.stop()
        engine.close()


if __name__ == "__main__":
    main()
