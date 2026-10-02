#!/usr/bin/env python3
"""Live video + model-output demo for the Work Zone VLA engine on Jetson.

Keeps the TensorRT engines loaded in a single persistent `llm_stream_video`
subprocess (see TensorRT-Edge-LLM/examples/llm/llm_stream_video.cpp) and feeds
it the most recent video frame as fast as it can keep up, while a second
thread plays the video and overlays the latest caption. This is the same
protocol that will later drive a live camera instead of a video file.

Usage:
    python3 pipeline/live_video_demo.py --video ../demo/boston.mp4
    python3 pipeline/live_video_demo.py --video ../demo/boston.mp4 --full-desc
"""

import argparse
import json
import os
import subprocess
import threading
import time

import cv2

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

SHM_FRAME = "/dev/shm/live_demo_frame.jpg"
SHM_REQUEST = "/dev/shm/live_demo_request.json"


class InferenceWorker:
    """Owns the persistent llm_stream_video subprocess and the shared latest-frame state."""

    def __init__(self, binary, plugin_path, engine_dir, mm_engine_dir, prompt, max_gen_len,
                 temperature, top_p, top_k, infer_width=480):
        self.prompt = prompt
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Frames are downscaled to this width before inference: fewer vision
        # tokens = faster vision encoder AND faster prefill. 480px ≈ 135 image
        # tokens (engine minimum is 128) — measured on Orin it cuts gate
        # latency from ~145ms to ~80ms with identical outputs. <=0 disables.
        self.infer_width = infer_width

        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = plugin_path
        self.proc = subprocess.Popen(
            [binary, "--engineDir", engine_dir, "--multimodalEngineDir", mm_engine_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )

        # Block until the engine is fully loaded (READY on stderr) so the
        # first frame we send doesn't race a half-initialized runtime. Benign
        # warnings (e.g. missing optional action/audio engine) may print
        # before it, so scan lines rather than checking only the first one.
        for _ in range(200):
            line = self.proc.stderr.readline()
            if not line:
                raise RuntimeError("llm_stream_video exited before signaling READY")
            if "READY" in line:
                break
        else:
            raise RuntimeError("llm_stream_video did not signal READY in time")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_caption = "(waiting for first inference...)"
        self._latest_latency_ms = None
        self._frame_counter = 0
        self._stop = False
        self._worker_thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._worker_thread.start()

    def submit_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def latest_caption(self):
        with self._lock:
            return self._latest_caption, self._latest_latency_ms

    def stop(self):
        self._stop = True
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self._worker_thread.join(timeout=5)
        self.proc.terminate()

    def _write_request_json(self, image_path):
        request = {
            "batch_size": 1,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_generate_length": self.max_gen_len,
            "requests": [{
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.prompt},
                    ],
                }]
            }],
        }
        with open(SHM_REQUEST, "w") as f:
            json.dump(request, f)

    def _run(self):
        while not self._stop:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.monotonic()
            if self.infer_width > 0 and frame.shape[1] > self.infer_width:
                h = int(frame.shape[0] * self.infer_width / frame.shape[1])
                frame = cv2.resize(frame, (self.infer_width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(SHM_FRAME, frame)
            self._write_request_json(SHM_FRAME)

            try:
                self.proc.stdin.write(SHM_REQUEST + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                break

            response_line = None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("{"):
                    response_line = line
                    break
                # Any other line is stray library logging on stdout — ignore.

            if response_line is None:
                break

            wall_ms = (time.monotonic() - t0) * 1000.0
            try:
                data = json.loads(response_line)
            except json.JSONDecodeError:
                continue

            with self._lock:
                self._frame_counter += 1
                frame_idx = self._frame_counter
                if data.get("ok"):
                    self._latest_caption = data.get("output_text", "").strip() or "(empty)"
                else:
                    self._latest_caption = f"[error: {data.get('error', 'unknown')}]"
                self._latest_latency_ms = data.get("latency_ms", wall_ms)
                latency_ms = self._latest_latency_ms
                caption = self._latest_caption

            hz = 1000.0 / latency_ms if latency_ms else 0.0
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] #{frame_idx:05d} {latency_ms:6.0f}ms ({hz:4.1f}Hz)  {caption}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--multimodal-engine-dir", default=DEFAULT_MM_ENGINE_DIR)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--full-desc", action="store_true",
                         help="Use the full descriptive prompt instead of the fast Yes/No gate "
                              "(much lower Hz, richer text)")
    parser.add_argument("--prompt", default=None, help="Override the prompt text entirely")
    parser.add_argument("--max-generate-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--loop", action="store_true", help="Loop the video when it ends")
    parser.add_argument("--infer-width", type=int, default=480,
                         help="Downscale frames to this width before inference "
                              "(480 ≈ engine's minimum image tokens = fastest; 0 disables)")
    args = parser.parse_args()

    if args.prompt is not None:
        prompt = args.prompt
        max_gen_len = args.max_generate_length or 60
    elif args.full_desc:
        prompt = DESC_PROMPT
        max_gen_len = args.max_generate_length or 60
    else:
        prompt = GATE_PROMPT
        max_gen_len = args.max_generate_length or 3

    worker = InferenceWorker(
        binary=args.binary, plugin_path=args.plugin_path, engine_dir=args.engine_dir,
        mm_engine_dir=args.multimodal_engine_dir, prompt=prompt, max_gen_len=max_gen_len,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        infer_width=args.infer_width,
    )
    worker.start()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / native_fps

    window_name = "Work Zone VLA - live demo (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Playing {args.video} at {native_fps:.1f} fps native. Prompt: {prompt!r} "
          f"(max_generate_length={max_gen_len})")
    print("Press 'q' in the video window to quit.")

    last_frame_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            worker.submit_frame(frame)
            caption, latency_ms = worker.latest_caption()

            # Wrap the caption across a few lines so full descriptions are
            # readable on-screen, not just a single truncated line.
            wrap_width = 70
            caption_lines = [caption[i:i + wrap_width] for i in range(0, len(caption), wrap_width)][:3] or [""]

            banner_h = 40 + 28 * len(caption_lines)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], banner_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            hz_text = f"{1000.0 / latency_ms:.1f} Hz ({latency_ms:.0f} ms)" if latency_ms else "warming up..."
            cv2.putText(frame, hz_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, cline in enumerate(caption_lines):
                cv2.putText(frame, cline, (10, 55 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)

            # Pace playback to the video's native frame rate.
            now = time.monotonic()
            wait = frame_interval - (now - last_frame_time)
            if wait > 0:
                time.sleep(wait)
            last_frame_time = time.monotonic()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        worker.stop()


if __name__ == "__main__":
    main()
