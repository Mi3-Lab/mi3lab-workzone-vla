"""Extrai frames dos 4 videos de teste e monta o JSON de batch pro llm_inference
(engine TensorRT INT4 real). Reaproveita as mesmas 5 perguntas/cadencia do v16.
"""
import os, sys, json
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
FRAME_DIR = f"{BASE}/models/workzone-2b-stage6-engines/video_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

VIDEOS = ["seattle_unseen", "boston", "denver", "chicago"]
STATE_EVERY_S = 0.6  # mesma cadencia do v16

QUESTIONS = {
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"EGO": 10, "TRAFFIC": 12, "ACTIVE": 10, "SIGN": 25, "DESC": 80}

requests = []
meta = []  # (city, sec, frame_idx, question_name) alinhado 1:1 com requests

for city in VIDEOS:
    cap = cv2.VideoCapture(f"{BASE}/videos/{city}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(STATE_EVERY_S * fps))

    idx = 0
    n_saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sec = idx / fps
            fpath = f"{FRAME_DIR}/{city}_{idx:05d}.jpg"
            cv2.imwrite(fpath, frame)
            for name, q in QUESTIONS.items():
                requests.append({
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": fpath},
                            {"type": "text", "text": q},
                        ],
                    }],
                    "max_generate_length": MAX_TOK[name],
                })
                meta.append({"city": city, "sec": round(sec, 2), "frame_idx": idx, "question": name})
            n_saved += 1
        idx += 1
    cap.release()
    print(f"{city}: {total} frames totais, {n_saved} amostrados a cada {STATE_EVERY_S}s")

batch = {
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 80,
    "requests": requests,
}

with open(f"{BASE}/models/workzone-2b-stage6-engines/input_video_batch.json", "w") as f:
    json.dump(batch, f)
with open(f"{BASE}/models/workzone-2b-stage6-engines/input_video_batch_meta.json", "w") as f:
    json.dump(meta, f)

print(f"\nTotal de requisicoes: {len(requests)}")
print(f"JSON salvo: input_video_batch.json ({os.path.getsize(f'{BASE}/models/workzone-2b-stage6-engines/input_video_batch.json')/1e6:.1f} MB)")
