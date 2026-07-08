"""Extrai frames do video de controle negativo continuo (comma2k19, direcao
generica em rodovia, SEM zona de obra, confirmado visualmente) e monta o
batch pro llm_inference -- mesma cadencia/perguntas do prepare_trt_batch.py,
pra depois rodar o mesmo filtro Bayesiano do render_trt_video.py e ver a
trajetoria do estado EGO ao longo do video real.
"""
import os, json
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
FRAME_DIR = f"{BASE}/models/workzone-2b-stage6-engines/negcontrol_video_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

VIDEO = f"{BASE}/videos_negcontrol/comma2k19_highway_night.mp4"
CLIP_NAME = "comma2k19_highway_night"
STATE_EVERY_S = 0.6

QUESTIONS = {
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"EGO": 10, "TRAFFIC": 12, "ACTIVE": 10, "SIGN": 25, "DESC": 80}

requests = []
meta = []

cap = cv2.VideoCapture(VIDEO)
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
        fpath = f"{FRAME_DIR}/{CLIP_NAME}_{idx:05d}.jpg"
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
            meta.append({"city": CLIP_NAME, "sec": round(sec, 2), "frame_idx": idx, "question": name})
        n_saved += 1
    idx += 1
cap.release()
print(f"{CLIP_NAME}: {total} frames totais, {n_saved} amostrados a cada {STATE_EVERY_S}s")

batch = {
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 80,
    "requests": requests,
}

ENG = f"{BASE}/models/workzone-2b-stage6-engines"
with open(f"{ENG}/input_negcontrol_video_batch.json", "w") as f:
    json.dump(batch, f)
with open(f"{ENG}/input_negcontrol_video_batch_meta.json", "w") as f:
    json.dump(meta, f)

print(f"Total de requisicoes: {len(requests)}")
