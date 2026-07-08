"""Monta o batch da GRANDE VALIDACAO contra anotacao humana
(workzone_annotations_full.json): 27 snippets ROADWork limpos (nunca usados
em nenhum treino), 30s cada, anotados por humanos com ranges de frames por
estado (outside/approaching/inside/exiting).

Extrai frames a 0.6s (mesma cadencia do pipeline) e gera as 6 perguntas
(GATE + 5 do v16) por frame.
"""
import os, json
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage7-1-engines"
FRAME_DIR = f"{ENG}/annval_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

STATE_EVERY_S = 0.6
QUESTIONS = {
    "GATE": ("Are there any road work indicators in this scene "
             "(cones, barriers, temporary signs, workers, work vehicles)? "
             "Answer Yes or No."),
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"GATE": 16, "EGO": 32, "TRAFFIC": 16, "ACTIVE": 16, "SIGN": 25, "DESC": 80}

snippets = json.load(open(f"{BASE}/data/roadwork/annotation_val_clean_snippets.json"))
print(f"{len(snippets)} snippets limpos")

requests = []
meta = []
for snip in snippets:
    clip_id = snip.replace("_snippet.mp4", "")
    path = f"{BASE}/data/roadwork/videos/{snip}"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(STATE_EVERY_S * fps))
    idx = 0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            fpath = f"{FRAME_DIR}/{clip_id}_{idx:05d}.jpg"
            cv2.imwrite(fpath, frame)
            for qname, q in QUESTIONS.items():
                requests.append({
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": fpath},
                            {"type": "text", "text": q},
                        ],
                    }],
                    "max_generate_length": MAX_TOK[qname],
                })
                meta.append({"clip": snip, "frame_idx": idx, "fps": fps,
                             "question": qname})
            n += 1
        idx += 1
    cap.release()
    print(f"  {clip_id}: {n} amostras")

batch = {
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 80,
    "requests": requests,
}
with open(f"{ENG}/input_annval_batch.json", "w") as f:
    json.dump(batch, f)
with open(f"{ENG}/input_annval_batch_meta.json", "w") as f:
    json.dump(meta, f)
print(f"\ntotal: {len(requests)} requisicoes")
