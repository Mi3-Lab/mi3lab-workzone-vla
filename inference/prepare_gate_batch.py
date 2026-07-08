"""Monta o batch da pergunta de GATE (existencia de indicadores de obra,
pergunta neutra sem pressuposicao) para TODOS os frames ja usados nos
benchmarks -- positivos (4 videos de obra) e negativos (BDD100k + video
comma2k19). As respostas de estado (EGO/TRAFFIC/ACTIVE/SIGN/DESC) ja foram
computadas nos batches anteriores e serao reusadas pela maquina de estados
nova (eval_cascade_design.py); so o gate precisa de inferencia nova.
"""
import os, json, glob

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"

GATE_Q = ("Are there any road work indicators in this scene "
          "(cones, barriers, temporary signs, workers, work vehicles)? "
          "Answer Yes or No.")

requests = []
meta = []


def add(source, clip_id, frame_idx, img_path):
    requests.append({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": GATE_Q},
            ],
        }],
        "max_generate_length": 8,
    })
    meta.append({"source": source, "clip_id": clip_id, "frame_idx": frame_idx})


# 1. Frames dos 4 videos de obra (positivos) -- mesmos frames do batch v16
wz_meta = json.load(open(f"{ENG}/input_video_batch_meta.json"))
seen = set()
for m in wz_meta:
    key = (m["city"], m["frame_idx"])
    if key in seen:
        continue
    seen.add(key)
    add("workzone_videos", m["city"], m["frame_idx"],
        f"{ENG}/video_frames/{m['city']}_{m['frame_idx']:05d}.jpg")
n_wz = len(seen)

# 2. Frames do video de controle negativo continuo (comma2k19)
nc_meta = json.load(open(f"{ENG}/input_negcontrol_video_batch_meta.json"))
seen = set()
for m in nc_meta:
    if m["frame_idx"] in seen:
        continue
    seen.add(m["frame_idx"])
    add("comma2k19", "comma2k19_highway_night", m["frame_idx"],
        f"{ENG}/negcontrol_video_frames/comma2k19_highway_night_{m['frame_idx']:05d}.jpg")
n_nc = len(seen)

# 3. Imagens BDD100k (controle negativo por-frame)
bdd = sorted(glob.glob(f"{BASE}/data/roadwork/bdd100k_unseen/images/*.jpg"))
for p in bdd:
    add("bdd100k", os.path.basename(p).replace(".jpg", ""), 0, p)

print(f"workzone_videos: {n_wz} frames | comma2k19: {n_nc} | bdd100k: {len(bdd)}")
print(f"total: {len(requests)} requisicoes")

batch = {
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 8,
    "requests": requests,
}
with open(f"{ENG}/input_gate_batch.json", "w") as f:
    json.dump(batch, f)
with open(f"{ENG}/input_gate_batch_meta.json", "w") as f:
    json.dump(meta, f)
print("salvo: input_gate_batch.json")
