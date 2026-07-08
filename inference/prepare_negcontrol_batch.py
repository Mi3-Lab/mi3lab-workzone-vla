"""Monta o batch de 'controle negativo' pro engine TensorRT INT4 (fixed):
imagens/frames SEM zona de obra, nunca vistos em nenhum estagio de treino,
pra medir taxa de falso-positivo (overfitting em 'sempre ha uma obra').

Duas fontes, ambas confirmadas fora de qualquer split de treino:
  1. BDD100k (data/roadwork/bdd100k_unseen/images/) -- 249 imagens de direcao
     generica, dataset sem nenhuma curadoria de zona de obra. Controle limpo.
  2. ROADWork snippets nunca referenciados em lingoqa_general (100 clipes de
     11 cidades nunca usadas no stage6) -- mesmo estilo de camera/dominio dos
     testes atuais, mas sem garantia de ground-truth "sem obra": tratado como
     checagem exploratoria/qualitativa, nao like-for-like com o BDD100k.
"""
import os, json, random
import cv2

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
ENG = f"{BASE}/models/workzone-2b-stage6-engines"
NEG = f"{BASE}/data/roadwork/bdd100k_unseen"
FRAME_DIR = f"{ENG}/negcontrol_frames"
os.makedirs(FRAME_DIR, exist_ok=True)
random.seed(13)

QUESTIONS = {
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"EGO": 10, "TRAFFIC": 12, "ACTIVE": 10, "SIGN": 25, "DESC": 80}

requests = []
meta = []  # (source, clip_id, frame_idx, question) alinhado 1:1 com requests


def add_requests(source, clip_id, frame_idx, img_path):
    for name, q in QUESTIONS.items():
        requests.append({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": q},
                ],
            }],
            "max_generate_length": MAX_TOK[name],
        })
        meta.append({"source": source, "clip_id": clip_id, "frame_idx": frame_idx, "question": name})


# ---------------------------------------------------------------- BDD100k
bdd_dir = f"{NEG}/images"
bdd_files = sorted(os.listdir(bdd_dir))
for fn in bdd_files:
    add_requests("bdd100k", fn.replace(".jpg", ""), 0, f"{bdd_dir}/{fn}")
print(f"BDD100k: {len(bdd_files)} imagens, {len(bdd_files) * len(QUESTIONS)} requisicoes")

# ---------------------------------------------------------------- ROADWork (unused snippets)
with open(f"{NEG}/_roadwork_unused_sample_100.txt") as f:
    snippet_paths = [l.strip() for l in f if l.strip()]

n_rw_frames = 0
for sp in snippet_paths:
    clip_id = os.path.basename(sp).replace("_snippet.mp4", "")
    cap = cv2.VideoCapture(sp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        continue
    # 2 frames por clipe: 25% e 75% da duracao (evita abertura/fechamento truncados)
    for frac, tag in ((0.25, "a"), (0.75, "b")):
        fidx = int(total * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        fpath = f"{FRAME_DIR}/{clip_id}_{tag}.jpg"
        cv2.imwrite(fpath, frame)
        add_requests("roadwork_unused", clip_id, fidx, fpath)
        n_rw_frames += 1
    cap.release()
print(f"ROADWork (unused): {n_rw_frames} frames de {len(snippet_paths)} clipes, "
      f"{n_rw_frames * len(QUESTIONS)} requisicoes")

batch = {
    "batch_size": 1,
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 80,
    "requests": requests,
}

with open(f"{ENG}/input_negcontrol_batch.json", "w") as f:
    json.dump(batch, f)
with open(f"{ENG}/input_negcontrol_batch_meta.json", "w") as f:
    json.dump(meta, f)

print(f"\nTotal de requisicoes: {len(requests)}")
print(f"JSON salvo: input_negcontrol_batch.json "
      f"({os.path.getsize(f'{ENG}/input_negcontrol_batch.json')/1e6:.1f} MB)")
