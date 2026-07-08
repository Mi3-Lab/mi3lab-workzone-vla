"""Stage 7 — hard negatives + restauracao do gate.

Motivacao (validacao de 2026-07-08): o stage6 nunca pareou as perguntas de
work zone (exceto EGO, so em BDD diurno) com cenas SEM obra. Resultado: o
modelo responde afirmativo a qualquer pergunta de obra em cena limpa (gate
95% Yes em BDD; 100% APPROACHING em rodovia noturna), enquanto o proprio
Cosmos-2B base discriminava perfeitamente (88% Yes obra / 0% Yes limpo).

Mix:
  A. NEGATIVOS (2500 imagens BDD frescas, fora do treino antigo e fora do
     benchmark): GATE + EGO + 2 sorteadas de {SIGN, ACTIVE, TRAFFIC, DESC}
     por imagem, com respostas negativas canonicas.       ~10.000 linhas
  B. GATE-POSITIVOS (imagens de obra ja usadas no stage6):
     pergunta GATE com resposta Yes.                       ~2.500 linhas
  C. REPLAY do stage6 (amostra do egomix_v3):              ~8.000 linhas
                                                    total  ~20.500 linhas
"""
import os, random, glob, shutil
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
SRC = f"{BASE}/data/roadwork/lingoqa_egomix_v3"
OUT = f"{BASE}/data/roadwork/lingoqa_stage7"
BDD_NEG_DIR = f"{BASE}/data/roadwork/bdd100k_stage7"
random.seed(7)

GATE_Q = ("Are there any road work indicators in this scene "
          "(cones, barriers, temporary signs, workers, work vehicles)? "
          "Answer Yes or No.")
Q = {
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
NEG_ANSWER = {
    "GATE":    "No. There are no road work indicators in this scene.",
    "EGO":     "OUTSIDE. No work zone elements near the vehicle. The road ahead is clear of construction activity.",
    "TRAFFIC": "Traffic is clear; there is no work zone affecting traffic flow in this scene.",
    "ACTIVE":  "There is no work zone in this scene.",
    "SIGN":    "No legible sign text in this scene.",
    "DESC":    "No work zone elements visible in this scene.",
}
GATE_POS_ANSWER = "Yes. There are road work indicators in this scene."

os.makedirs(OUT, exist_ok=True)

# --- links das raizes de imagem existentes (replay usa esses prefixos) ---
for sub in ("images", "scene_images", "general_frames", "bdd"):
    dst = os.path.join(OUT, sub)
    if not os.path.exists(dst):
        os.symlink(os.path.join(SRC, sub), dst)
# raiz nova pros negativos
dst = os.path.join(OUT, "bdd_neg")
if not os.path.exists(dst):
    os.symlink(BDD_NEG_DIR, dst)

rows = []
qid = 0


def add(images, question, answer, seg):
    global qid
    rows.append({
        "question_id": f"s7_{qid:06d}",
        "segment_id": seg,
        "images": images,
        "question": question,
        "answer": answer,
    })
    qid += 1


# ------------------------------------------------------- A. negativos
neg_imgs = sorted(os.path.basename(p) for p in glob.glob(f"{BDD_NEG_DIR}/*.jpg"))
print(f"negativos BDD disponiveis: {len(neg_imgs)}")
optional = ["SIGN", "ACTIVE", "TRAFFIC", "DESC"]
for fn in neg_imgs:
    img = [f"bdd_neg/{fn}"]
    seg = f"neg_{fn[:-4]}"
    add(img, GATE_Q, NEG_ANSWER["GATE"], seg)
    add(img, Q["EGO"], NEG_ANSWER["EGO"], seg)
    for qname in random.sample(optional, 2):
        add(img, Q[qname], NEG_ANSWER[qname], seg)
n_neg = len(rows)
print(f"A. linhas negativas: {n_neg}")

# ------------------------------------------------------- B. gate-positivos
df = pd.read_parquet(f"{SRC}/train.parquet")
ego = df[df["question"].str.startswith("Regarding the road work zone")]
pos = ego[ego["answer"].str.startswith(("INSIDE", "APPROACHING"))]
pos_imgs = list({tuple(i) for i in pos["images"]})
random.shuffle(pos_imgs)
# Balancear com os 2500 gate-negativos: se ha menos imagens unicas de obra,
# repete (com shuffle) ate ~2500 pra nao ensinar o gate a sub-disparar.
gate_pos = []
while len(gate_pos) < 2500:
    gate_pos.extend(pos_imgs)
for images in gate_pos[:2500]:
    add(list(images), GATE_Q, GATE_POS_ANSWER, "gatepos")
print(f"B. linhas gate-positivas: {len(rows) - n_neg}")

# ------------------------------------------------------- C. replay
replay = df.sample(n=8000, random_state=7)
for _, r in replay.iterrows():
    add(list(r["images"]), r["question"], r["answer"], r["segment_id"])
print(f"C. linhas replay: 8000")

random.shuffle(rows)
out_df = pd.DataFrame(rows)
val_df = out_df.iloc[:200]
train_df = out_df.iloc[200:]
train_df.to_parquet(f"{OUT}/train.parquet")
val_df.to_parquet(f"{OUT}/val.parquet")
print(f"\ntrain: {len(train_df)} | val: {len(val_df)}")
print(f"salvo em {OUT}/")
