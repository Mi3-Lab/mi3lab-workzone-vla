"""Stage 7.1 — corrige a regressao do stage7: DESC e TRAFFIC (perguntas v16)
so tinham exemplos NEGATIVOS no mix do stage7 (as perguntas nunca existiram
no stage6), entao o modelo colapsou pra resposta negativa mesmo em obra
(DESC 92-100% negativa-canonica nos 4 videos de obra).

Mix = stage7 completo (20.5k) + positivos novos:
  D. DESC-positivos: pergunta v16-DESC pareada com as imagens+respostas das
     linhas 'Provide a comprehensive description...' (mesmo formato de
     resposta que o v16 espera).                              ~700 linhas
  E. TRAFFIC-positivos: sintetizados do campo 'Lane status: X' presente nas
     respostas de descricao (X mapeado pra frase com keyword que o parser
     do v16 reconhece).                                      ~1300 linhas
"""
import os, random, re
import pandas as pd

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
SRC = f"{BASE}/data/roadwork/lingoqa_egomix_v3"
S7 = f"{BASE}/data/roadwork/lingoqa_stage7"
OUT = f"{BASE}/data/roadwork/lingoqa_stage7_1"
random.seed(71)

Q_DESC = "Describe the work zone elements visible in this scene."
Q_TRAFFIC = "How is traffic flow affected in this scene?"

LANE_RE = re.compile(r"Lane status:\s*([a-z ]+?)[\.\;\n]", re.I)
TRAFFIC_ANSWER = {
    "partially blocked": "Traffic is affected: the lane is partially blocked by work zone elements.",
    "fully blocked": "Traffic is affected: the lane is fully blocked by the work zone.",
    "open but affected": "Traffic flows, but it is affected by the adjacent work zone.",
}

os.makedirs(OUT, exist_ok=True)
for sub in ("images", "scene_images", "general_frames", "bdd", "bdd_neg"):
    dst = os.path.join(OUT, sub)
    if not os.path.exists(dst):
        os.symlink(os.path.join(S7, sub) if sub == "bdd_neg" else os.path.join(SRC, sub), dst)

df = pd.read_parquet(f"{SRC}/train.parquet")
rows = []
qid = 0


def add(images, question, answer, seg):
    global qid
    rows.append({
        "question_id": f"s71_{qid:06d}",
        "segment_id": seg,
        "images": images,
        "question": question,
        "answer": answer,
    })
    qid += 1


# D. DESC positivos (formato ja identico ao esperado)
comp = df[df["question"].str.startswith("Provide a comprehensive")]
for _, r in comp.iterrows():
    add(list(r["images"]), Q_DESC, r["answer"], f"descpos_{r['segment_id']}")
n_desc = len(rows)
print(f"D. DESC-positivos: {n_desc}")

# E. TRAFFIC positivos (sintetizados do Lane status)
lane_rows = df[df["answer"].str.contains("Lane status", na=False)]
seen_img = set()
n_traffic = 0
for _, r in lane_rows.iterrows():
    key = tuple(r["images"])
    if key in seen_img:
        continue
    m = LANE_RE.search(r["answer"])
    if not m:
        continue
    status = m.group(1).strip().lower()
    ans = TRAFFIC_ANSWER.get(status)
    if ans is None:
        continue
    seen_img.add(key)
    add(list(r["images"]), Q_TRAFFIC, ans, f"trafpos_{r['segment_id']}")
    n_traffic += 1
print(f"E. TRAFFIC-positivos: {n_traffic}")

new_df = pd.DataFrame(rows)
s7_train = pd.read_parquet(f"{S7}/train.parquet")
s7_val = pd.read_parquet(f"{S7}/val.parquet")
merged = pd.concat([s7_train, new_df], ignore_index=True).sample(frac=1, random_state=71)
merged.to_parquet(f"{OUT}/train.parquet")
s7_val.to_parquet(f"{OUT}/val.parquet")
print(f"train: {len(merged)} (stage7 {len(s7_train)} + novos {len(new_df)}) | val: {len(s7_val)}")
print(f"salvo em {OUT}/")
