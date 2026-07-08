"""Compara BF16 vs INT4-v1 (512, texto puro) vs INT4-ego (530, multimodal foco EGO).

Mesma metodologia de validate_int4awq_v1v2.py (receita build_quant_config
exata, fake-quant em memória, sem exportar). A calibração "ego" replica
EXATAMENTE a mistura usada no export real (quantize_int4awq_ego.py): 330
pares imagem+pergunta reais do lingoqa_egostate (balanceados nos 3 estados)
+ 200 amostras de texto puro de outras tarefas.
"""
import os, sys, json, random
import cv2
import pandas as pd
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})

import modelopt.torch.quantization as mtq
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, f"{BASE}/TensorRT-Edge-LLM")
from tensorrt_edgellm.quantization.quantization_configs import build_quant_config

QUANT_CFG = build_quant_config(quantization="int4_awq")
MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
CALIB_V1 = f"{BASE}/data/roadwork/calib_workzone.jsonl"

QUESTIONS = {
    "EGO":     "Regarding the road work zone, what is this vehicle's current position status: OUTSIDE, APPROACHING, or INSIDE?",
    "TRAFFIC": "How is traffic flow affected in this scene?",
    "ACTIVE":  "Is this an active work zone with workers present, or a passive zone?",
    "SIGN":    "Read the text on the temporary traffic control signs in this scene.",
    "DESC":    "Describe the work zone elements visible in this scene.",
}
MAX_TOK = {"EGO": 10, "TRAFFIC": 12, "ACTIVE": 10, "SIGN": 25, "DESC": 60}

FRAME_COUNTS = {"boston": 451, "denver": 455, "chicago": 627, "seattle_unseen": 900}
N_PER_CITY = 7
CLIPS = []
for city, total in FRAME_COUNTS.items():
    lo, hi = int(0.05 * total), int(0.95 * total)
    step = (hi - lo) // (N_PER_CITY - 1)
    for k in range(N_PER_CITY):
        CLIPS.append((city, lo + k * step))


def grab_frame(city, frame_idx):
    cap = cv2.VideoCapture(f"{BASE}/videos/{city}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"failed to read {city} frame {frame_idx}"
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def ask(model, proc, tok, pil, question, max_new_tokens):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": question},
    ]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_calib_texts(jsonl_path, num_samples):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= num_samples:
                break
    return texts


def make_text_calib_forward_loop(tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = enc["input_ids"]

    def forward_loop(m):
        for i in range(input_ids.shape[0]):
            with torch.no_grad():
                m(input_ids=input_ids[i : i + 1].cuda())

    return forward_loop


def build_ego_calib_rows(seed=44, n_per_state=110):
    random.seed(seed)
    egostate_dir = f"{BASE}/data/roadwork/lingoqa_egostate"
    df_ego = pd.read_parquet(f"{egostate_dir}/train.parquet")
    df_ego["state"] = df_ego["answer"].str.extract(r"^(OUTSIDE|APPROACHING|INSIDE)")
    rows = []
    for state in ["OUTSIDE", "APPROACHING", "INSIDE"]:
        sub = df_ego[df_ego["state"] == state]
        idx = random.sample(range(len(sub)), min(n_per_state, len(sub)))
        for i in idx:
            row = sub.iloc[i]
            img_path = os.path.normpath(os.path.join(egostate_dir, row["images"][0]))
            rows.append(("mm", img_path, row["question"], None))
    text_sources = {
        f"{BASE}/data/roadwork/lingoqa_combined/train.parquet": 80,
        f"{BASE}/data/roadwork/lingoqa_signtext/train.parquet": 60,
        f"{BASE}/data/roadwork/lingoqa_general/train.parquet": 60,
    }
    for path, n in text_sources.items():
        df = pd.read_parquet(path)
        idx = random.sample(range(len(df)), min(n, len(df)))
        for i in idx:
            q, a = str(df.iloc[i]["question"]), str(df.iloc[i]["answer"])
            rows.append(("text", None, None, f"{q} {a}"))
    random.shuffle(rows)
    return rows


def make_ego_calib_forward_loop(proc, tok, rows):
    def forward_loop(m):
        for kind, img_path, question, text in rows:
            with torch.no_grad():
                if kind == "mm":
                    pil = Image.open(img_path).convert("RGB")
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": question},
                    ]}]
                    inputs = proc.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt",
                    ).to("cuda")
                    m(**inputs)
                else:
                    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
                    m(input_ids=enc["input_ids"].cuda())

    return forward_loop


def load_fresh_model():
    return AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()


proc = AutoProcessor.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

frames = {(c, i): grab_frame(c, i) for c, i in CLIPS}


def collect_answers(model, tag):
    print(f"\n[{tag}] Coletando respostas — {len(CLIPS)} frames...")
    answers = {}
    for key in CLIPS:
        city, idx = key
        answers[key] = {}
        for name, q in QUESTIONS.items():
            ans = ask(model, proc, tok, frames[key], q, MAX_TOK[name])
            answers[key][name] = ans
        print(f"    [{city:16s}#{idx:4d}] " +
              " | ".join(f"{n}={answers[key][n][:30]!r}" for n in QUESTIONS))
    return answers


# ── Passe 1: BF16 ──────────────────────────────────────────────────────────────
model = load_fresh_model()
bf16_answers = collect_answers(model, "BF16")
del model
torch.cuda.empty_cache()

# ── Passe 2: INT4-v1 (512 texto puro) ─────────────────────────────────────────
print(f"\n[INT4-v1] Calibrando com {CALIB_V1} (512 amostras, texto puro)...")
texts_v1 = load_calib_texts(CALIB_V1, 512)
model = load_fresh_model()
mtq.quantize(model, QUANT_CFG, forward_loop=make_text_calib_forward_loop(tok, texts_v1))
int4_v1_answers = collect_answers(model, "INT4-v1")
del model
torch.cuda.empty_cache()

# ── Passe 3: INT4-ego (330 multimodal EGO balanceado + 200 texto) ────────────
print(f"\n[INT4-ego] Calibrando com 530 amostras (330 multimodal EGO + 200 texto)...")
ego_rows = build_ego_calib_rows()
model = load_fresh_model()
mtq.quantize(model, QUANT_CFG, forward_loop=make_ego_calib_forward_loop(proc, tok, ego_rows))
int4_ego_answers = collect_answers(model, "INT4-ego")
del model
torch.cuda.empty_cache()

# ── Comparativo ───────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("COMPARATIVO BF16 vs INT4-v1 (512 texto) vs INT4-ego (530 multimodal EGO)")
print("=" * 100)

per_question_v1 = {name: [0, 0] for name in QUESTIONS}
per_question_ego = {name: [0, 0] for name in QUESTIONS}
n_v1, n_ego, n_total = 0, 0, 0
ego_state_confusion = {"v1": [0, 0], "ego": [0, 0]}  # [certo, total] so pra EGO

for key in CLIPS:
    city, idx = key
    for name in QUESTIONS:
        b = bf16_answers[key][name].strip().lower()
        i1 = int4_v1_answers[key][name].strip().lower()
        ie = int4_ego_answers[key][name].strip().lower()
        same_v1, same_ego = b == i1, b == ie
        n_total += 1
        n_v1 += int(same_v1)
        n_ego += int(same_ego)
        per_question_v1[name][1] += 1
        per_question_v1[name][0] += int(same_v1)
        per_question_ego[name][1] += 1
        per_question_ego[name][0] += int(same_ego)
        if name == "EGO":
            # compara so o PRIMEIRO token do estado (robusto a truncamento leve)
            b_state = b.split(".")[0].split()[0] if b else ""
            i1_state = i1.split(".")[0].split()[0] if i1 else ""
            ie_state = ie.split(".")[0].split()[0] if ie else ""
            ego_state_confusion["v1"][1] += 1
            ego_state_confusion["v1"][0] += int(b_state == i1_state)
            ego_state_confusion["ego"][1] += 1
            ego_state_confusion["ego"][0] += int(b_state == ie_state)
        if same_v1 != same_ego:
            print(f"\n[{city}#{idx} {name}] DIVERGENCIA v1 vs ego:")
            print(f"    BF16    : {bf16_answers[key][name][:100]!r}")
            print(f"    INT4-v1 : {int4_v1_answers[key][name][:100]!r}  [{'IGUAL' if same_v1 else 'DIFERENTE'}]")
            print(f"    INT4-ego: {int4_ego_answers[key][name][:100]!r}  [{'IGUAL' if same_ego else 'DIFERENTE'}]")

print(f"\nINT4-v1  (512 texto)         : {n_v1}/{n_total} ({100*n_v1/n_total:.0f}%)")
print(f"INT4-ego (530 multimodal EGO): {n_ego}/{n_total} ({100*n_ego/n_total:.0f}%)")

print("\nPor pergunta:")
print(f"  {'':8s} {'v1 (512 txt)':>14s} {'ego (530 mm)':>14s}")
for name in QUESTIONS:
    m1, t1 = per_question_v1[name]
    me, te = per_question_ego[name]
    print(f"  {name:8s} {m1}/{t1} ({100*m1/t1:3.0f}%)    {me}/{te} ({100*me/te:3.0f}%)")

print("\nEGO — match do ESTADO (primeira palavra, robusto a truncamento):")
v1c, v1t = ego_state_confusion["v1"]
egc, egt = ego_state_confusion["ego"]
print(f"  INT4-v1 : {v1c}/{v1t} ({100*v1c/v1t:.0f}%)")
print(f"  INT4-ego: {egc}/{egt} ({100*egc/egt:.0f}%)")

print("\n=== VALIDACAO EGO-FOCUSED CONCLUIDA ===")
