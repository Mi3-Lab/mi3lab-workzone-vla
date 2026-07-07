"""Valida a degradação de qualidade do INT4 AWQ antes do build final no Jetson.

O checkpoint exportado (models/workzone-2b-stage6-int4awq/) tem pesos
empacotados (U8 + weight_scale) — formato consumido pelo compilador C++ do
TensorRT Edge-LLM pra virar engine, não carregável direto via transformers.
E o engine em si precisa ser compilado NO PRÓPRIO Jetson (arquitetura da GPU
do Orin != A100 daqui).

Então validamos a via mais direta: aplicamos a MESMA receita AWQ (modelopt,
fake-quantização) no modelo original em memória — simula a perda numérica do
INT4 sem exportar/empacotar nada — e comparamos as respostas do modelo
original (bf16) vs fake-quantizado (int4) nas 5 perguntas do v16, em frames
reais dos 4 vídeos de work zone.
"""
import os, sys
import cv2
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})

import modelopt.torch.quantization as mtq
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"

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
    # amostra uniforme cobrindo antes/aproximando/dentro/saindo da zona,
    # evitando os primeiros/ultimos 5% (frames de transicao de video)
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


print("[1] Carregando modelo base (bf16)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

frames = {(c, i): grab_frame(c, i) for c, i in CLIPS}

print(f"\n[2] Coletando respostas BF16 (referência) — {len(CLIPS)} frames...")
bf16_answers = {}
for key in CLIPS:
    city, idx = key
    bf16_answers[key] = {}
    for name, q in QUESTIONS.items():
        ans = ask(model, proc, tok, frames[key], q, MAX_TOK[name])
        bf16_answers[key][name] = ans
        print(f"    [{city:16s}#{idx:4d}] {name:8s}: {ans[:90]!r}")

print("\n[3] Aplicando fake-quantizacao INT4 AWQ (mesma receita do export)...")


def calib_forward_loop(m):
    calib_cities = list(dict.fromkeys(c for c, _ in CLIPS))  # 1 frame por cidade
    for city in calib_cities:
        idx = next(i for c, i in CLIPS if c == city)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": frames[(city, idx)]},
            {"type": "text", "text": QUESTIONS["DESC"]},
        ]}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            m(**inputs)


mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop=calib_forward_loop)
print("    Fake-quantizacao aplicada.")

print(f"\n[4] Coletando respostas INT4 AWQ (fake-quant) — {len(CLIPS)} frames...")
int4_answers = {}
for key in CLIPS:
    city, idx = key
    int4_answers[key] = {}
    for name, q in QUESTIONS.items():
        ans = ask(model, proc, tok, frames[key], q, MAX_TOK[name])
        int4_answers[key][name] = ans
        print(f"    [{city:16s}#{idx:4d}] {name:8s}: {ans[:90]!r}")

print("\n" + "=" * 100)
print("COMPARATIVO BF16 vs INT4 AWQ (fake-quant)")
print("=" * 100)
n_match, n_total = 0, 0
per_question = {name: [0, 0] for name in QUESTIONS}  # [match, total]
for key in CLIPS:
    city, idx = key
    print(f"\n--- {city} #{idx} ---")
    for name in QUESTIONS:
        b, i = bf16_answers[key][name], int4_answers[key][name]
        same = b.strip().lower() == i.strip().lower()
        n_total += 1
        n_match += int(same)
        per_question[name][1] += 1
        per_question[name][0] += int(same)
        tag = "IGUAL" if same else "DIFERENTE"
        print(f"  {name:8s} [{tag}]")
        print(f"    BF16: {b[:100]!r}")
        print(f"    INT4: {i[:100]!r}")

print(f"\nRespostas idênticas: {n_match}/{n_total} ({100*n_match/n_total:.0f}%)")
print("\nPor pergunta:")
for name, (m_, t_) in per_question.items():
    print(f"  {name:8s}: {m_}/{t_} ({100*m_/t_:.0f}%)")
print("\n=== VALIDACAO CONCLUIDA ===")
