"""Compara BF16 vs INT4 AWQ v1 (512 amostras) vs INT4 AWQ v2 (2000 amostras).

Correção em relação ao validate_int4awq.py original: aquele calibrava com
imagem+pergunta dos próprios frames de teste, o que NÃO é a receita usada na
quantização real (tensorrt-edgellm-quantize usa só texto, via
_text_calib_dataloader, já que visual_quantization não está habilitado).
Aqui replicamos a receita real: calibração só-texto a partir dos mesmos
arquivos JSONL usados no export (calib_workzone.jsonl / calib_workzone_v2.jsonl).

Cada variante quantiza um modelo CARREGADO DO ZERO (fake-quant em memória,
sem exportar/empacotar) pra evitar quantização composta entre passes.
"""
import os, sys, json
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

sys.path.insert(0, f"{BASE}/TensorRT-Edge-LLM")
from tensorrt_edgellm.quantization.quantization_configs import build_quant_config

# Receita EXATA do tensorrt-edgellm-quantize (mesma usada nos checkpoints reais
# int4awq/int4awq-v2): torre visual e KV-cache ficam de fora por padrao. Usar
# mtq.INT4_AWQ_CFG cru (sem essa funcao) tambem quantiza a torre visual e liga
# calibracao de KV-cache — simula um checkpoint DIFERENTE do que foi exportado,
# e e ~10x mais lento (achado ao investigar por que a calibracao de 2000
# amostras nao terminava em 40 min, vs 219s no job de quantizacao real).
QUANT_CFG = build_quant_config(quantization="int4_awq")

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
CALIB_V1 = f"{BASE}/data/roadwork/calib_workzone.jsonl"      # 512 usadas de 640
CALIB_V2 = f"{BASE}/data/roadwork/calib_workzone_v2.jsonl"   # 2000

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


def make_calib_forward_loop(tok, texts):
    """Replica _text_calib_dataloader do tensorrt_edgellm: só texto, sem imagem."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = enc["input_ids"]

    def forward_loop(m):
        for i in range(input_ids.shape[0]):
            batch = input_ids[i : i + 1].cuda()
            with torch.no_grad():
                m(input_ids=batch)

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


# ── Passe 1: BF16 (referência) ────────────────────────────────────────────────
model = load_fresh_model()
bf16_answers = collect_answers(model, "BF16")
del model
torch.cuda.empty_cache()

# ── Passe 2: INT4 AWQ v1 (512 amostras, calib_workzone.jsonl) ────────────────
print(f"\n[INT4-v1] Calibrando com {CALIB_V1} (512 amostras, texto puro)...")
texts_v1 = load_calib_texts(CALIB_V1, 512)
model = load_fresh_model()
mtq.quantize(model, QUANT_CFG, forward_loop=make_calib_forward_loop(tok, texts_v1))
int4_v1_answers = collect_answers(model, "INT4-v1")
del model
torch.cuda.empty_cache()

# ── Passe 3: INT4 AWQ v2 (2000 amostras, calib_workzone_v2.jsonl) ────────────
print(f"\n[INT4-v2] Calibrando com {CALIB_V2} (2000 amostras, texto puro)...")
texts_v2 = load_calib_texts(CALIB_V2, 2000)
model = load_fresh_model()
mtq.quantize(model, QUANT_CFG, forward_loop=make_calib_forward_loop(tok, texts_v2))
int4_v2_answers = collect_answers(model, "INT4-v2")
del model
torch.cuda.empty_cache()

# ── Comparativo ───────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("COMPARATIVO BF16 vs INT4-v1 (512) vs INT4-v2 (2000)")
print("=" * 100)

per_question_v1 = {name: [0, 0] for name in QUESTIONS}
per_question_v2 = {name: [0, 0] for name in QUESTIONS}
n_v1, n_v2, n_total = 0, 0, 0

for key in CLIPS:
    city, idx = key
    for name in QUESTIONS:
        b = bf16_answers[key][name].strip().lower()
        i1 = int4_v1_answers[key][name].strip().lower()
        i2 = int4_v2_answers[key][name].strip().lower()
        same_v1, same_v2 = b == i1, b == i2
        n_total += 1
        n_v1 += int(same_v1)
        n_v2 += int(same_v2)
        per_question_v1[name][1] += 1
        per_question_v1[name][0] += int(same_v1)
        per_question_v2[name][1] += 1
        per_question_v2[name][0] += int(same_v2)
        if same_v1 != same_v2:
            print(f"\n[{city}#{idx} {name}] DIVERGENCIA v1 vs v2:")
            print(f"    BF16   : {bf16_answers[key][name][:100]!r}")
            print(f"    INT4-v1: {int4_v1_answers[key][name][:100]!r}  [{'IGUAL' if same_v1 else 'DIFERENTE'}]")
            print(f"    INT4-v2: {int4_v2_answers[key][name][:100]!r}  [{'IGUAL' if same_v2 else 'DIFERENTE'}]")

print(f"\nINT4-v1 (512 amostras) : {n_v1}/{n_total} ({100*n_v1/n_total:.0f}%)")
print(f"INT4-v2 (2000 amostras): {n_v2}/{n_total} ({100*n_v2/n_total:.0f}%)")

print("\nPor pergunta:")
print(f"  {'':8s} {'v1 (512)':>12s} {'v2 (2000)':>12s}")
for name in QUESTIONS:
    m1, t1 = per_question_v1[name]
    m2, t2 = per_question_v2[name]
    print(f"  {name:8s} {m1}/{t1} ({100*m1/t1:3.0f}%)   {m2}/{t2} ({100*m2/t2:3.0f}%)")

print("\n=== VALIDACAO v1 vs v2 CONCLUIDA ===")
