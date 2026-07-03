"""Video inference v12 — estado ego-cêntrico APRENDIDO (Stage 5) + HMM.

Fix vs v11: a descrição gerada pelo modelo inclui uma frase memorizada
"Scene: {env} environment in {city}, ..." porque o dado de treino
(convert_to_lingoqa.py) cola o nome real da cidade vindo do metadado do
dataset em toda resposta. O modelo não infere a cidade da imagem — decorou
o template. Essa frase é removida por regex do texto exibido no painel,
para não sugerir uma capacidade de percepção que não existe.

Fix vs v10: sensores usam geração real (model.vlm.generate) + parsing, não
scoring por logprob via forward manual — este último dá distribuições
congeladas no checkpoint Stage-5 (bug de preprocessing/posições, não do
modelo; validado com job 170352, generate() responde corretamente).

Diferença vs v9: o sensor principal agora é a pergunta ego-cêntrica que o
Stage 5 ENSINOU ao modelo (labels auto-derivados da geometria das anotações
ROADWork). O modelo responde OUTSIDE/APPROACHING/INSIDE considerando a posição
do VEÍCULO em relação à zona — não o conteúdo da cena. Ver workers a 100 m é
APPROACHING; INSIDE só quando os elementos estão ao lado do veículo.

EXITING não tem label de imagem única: emerge da dinâmica do HMM (estando
INSIDE, evidência de zona sumindo força a crença a passar por EXITING).

Sensores:
  ego     (principal) — 3 respostas treinadas do Stage 5, in-domain
  traffic (secundário) — vocabulário ROADWork, discrimina zona presente/ausente
  active  (painel + emissão fraca) — workers presentes na cena
"""
import os, re, sys, textwrap
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoProcessor
import glob

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

VIDEO_IN  = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/videos/boston.mp4"
VIDEO_OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/outputs/v12/v12_seattle_unseen.mp4"
CKPT      = sys.argv[3] if len(sys.argv) > 3 else f"{BASE}/checkpoints/sft_stage5_egostate"

# Se CKPT é o diretório de treino, usar o checkpoint-* mais recente
if not glob.glob(os.path.join(CKPT, "model*.safetensors")):
    cands = sorted(glob.glob(os.path.join(CKPT, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    if cands:
        CKPT = cands[-1]

STATE_EVERY = 9    # crença a cada N frames (0.3 s @ 30fps)
DESC_EVERY  = 15

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"

print(f"Video  : {os.path.basename(VIDEO_IN)}")
print(f"Output : {VIDEO_OUT}")
print(f"CKPT   : {CKPT}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

from alpamayo1_5_sft.models.kd_model import build_student_model
from workzone_state import WZState

print("\nLoading Alpamayo 2B Stage-5 (ego-centric state)...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
ckpt_file = os.path.join(CKPT, "model.safetensors")
if os.path.exists(ckpt_file):
    sd = load_file(ckpt_file, device="cpu")
else:
    sd = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        sd.update(load_file(f, device="cpu"))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
model = model.to(torch.bfloat16).cuda().eval()

proc_qa = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)

print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("Model ready!\n")


# ══ Sensores ═════════════════════════════════════════════════════════════════
# candidate_probs() (forward manual + teacher-forcing) dá distribuições
# congeladas no checkpoint Stage-5 — não bate com o preprocessing usado no
# treino (mRoPE/posições divergem entre forward() direto e generate()).
# Validado (job 170352): model.vlm.generate() de verdade responde
# corretamente e varia com a cena. Sensores usam geração real + parsing.
STATES      = [WZState.OUTSIDE, WZState.APPROACHING, WZState.INSIDE, WZState.EXITING]
STATE_NAMES = ["OUTSIDE", "APPROACHING", "INSIDE", "EXITING"]

# SENSOR PRINCIPAL: pergunta ego-cêntrica ensinada no Stage 5 (in-domain)
Q_EGO = (
    "Regarding the road work zone, what is this vehicle's current position status: "
    "OUTSIDE, APPROACHING, or INSIDE?"
)
EGO_SHORT = ["OUTSIDE", "APPROACHING", "INSIDE"]

Q_TRAFFIC = "How is traffic flow affected in this scene?"
TRAFFIC_SHORT = ["partial block", "lane shift", "fully blocked", "clear"]

Q_ACTIVE = "Is this an active work zone with workers present, or a passive zone?"
ACTIVE_SHORT = ["ACTIVE workers", "passive devices"]

# Confiança atribuída à resposta greedy (pico da distribuição no rótulo
# decodificado; resto espalhado). Evita colapso instantâneo da crença por
# um único frame ruidoso, mantendo o update Bayesiano bem-formado.
PEAK = 0.80


def peaked_dist(idx, n):
    p = np.full(n, (1.0 - PEAK) / max(n - 1, 1))
    p[idx] = PEAK
    return p


# Emissões P(rótulo decodificado | estado)
# Ego (aprendido): quase identidade. EXITING responde misto OUT/INS (zona
# ficando para trás), o que junto com a dinâmica identifica a saída.
#                 OUT    APP    INS
EGO_EMIT = np.array([
    [0.85, 0.12, 0.03],   # OUTSIDE
    [0.10, 0.75, 0.15],   # APPROACHING
    # INSIDE: leitura "APPROACHING" é COMUM dentro de zona esparsa (rodovia,
    # poucos cones — o sensor vê elementos à frente e responde APPROACHING).
    # Falta de evidência ao lado != evidência de saída. A leitura APP precisa
    # ser ~neutra APP-vs-INS, senão leituras repetidas viram o filtro.
    [0.05, 0.55, 0.40],   # INSIDE
    [0.50, 0.25, 0.25],   # EXITING
])
# Cena: só discrimina zona presente/ausente (feedback ego-cêntrico do usuário)
# INSIDE com "clear" também é comum em trecho esparso → 0.30 (não 0.18)
#                     partial  shift   full   none
TRAFFIC_EMIT = np.array([
    [0.10, 0.03, 0.02, 0.85],   # OUTSIDE
    [0.50, 0.14, 0.13, 0.23],   # APPROACHING
    [0.45, 0.14, 0.11, 0.30],   # INSIDE
    [0.30, 0.06, 0.05, 0.59],   # EXITING
])
#                     active  passive
ACTIVE_EMIT = np.array([
    [0.10, 0.90],   # OUTSIDE
    [0.45, 0.55],   # APPROACHING
    [0.50, 0.50],   # INSIDE
    [0.25, 0.75],   # EXITING
])

# HMM (passo 0.3 s). Física da saída: INSIDE→EXITING→OUTSIDE, sem atalhos.
# EXITING→APPROACHING = 0: voltar a "aproximando" exige passar por OUTSIDE
# (fecha a rota de fuga INSIDE→EXITING→APPROACHING que causava regressão
# de estado em trechos com pouca evidência).
TRANS = np.array([
    # OUT   APP   INS   EXI
    [0.95, 0.05, 0.00, 0.00],   # OUTSIDE
    [0.02, 0.88, 0.10, 0.00],   # APPROACHING
    [0.00, 0.00, 0.98, 0.02],   # INSIDE (pegajoso: sai só com evidência sustentada)
    [0.06, 0.00, 0.06, 0.88],   # EXITING (drena devagar: rótulo visível na saída)
])
PRIOR = np.array([0.55, 0.20, 0.20, 0.05])


@torch.no_grad()
def generate_short(pil, question, max_new_tokens=16):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text": question},
    ]}]
    text   = proc_qa.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc_qa(text=[text], images=[pil], return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=proc_qa.tokenizer.eos_token_id,
            return_dict_in_generate=False, output_logits=False)
    n = inputs["input_ids"].shape[1]
    return proc_qa.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def classify_answer(answer, labels):
    a = answer.lower()
    for i, lab in enumerate(labels):
        if lab.lower() in a:
            return i
    return None  # sem match: sensor não opina neste frame


def state_emission(pil):
    ego_txt = generate_short(pil, Q_EGO, max_new_tokens=10)
    ego_i   = classify_answer(ego_txt, EGO_SHORT)
    p_ego   = peaked_dist(ego_i, 3) if ego_i is not None else np.ones(3) / 3

    traffic_txt = generate_short(pil, Q_TRAFFIC, max_new_tokens=12)
    tr_i = None
    tl = traffic_txt.lower()
    if "partially blocked" in tl or "partial" in tl:      tr_i = 0
    elif "shift" in tl:                                    tr_i = 1
    elif "fully blocked" in tl or "fully" in tl:            tr_i = 2
    elif "clear" in tl or "no lane" in tl:                  tr_i = 3
    p_traffic = peaked_dist(tr_i, 4) if tr_i is not None else np.ones(4) / 4

    active_txt = generate_short(pil, Q_ACTIVE, max_new_tokens=10)
    al = active_txt.lower()
    ac_i = 0 if "active" in al else (1 if "passive" in al else None)
    p_active = peaked_dist(ac_i, 2) if ac_i is not None else np.ones(2) / 2

    e = (EGO_EMIT @ p_ego) * (TRAFFIC_EMIT @ p_traffic) * (ACTIVE_EMIT @ p_active)
    return e / e.sum(), p_ego, p_traffic, p_active


class BayesFilter:
    def __init__(self):
        self.belief = PRIOR.copy()

    def update(self, emission):
        b = (self.belief @ TRANS) * emission
        self.belief = b / b.sum()
        return self.belief

    @property
    def state(self):
        return STATES[int(self.belief.argmax())]


# ══ Descrição (painel) ═══════════════════════════════════════════════════════
Q_DESC = "Describe the work zone elements visible in this scene."

# Remove a cláusula "Scene: {env} environment in {city}, {weather} conditions,
# {daytime} lighting." — string memorizada do template de treino
# (convert_to_lingoqa.py insere o city_name real do metadado), não uma
# inferência visual. Exibi-la sugeriria uma capacidade de percepção que o
# modelo não tem.
_CITY_CLAUSE_RE = re.compile(
    r"\s*Scene:\s*\w+\s+environment\s+in\s+[^,]+,\s*[^,]*conditions,\s*[^.]*lighting\.\s*",
    re.IGNORECASE,
)

def strip_city_clause(text: str) -> str:
    return _CITY_CLAUSE_RE.sub(" ", text).strip()

def run_qa(pil_image, question, max_new_tokens=80):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": question},
    ]}]
    text   = proc_qa.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc_qa(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=proc_qa.tokenizer.eos_token_id,
            return_dict_in_generate=False, output_logits=False)
    n = inputs["input_ids"].shape[1]
    return proc_qa.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


# ══ Loop principal ═══════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)
cap      = cv2.VideoCapture(VIDEO_IN)
fps      = cap.get(cv2.CAP_PROP_FPS)
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

PANEL_W = 680
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W + PANEL_W, H))

try:
    font    = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 18)
    font_sm = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 14)
    font_lg = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",    22)
except Exception:
    font = font_sm = font_lg = ImageFont.load_default()

STATE_STYLE = {
    WZState.OUTSIDE:     {"bg": (40,  40,  40), "fg": (180, 180, 180), "label": "OUTSIDE"},
    WZState.APPROACHING: {"bg": (80,  60,   0), "fg": (255, 200,  50), "label": "APPROACHING"},
    WZState.INSIDE:      {"bg": (80,  10,  10), "fg": (255,  80,  80), "label": "INSIDE"},
    WZState.EXITING:     {"bg": (10,  60,  10), "fg": (80,  230,  80), "label": "EXITING"},
}
BAR_COLORS = [(180, 180, 180), (255, 200, 50), (255, 80, 80), (80, 230, 80)]

print(f"Processing {total_fr} frames @ {fps:.0f}fps ({total_fr/fps:.0f}s)\n")

bf               = BayesFilter()
belief           = PRIOR.copy()
current_state    = bf.state
current_text     = "Analyzing..."
p_ego            = np.ones(3) / 3
p_traffic        = np.ones(4) / 4
p_active         = np.ones(2) / 2
frame_idx        = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    sec = frame_idx / fps
    pil = None

    if frame_idx % STATE_EVERY == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emission, p_ego, p_traffic, p_active = state_emission(pil)
        prev_state = bf.state
        belief = bf.update(emission)
        current_state = bf.state
        if current_state != prev_state:
            print(f"  t={sec:5.1f}s [TRANSITION] {prev_state.value.upper()} → {current_state.value.upper()}")
        estr = "  ".join(f"{n[:3]}={v:.2f}" for n, v in zip(EGO_SHORT, p_ego))
        bstr = "  ".join(f"{n[:3]}={v:.2f}" for n, v in zip(STATE_NAMES, belief))
        print(f"  t={sec:5.1f}s ego[{EGO_SHORT[p_ego.argmax()]:>11s}]: {estr}  belief: {bstr}")

    if frame_idx % DESC_EVERY == 0:
        if pil is None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_text = strip_city_clause(run_qa(pil, Q_DESC, max_new_tokens=80))

    frame_out = frame.copy()

    panel = Image.new("RGB", (PANEL_W, H), color=(15, 15, 15))
    draw  = ImageDraw.Draw(panel)

    draw.rectangle([0, 0, PANEL_W, 50], fill=(20, 40, 70))
    draw.text((10,  8), "Alpamayo 2B v12 — Unseen Video", font=font, fill=(100, 200, 255))
    draw.text((10, 28), f"t={sec:.1f}s  frame {frame_idx}", font=font_sm, fill=(120, 160, 200))

    style = STATE_STYLE[current_state]
    draw.rectangle([0, 55, PANEL_W, 105], fill=style["bg"])
    draw.text((10, 58), "WORK ZONE STATUS:", font=font_sm, fill=(200, 200, 200))
    draw.text((10, 74), style["label"], font=font_lg, fill=style["fg"])
    draw.text((PANEL_W - 150, 74), f"P={belief.max():.2f}", font=font, fill=style["fg"])

    y = 115
    draw.text((10, y), "STATE BELIEF (Bayes filter):", font=font_sm, fill=(180, 220, 255))
    y += 18
    bar_max = PANEL_W - 190
    for i, (name, p) in enumerate(zip(STATE_NAMES, belief)):
        draw.text((10, y), f"{name:<11s}", font=font_sm, fill=(200, 200, 200))
        draw.rectangle([120, y + 2, 120 + int(bar_max * p), y + 12], fill=BAR_COLORS[i])
        draw.text((130 + int(bar_max * p), y), f"{p:.2f}", font=font_sm, fill=(160, 160, 160))
        y += 17
    y += 6

    draw.text((10, y), "EGO SENSOR (Stage-5 learned):", font=font_sm, fill=(180, 220, 255))
    y += 17
    ei = int(p_ego.argmax())
    draw.text((10, y), "  ".join(f"{n[:3]}={v:.2f}" for n, v in zip(EGO_SHORT, p_ego)),
              font=font_sm, fill=(140, 255, 180))
    y += 18
    ti, ai = int(p_traffic.argmax()), int(p_active.argmax())
    draw.text((10, y), f"scene: {TRAFFIC_SHORT[ti]} ({p_traffic[ti]:.2f}) | "
                       f"{ACTIVE_SHORT[ai]} ({p_active[ai]:.2f})",
              font=font_sm, fill=(200, 200, 180))
    y += 20

    draw.text((10, y), "DESCRIPTION:", font=font_sm, fill=(255, 200, 80))
    y += 16
    for line in textwrap.wrap(current_text, width=46)[:9]:
        draw.text((10, y), line, font=font_sm, fill=(230, 230, 200))
        y += 15
        if y > H - 25:
            break

    progress = frame_idx / max(total_fr, 1)
    draw.rectangle([0, H - 20, PANEL_W, H],                 fill=(20, 20, 20))
    draw.rectangle([0, H - 20, int(PANEL_W * progress), H], fill=(50, 100, 180))
    draw.text((5, H - 18), f"{sec:.1f}s / {total_fr/fps:.0f}s", font=font_sm, fill=(200, 200, 200))

    panel_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    out_vid.write(np.hstack([frame_out, panel_bgr]))
    frame_idx += 1

cap.release()
out_vid.release()
print(f"\nDone! Saved: {VIDEO_OUT}  ({frame_idx} frames)")
