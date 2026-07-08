"""Quantiza INT4 AWQ com calibração multimodal focada no estado ego-cêntrico.

Motivação: as calibrações anteriores (v1: 512, v2: 2000) usaram só TEXTO
genérico — sem imagem — o que nunca acontece de verdade em produção (toda
inferência real do v16 é imagem+pergunta). Além disso, nenhuma delas
priorizava especificamente o EGO_STATE (OUTSIDE/APPROACHING/INSIDE), que é
o "sensor principal" do filtro Bayesiano do v16.

Aqui: calibração multimodal (imagem real + pergunta real) usando o dataset
lingoqa_egostate (1399 exemplos reais, mesma pergunta exata do v16),
balanceado entre os 3 estados, misturado com uma fatia menor de outras
tarefas (ACTIVE/SIGN/DESC) pra não regredir nelas.

Segue a mesma receita curada (build_quant_config) validada como correta —
torre visual e KV-cache continuam FORA do INT4 (igual v1/v2).
"""
import os, sys, json, random
import pandas as pd
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})
sys.path.insert(0, f"{BASE}/TensorRT-Edge-LLM")

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from tensorrt_edgellm.quantization.quantization_configs import build_quant_config

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
OUTPUT_DIR = f"{BASE}/models/workzone-2b-stage6-int4awq-ego"

random.seed(44)

# ── Monta o conjunto de calibracao ────────────────────────────────────────────
print("[1] Montando conjunto de calibracao (multimodal, foco em EGO)...")

egostate_dir = f"{BASE}/data/roadwork/lingoqa_egostate"
df_ego = pd.read_parquet(f"{egostate_dir}/train.parquet")
df_ego["state"] = df_ego["answer"].str.extract(r"^(OUTSIDE|APPROACHING|INSIDE)")

N_PER_STATE = 110  # ~330 amostras multimodais, balanceadas
ego_rows = []
for state in ["OUTSIDE", "APPROACHING", "INSIDE"]:
    sub = df_ego[df_ego["state"] == state]
    idx = random.sample(range(len(sub)), min(N_PER_STATE, len(sub)))
    for i in idx:
        row = sub.iloc[i]
        img_path = os.path.normpath(os.path.join(egostate_dir, row["images"][0]))
        ego_rows.append(("mm", img_path, row["question"], None))

print(f"    EGO multimodal: {len(ego_rows)} amostras "
      f"({N_PER_STATE} por estado x 3)")

# Fatia menor de texto puro pra nao regredir nas outras perguntas
text_sources = {
    f"{BASE}/data/roadwork/lingoqa_combined/train.parquet": 80,
    f"{BASE}/data/roadwork/lingoqa_signtext/train.parquet": 60,
    f"{BASE}/data/roadwork/lingoqa_general/train.parquet": 60,
}
text_rows = []
for path, n in text_sources.items():
    df = pd.read_parquet(path)
    idx = random.sample(range(len(df)), min(n, len(df)))
    for i in idx:
        q, a = str(df.iloc[i]["question"]), str(df.iloc[i]["answer"])
        text_rows.append(("text", None, None, f"{q} {a}"))

print(f"    Texto puro (outras tarefas): {len(text_rows)} amostras")

calib_rows = ego_rows + text_rows
random.shuffle(calib_rows)
print(f"    TOTAL: {len(calib_rows)} amostras de calibracao")

# ── Carrega modelo (mesma receita do export real: fp16) ──────────────────────
print(f"\n[2] Carregando {MODEL_DIR} (fp16)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.float16, trust_remote_code=True,
).to("cuda")
model.to(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
processor = AutoProcessor.from_pretrained(
    MODEL_DIR, trust_remote_code=True,
    min_pixels=128 * 28 * 28, max_pixels=2048 * 32 * 32,
)

# ── Forward loop de calibracao (mistura multimodal + texto) ──────────────────
def calib_forward_loop(m):
    for i, (kind, img_path, question, text) in enumerate(calib_rows):
        with torch.no_grad():
            if kind == "mm":
                pil = Image.open(img_path).convert("RGB")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": question},
                ]}]
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to("cuda")
                m(**inputs)
            else:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                m(input_ids=enc["input_ids"].cuda())
        if (i + 1) % 100 == 0:
            print(f"    calibracao: {i + 1}/{len(calib_rows)}", flush=True)


print("\n[3] Quantizando (INT4 AWQ, receita curada build_quant_config)...")
quant_cfg = build_quant_config(quantization="int4_awq")
mtq.quantize(model, quant_cfg, forward_loop=calib_forward_loop)
mtq.print_quant_summary(model)

# ── Export (mesma logica de quantize_and_export) ─────────────────────────────
print(f"\n[4] Exportando para {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

import subprocess
size = subprocess.run(["du", "-sh", OUTPUT_DIR], capture_output=True, text=True).stdout.split()[0]
print(f"\n[5] Tamanho do export: {size}")
print("\n=== QUANTIZACAO EGO-FOCUSED CONCLUIDA ===")
