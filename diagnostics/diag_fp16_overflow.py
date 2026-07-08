"""Diagnostico da hipotese fp16-overflow para o bug do 1o token no TensorRT.

Contexto: a simulacao Python do INT4 (ModelOpt fake-quant) em BF16 e' limpa,
mas o engine TensorRT (que computa TUDO em fp16 - cast hardcoded no export
ONNX) gera token espurio ('.DATA' etc.) na 1a posicao pos-prefill em 92% dos
casos. Literatura (arXiv 2405.14428 "Activation Spikes in GLU-based LLMs",
2403.01241 IntactKV) mostra que modelos Qwen tem "massive activations" em
tokens delimitadores/sink que excedem o range do fp16 (max 65504).

Teste A: roda o checkpoint em fp16 PURO (sem quantizacao) nos frames
         quebrados. Se reproduzir o lixo no 1o token -> causa e' fp16,
         nao INT4.
Teste B: roda em bf16 com hooks que medem o max(|ativacao|) por camada.
         Se algum valor > 65504 -> overflow garantido em fp16.
"""
import os
import torch
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{BASE}/.hf_cache",
})
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_DIR = f"{BASE}/models/workzone-2b-stage6-hf"
Q_DESC = "Describe the work zone elements visible in this scene."
Q_EGO = ("Regarding the road work zone, what is this vehicle's current "
         "position status: OUTSIDE, APPROACHING, or INSIDE?")

FRAMES = [
    ("seattle_unseen", 90),
    ("boston", 36),
    ("chicago", 18),
    ("boston", 18),
]

FP16_MAX = 65504.0

proc = AutoProcessor.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def build_inputs(pil, question):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": question},
    ]}]
    return proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")


def load_frame(city, idx):
    img_path = f"{BASE}/models/workzone-2b-stage6-engines/video_frames/{city}_{idx:05d}.jpg"
    assert os.path.exists(img_path), img_path
    return Image.open(img_path).convert("RGB")


# ---------------------------------------------------------------- Teste A
print("=" * 70)
print("[TESTE A] Checkpoint em FP16 puro (sem quantizacao), greedy + sampled")
print("=" * 70)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.float16, attn_implementation="sdpa",
).cuda().eval()

for city, idx in FRAMES:
    pil = load_frame(city, idx)
    for qname, q, ntok in (("DESC", Q_DESC, 80), ("EGO", Q_EGO, 20)):
        inputs = build_inputs(pil, q)
        with torch.no_grad():
            out_g = model.generate(**inputs, max_new_tokens=ntok, do_sample=False)
            out_s = model.generate(**inputs, max_new_tokens=ntok, do_sample=True,
                                   temperature=0.4, top_p=0.9, top_k=40)
        n_in = inputs["input_ids"].shape[1]
        first_id_g = out_g[0, n_in].item()
        ans_g = tok.decode(out_g[0, n_in:], skip_special_tokens=True)
        ans_s = tok.decode(out_s[0, n_in:], skip_special_tokens=True)
        print(f"\n[{city}#{idx} {qname}] first_token_greedy={first_id_g} "
              f"({tok.decode([first_id_g])!r})")
        print(f"  FP16 greedy : {ans_g[:160]!r}")
        print(f"  FP16 sample : {ans_s[:160]!r}")

del model
torch.cuda.empty_cache()

# ---------------------------------------------------------------- Teste B
print()
print("=" * 70)
print("[TESTE B] BF16 instrumentado: max |ativacao| por camada (limite fp16 = 65504)")
print("=" * 70)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()

stats = {}  # name -> max abs seen

def make_hook(name):
    def hook(mod, args, output):
        t = output[0] if isinstance(output, tuple) else output
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return
        m = t.detach().abs().max().item()
        if m > stats.get(name, 0.0):
            stats[name] = m
    return hook

hooks = []
for name, mod in model.named_modules():
    # decoder layers inteiras + MLPs + projecoes (entrada/saida dos blocos
    # onde a literatura reporta os spikes: down_proj / hidden states)
    if name.endswith((".mlp", ".mlp.down_proj", ".self_attn.o_proj",
                      ".input_layernorm", ".post_attention_layernorm")) \
       or name.endswith("model.visual.merger"):
        hooks.append(mod.register_forward_hook(make_hook(name)))
print(f"{len(hooks)} hooks registrados")

for city, idx in FRAMES:
    pil = load_frame(city, idx)
    inputs = build_inputs(pil, Q_DESC)
    with torch.no_grad():
        model(**inputs)

for h in hooks:
    h.remove()

over = {k: v for k, v in stats.items() if v > FP16_MAX}
top = sorted(stats.items(), key=lambda kv: -kv[1])[:25]
print("\nTop-25 max |ativacao| por modulo (prefill, 4 frames DESC):")
for k, v in top:
    flag = "  <<< OVERFLOW fp16" if v > FP16_MAX else ""
    print(f"  {v:12.1f}  {k}{flag}")
print(f"\nModulos acima do limite fp16 (65504): {len(over)}/{len(stats)}")

print("\n=== DIAGNOSTICO CONCLUIDO ===")
