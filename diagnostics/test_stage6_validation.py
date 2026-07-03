"""Validação do Stage 6 (retenção de direção geral + especialização work zone).

Repete o teste que expôs o colapso original (job 170550: 2B Stage-5 respondia
"OUTLUS NO. NO." em cenas de BDD100k) contra o checkpoint do Stage 6, que
treinou com dado de self-distillation do Cosmos-2B base + retenção ego-state/
sign. Duas perguntas:

  1) O raciocínio de direção geral voltou? (comparar com Cosmos-2B base)
  2) As habilidades de work zone (ego-state, leitura de placa) sobreviveram
     ao Stage 6, ou foram diluídas pelo dado de retenção geral?
"""
import os, sys, glob, random
import torch
import numpy as np
from PIL import Image

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, os.path.join(BASE, "alpamayo1.5", "src"))
sys.path.insert(0, os.path.join(BASE, "alpamayo-recipes/recipes/alpamayo1_5_sft"))
os.environ.update({
    "WANDB_DISABLED": "true", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo1_5_sft.models.kd_model import build_student_model
from alpamayo1_5 import helper as alp_helper
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoModelForImageTextToText

A1_BASE   = f"{BASE}/models/Alpamayo-1.5-10B-A1-format"
COSMOS_2B = f"{BASE}/models/Cosmos-Reason2-2B"
CKPT      = f"{BASE}/checkpoints/sft_stage6_general/checkpoint-2194"

Q_EGO = ("Regarding the road work zone, what is this vehicle's current position status: "
         "OUTSIDE, APPROACHING, or INSIDE?")
Q_SIGN = "Read the text on the temporary traffic control signs in this scene."

QUESTIONS = [
    "Describe this driving scene.",
    "What should the ego vehicle do next, and why?",
    "Are there any hazards ahead? Answer briefly.",
    "Is it safe to change to the left lane right now?",
]

# 6 imagens BDD100k (direção normal, sem work zone)
random.seed(3)
bdd = sorted(glob.glob(f"{BASE}/data/roadwork/bdd100k/*.jpg"))
frames = random.sample(bdd, 6)


def make_generate(model_vlm, proc):
    @torch.no_grad()
    def gen(pil, question, max_new_tokens=70):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text": question},
        ]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[pil], return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model_vlm.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=proc.tokenizer.eos_token_id,
                return_dict_in_generate=False, output_logits=False)
        n = inputs["input_ids"].shape[1]
        return proc.tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()
    return gen


# ── A) nosso 2B Stage-6 ────────────────────────────────────────────────────────
print("[LOAD A] 2B Stage-6 (nosso)...")
model = build_student_model(alpamayo_base_path=A1_BASE, student_vlm_path=COSMOS_2B)
sd = {}
for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
    sd.update(load_file(f, device="cpu"))
if not sd:
    sd = load_file(os.path.join(CKPT, "model.safetensors"), device="cpu")
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).cuda().eval()
proc = AutoProcessor.from_pretrained(COSMOS_2B, trust_remote_code=True)
gen_ours = make_generate(model.vlm, proc)

print("\n" + "=" * 76)
print("A) NOSSO 2B Stage-6 em cenas de direção normal (BDD100k)")
print("=" * 76)
for path in frames[:3]:
    pil = Image.open(path).convert("RGB")
    print(f"\n[{os.path.basename(path)}]")
    for q in QUESTIONS:
        print(f"  Q: {q}")
        print(f"  A: {gen_ours(pil, q)[:130]}")

# ── Trajetória do nosso 2B em estrada normal ──────────────────────────────────
print("\n" + "=" * 76)
print("A2) Trajetória do 2B em estrada normal (herdada do KD do 10B AIAV)")
print("=" * 76)
proc_traj = alp_helper.get_processor(model.tokenizer)
NUM_HIST = 16
ego_xyz = torch.zeros(1, 1, NUM_HIST, 3, dtype=torch.float32, device="cuda")
ego_xyz[0, 0, :, 0] = torch.arange(NUM_HIST, dtype=torch.float32).cuda() * 0.1 * 8.0
ego_rot = (torch.eye(3, dtype=torch.float32, device="cuda")
           .view(1, 1, 1, 3, 3).expand(1, 1, NUM_HIST, 3, 3).contiguous())
for path in frames[:4]:
    pil = Image.open(path).convert("RGB")
    frame_t = torch.from_numpy(np.array(pil).transpose(2, 0, 1)).unsqueeze(0)
    messages = alp_helper.create_message(frames=frame_t, camera_indices=None)
    inputs = proc_traj.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = model.sample_trajectories_from_data(
                data={"tokenized_data": inputs,
                      "ego_history_xyz": ego_xyz.clone(),
                      "ego_history_rot": ego_rot.clone()},
                num_traj_samples=1, max_generation_length=200)
        xyz = pred_xyz[0, 0, 0, :, :].cpu().float().numpy()
        x_end, y_end = xyz[-1, 0], xyz[-1, 1]
        lateral_ratio = abs(y_end) / max(abs(x_end), 0.1)
        verdict = "OK forward" if (x_end >= 3 and lateral_ratio <= 1.0) else "RUIM (lateral/curta)"
        print(f"  [{os.path.basename(path)[:40]}] fim=({x_end:+.1f}, {y_end:+.1f})m  ratio_lat={lateral_ratio:.2f}  → {verdict}")
    except Exception as e:
        print(f"  [{os.path.basename(path)[:40]}] FALHOU: {str(e)[:80]}")

# ── A3) Retenção de work zone: ego-state (Stage 5) + leitura de placa (5.1) ──
print("\n" + "=" * 76)
print("A3) Retenção de work zone — sobreviveu ao Stage 6?")
print("=" * 76)
import cv2
for city, fi_list in [("boston", [90]), ("denver", [60]), ("seattle_unseen", [300])]:
    path = f"{BASE}/videos/{city}.mp4"
    cap = cv2.VideoCapture(path)
    for fi in fi_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ego = gen_ours(pil, Q_EGO, max_new_tokens=15)
        sign = gen_ours(pil, Q_SIGN, max_new_tokens=40)
        print(f"\n  [{city} frame {fi}]")
        print(f"    EGO : {ego[:80]}")
        print(f"    SIGN: {sign[:100]}")
    cap.release()

# liberar VRAM antes do modelo B
del model
torch.cuda.empty_cache()

# ── B) Cosmos-Reason2-2B base ─────────────────────────────────────────────────
print("\n[LOAD B] Cosmos-Reason2-2B base...")
base = AutoModelForImageTextToText.from_pretrained(
    COSMOS_2B, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
gen_base = make_generate(base, proc)

print("\n" + "=" * 76)
print("B) Cosmos-Reason2-2B BASE nas mesmas cenas")
print("=" * 76)
for path in frames[:3]:
    pil = Image.open(path).convert("RGB")
    print(f"\n[{os.path.basename(path)}]")
    for q in QUESTIONS:
        print(f"  Q: {q}")
        print(f"  A: {gen_base(pil, q)[:130]}")

print("\n=== DONE ===")
