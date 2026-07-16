"""Teste de fumaca: carregar nossos 39 chunks locais de PhysicalAI-AV via
PAIDataset (versao sem nav, GT direto de egomotion) e inspecionar 1 amostra.
"""
import os, sys, glob, re

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"
sys.path.insert(0, f"{BASE}/alpamayo-recipes/recipes/alpamayo1_5_sft")
os.environ.update({
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": f"{BASE}/.hf_cache",
})

from alpamayo_r1.data.pai import PAIDataset

PAI_DIR = f"{BASE}/data/PhysicalAI-AV"

chunk_ids = sorted(
    int(re.search(r"chunk_(\d+)", os.path.basename(p)).group(1))
    for p in glob.glob(f"{PAI_DIR}/labels/egomotion/*.zip")
)
print(f"chunks locais disponiveis: {len(chunk_ids)}")
print(chunk_ids)

print("\n[1] Instanciando PAIDataset...")
ds = PAIDataset(
    local_dir=PAI_DIR,
    chunk_ids=chunk_ids,
    num_history_steps=16,
    num_future_steps=64,
    time_step=0.1,
    use_default_keyframe=True,
)
print(f"    OK -- {len(ds)} clipes no dataset")

print("\n[2] Carregando amostra 0...")
sample = ds[0]
print("    OK -- chaves:", list(sample.keys()))
for k, v in sample.items():
    shape = getattr(v, "shape", None)
    print(f"      {k}: shape={shape} dtype={getattr(v, 'dtype', type(v))}")

print("\n=== TESTE DE FUMACA DE DADOS CONCLUIDO ===")
