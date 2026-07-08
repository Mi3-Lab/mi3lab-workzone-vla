#!/bin/bash
#SBATCH --job-name=alp_s7_eval
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/s7_eval_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/slurm-logs/video/s7_eval_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
set -e

echo "=== [1/3] Export stage7 -> HF ==="
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
export PYTHONUNBUFFERED=1
python $BASE/mi3lab-workzone-vla/data_prep/export_stage7_hf.py
deactivate

echo "=== [2/3] Fix tie_word_embeddings=false no config exportado ==="
python3 - <<'EOF'
import json
p = "/data/wesleyferreiramaia/wokzone-alpamayo/models/workzone-2b-stage7-hf/config.json"
c = json.load(open(p))
changed = []
if c.get("tie_word_embeddings") is True:
    c["tie_word_embeddings"] = False; changed.append("top")
tc = c.get("text_config")
if isinstance(tc, dict) and tc.get("tie_word_embeddings") is True:
    tc["tie_word_embeddings"] = False; changed.append("text_config")
json.dump(c, open(p, "w"), indent=2)
print("tie_word_embeddings=false aplicado em:", changed or "nada (ja estava false)")
EOF

echo "=== [3/3] Avaliacao BF16 nos 3 benchmarks (466 frames x 6 perguntas) ==="
source $BASE/.venv-edgellm/bin/activate
python $BASE/mi3lab-workzone-vla/diagnostics/eval_stage7_bf16.py

echo "=== STAGE7 EXPORT+EVAL DONE ==="
