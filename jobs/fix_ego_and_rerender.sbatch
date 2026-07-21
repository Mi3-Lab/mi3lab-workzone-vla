#!/bin/bash
#SBATCH --job-name=fix_ego_cascade
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/fix_ego_cascade_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/fix_ego_cascade_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test

export HF_HOME=$BASE/.hf_cache
export UV_CACHE_DIR=$BASE/.uv_cache
export COSMOS_TRAINING=false

cd $CF
echo "[setup] uv sync --group=cu128..."
~/.local/bin/uv sync --group=cu128 2>&1 | tail -15
~/.local/bin/uv pip install boto3 qwen-vl-utils 2>&1 | tail -5

echo
echo "[run] re-rodando so' o EGO (25 amostras, prompt mais direto)..."
.venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/workzone/frames/f*_EGO.json" \
    -o outputs/workzone_cascade_frames_ego_fix \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 \
    --no-guardrails

echo
echo "[merge] atualizando so' o campo EGO no JSON existente (GATE/DESC/SIGN ficam como estavam)..."
python3 -c "
import json, glob, re

CF = '$CF'
DELIVER = '$DELIVER'

raw = json.load(open(f'{DELIVER}/cosmos3edge_cascade_frames_raw.json'))
pat = re.compile(r'f(\d{3})_EGO')
n = 0
for d in glob.glob(f'{CF}/outputs/workzone_cascade_frames_ego_fix/*/sample_outputs.json'):
    name = d.split('/')[-2]
    m = pat.match(name)
    if not m:
        continue
    frame_idx = str(int(m.group(1)))
    data = json.load(open(d))
    raw[frame_idx]['EGO'] = data['outputs'][0]['content']['reasoner_text']
    n += 1
json.dump(raw, open(f'{DELIVER}/cosmos3edge_cascade_frames_raw.json', 'w'), indent=2, ensure_ascii=False)
print(f'{n} respostas de EGO atualizadas')
"

echo
echo "[render] refazendo o video com o EGO corrigido..."
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python $BASE/mi3lab-workzone-vla/inference/render_cosmos3edge_cascade.py
deactivate

echo
echo "[reencode] h264 + faststart..."
ffmpeg -y -v error -i "$DELIVER/_cascade_raw.mp4" \
    -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p -crf 20 -preset medium \
    -movflags +faststart \
    "$DELIVER/boston_CASCADE_cosmos3edge.mp4"
rm -f "$DELIVER/_cascade_raw.mp4"

echo
echo "[ok]"
ls -la "$DELIVER/boston_CASCADE_cosmos3edge.mp4"

echo
echo "[conferencia] extraindo frames..."
mkdir -p "$DELIVER/_check_frames"
ffmpeg -y -v error -ss 1 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t1.png"
ffmpeg -y -v error -ss 6 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t6.png"
ffmpeg -y -v error -ss 10.5 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t10.png"
ffmpeg -y -v error -ss 14 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t14.png"
echo "salvo"
