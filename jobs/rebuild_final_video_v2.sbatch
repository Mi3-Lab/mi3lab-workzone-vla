#!/bin/bash
#SBATCH --job-name=rebuild_vid2
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/rebuild_vid2_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/rebuild_vid2_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test
VIDEO=$BASE/videos/boston.mp4

mkdir -p "$DELIVER"

echo "[1/5] re-extraindo JSONs dos outputs brutos que sobreviveram (sem GPU)..."
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python3 -c "
import json

CF = '$CF'
DELIVER = '$DELIVER'

for mode, prefix in [('trajectory_test', 'inverse_dynamics'), ('policy_test', 'policy')]:
    samples = []
    for i in range(5):
        p = f'{CF}/outputs/workzone_5samples/{mode}_seed{i}/sample_outputs.json'
        d = json.load(open(p))
        samples.append(d['outputs'][0]['content']['action'])
    json.dump(samples, open(f'{DELIVER}/cosmos3edge_{prefix}_5samples.json', 'w'))
    print(f'{prefix}: {len(samples)} amostras re-extraidas')

qa = {}
for name in ['gate_test', 'ego_test', 'active_test', 'sign_test', 'desc_test']:
    p = f'{CF}/outputs/workzone_full_test_fresh/{name}/sample_outputs.json'
    d = json.load(open(p))
    qa[name] = {'prompt': d['args']['prompt'], 'answer': d['outputs'][0]['content']['reasoner_text']}
json.dump(qa, open(f'{DELIVER}/cosmos3edge_qa_fresh.json', 'w'), indent=2, ensure_ascii=False)
print('QA re-extraido')
"
deactivate

echo
echo "[2/5] re-rodando extracao do Alpamayo (unico dado que nao sobreviveu, GPU)..."
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python $BASE/mi3lab-workzone-vla/diagnostics/extract_traj_boston.py
deactivate

echo
echo "[3/5] gerando legendas (imagens PNG, texto nao vai no comando shell)..."
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python3 -c "
import json, sys, textwrap
sys.path.insert(0, '$BASE/mi3lab-workzone-vla/diagnostics')
from PIL import Image, ImageDraw, ImageFont

DELIVER = '$DELIVER'
CAPTION_W = 1862  # boston escalado p/ altura 800 (1422, fixo, ver ffmpeg abaixo) + painel de trajetoria (440)

qa = json.load(open(f'{DELIVER}/cosmos3edge_qa_fresh.json'))
order = ['gate_test', 'ego_test', 'active_test', 'sign_test', 'desc_test']
labels = {'gate_test': 'GATE', 'ego_test': 'EGO', 'active_test': 'ACTIVE', 'sign_test': 'SIGN', 'desc_test': 'DESC'}

try:
    font_q = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
    font_a = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except OSError:
    font_q = font_a = ImageFont.load_default()

for i, key in enumerate(order):
    prompt = qa[key]['prompt']
    answer = qa[key]['answer']
    answer_clean = answer.split('</think>')[-1].strip() or answer.strip()
    answer_short = textwrap.shorten(answer_clean.replace(chr(10), ' '), width=190, placeholder='...')

    bar_h = 110
    img = Image.new('RGB', (CAPTION_W, bar_h), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    draw.text((14, 10), f'[{labels[key]}] {prompt}', font=font_q, fill=(255, 205, 70))
    for j, line in enumerate(textwrap.wrap(answer_short, width=110)):
        draw.text((14, 40 + j * 21), line, font=font_a, fill=(235, 235, 235))
    img.save(f'{DELIVER}/caption_{i}.png')
    print(f'legenda {i} ({labels[key]}) ok')
"
deactivate

echo
echo "[4/5] renderizando trajetoria ANIMADA (video, nao imagem estatica)..."
source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
python $BASE/mi3lab-workzone-vla/diagnostics/render_animated_trajectory.py "$DELIVER" 15 "$DUR"
deactivate

ffmpeg -y -v error -framerate 15 -i "$DELIVER/_traj_anim_frames/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -preset medium \
    "$DELIVER/_traj_animated.mp4"
echo "video de trajetoria animada montado"

echo
echo "[5/5] compondo video final: boston (esquerda, tamanho normal, 1422x800) + trajetoria animada (direita, 440x800) + legendas (embaixo, fora do video)..."
# boston escalado pra altura 800 com largura FIXA 1422 (722x406 -> mesma razao
# de aspecto, arredondado pra par) -- fixo, nao "-2", pra bater exatamente com
# CAPTION_W=1862 (1422+440) usado na geracao das legendas. Legendas ficam numa
# faixa ABAIXO de tudo, nao cobrem a imagem do video.
ffmpeg -y -v error \
    -i "$VIDEO" \
    -i "$DELIVER/_traj_animated.mp4" \
    -i "$DELIVER/caption_0.png" -i "$DELIVER/caption_1.png" -i "$DELIVER/caption_2.png" \
    -i "$DELIVER/caption_3.png" -i "$DELIVER/caption_4.png" \
    -filter_complex "\
[0:v] scale=1422:800 [vid]; \
[vid][1:v] hstack=inputs=2 [row]; \
color=c=0x121212:s=1862x110:d=$DUR [capbg]; \
[capbg][2:v] overlay=0:0:enable='between(t,0,$DUR/5)' [c1]; \
[c1][3:v] overlay=0:0:enable='between(t,$DUR/5,2*$DUR/5)' [c2]; \
[c2][4:v] overlay=0:0:enable='between(t,2*$DUR/5,3*$DUR/5)' [c3]; \
[c3][5:v] overlay=0:0:enable='between(t,3*$DUR/5,4*$DUR/5)' [c4]; \
[c4][6:v] overlay=0:0:enable='between(t,4*$DUR/5,5*$DUR/5)' [capfinal]; \
[row][capfinal] vstack=inputs=2 [vout]" \
    -map "[vout]" -map 0:a? \
    -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p -crf 20 -preset medium \
    -movflags +faststart \
    "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"

echo
echo "[ok] video final:"
ls -la "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"
ffprobe -v error -show_entries format=duration,bit_rate -show_entries stream=codec_name,profile,width,height -of default=noprint_wrappers=1 "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"

echo
echo "[extra] extraindo 3 frames de amostra pra conferencia visual..."
mkdir -p "$DELIVER/_check_frames"
ffmpeg -y -v error -ss 1 -i "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4" -frames:v 1 "$DELIVER/_check_frames/frame_t1.png"
ffmpeg -y -v error -ss 5 -i "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4" -frames:v 1 "$DELIVER/_check_frames/frame_t5.png"
ffmpeg -y -v error -ss 10 -i "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4" -frames:v 1 "$DELIVER/_check_frames/frame_t10.png"
echo "frames de conferencia salvos em $DELIVER/_check_frames/"

rm -rf "$DELIVER/_traj_anim_frames"
echo "[limpeza] frames intermediarios removidos"
