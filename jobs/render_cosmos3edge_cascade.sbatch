#!/bin/bash
#SBATCH --job-name=render_cascade
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/render_cascade_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/render_cascade_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test

source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate
python $BASE/mi3lab-workzone-vla/inference/render_cosmos3edge_cascade.py

echo
echo "[reencode] h264 + faststart + bitrate adequado..."
ffmpeg -y -v error -i "$DELIVER/_cascade_raw.mp4" \
    -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p -crf 20 -preset medium \
    -movflags +faststart \
    "$DELIVER/boston_CASCADE_cosmos3edge.mp4"
rm -f "$DELIVER/_cascade_raw.mp4"

echo
echo "[ok]"
ls -la "$DELIVER/boston_CASCADE_cosmos3edge.mp4"
ffprobe -v error -show_entries format=duration,bit_rate -show_entries stream=codec_name,profile,width,height -of default=noprint_wrappers=1 "$DELIVER/boston_CASCADE_cosmos3edge.mp4"

echo
echo "[conferencia] extraindo frames de amostra..."
ffmpeg -y -v error -ss 1 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t1.png"
ffmpeg -y -v error -ss 6 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t6.png"
ffmpeg -y -v error -ss 12 -i "$DELIVER/boston_CASCADE_cosmos3edge.mp4" -frames:v 1 "$DELIVER/_check_frames/cascade_t12.png"
echo "frames de conferencia salvos"
