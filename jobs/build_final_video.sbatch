#!/bin/bash
#SBATCH --job-name=build_final_vid
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/build_final_vid_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/build_final_vid_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
DELIVER=$BASE/outputs/cosmos3edge_trajectory_test
VIDEO=$BASE/videos/boston.mp4

source $BASE/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/activate

echo "[1/2] gerando legendas + painel de trajetoria..."
python $BASE/mi3lab-workzone-vla/diagnostics/build_final_video_assets.py "$DELIVER"

echo
echo "[2/2] compondo video final (boston.mp4 inteiro, 15s) com ffmpeg..."
# painel de trajetoria: overlay fixo no canto superior direito, do inicio ao fim
# legendas: 5 imagens, cada uma visivel por 1/5 da duracao do video (ciclo
# sincronizado com a duracao real, nao um valor fixo chutado)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
SEG=$(python3 -c "print($DUR/5)")

ffmpeg -y -v error \
    -i "$VIDEO" \
    -i "$DELIVER/traj_panel.png" \
    -i "$DELIVER/caption_0.png" \
    -i "$DELIVER/caption_1.png" \
    -i "$DELIVER/caption_2.png" \
    -i "$DELIVER/caption_3.png" \
    -i "$DELIVER/caption_4.png" \
    -filter_complex "\
[0:v][1:v] overlay=W-w-10:10 [v1]; \
[v1][2:v] overlay=0:H-h:enable='between(t,0,$SEG)' [v2]; \
[v2][3:v] overlay=0:H-h:enable='between(t,$SEG,2*$SEG)' [v3]; \
[v3][4:v] overlay=0:H-h:enable='between(t,2*$SEG,3*$SEG)' [v4]; \
[v4][5:v] overlay=0:H-h:enable='between(t,3*$SEG,4*$SEG)' [v5]; \
[v5][6:v] overlay=0:H-h:enable='between(t,4*$SEG,5*$SEG)' [vout]" \
    -map "[vout]" -map 0:a? \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"

echo
echo "[ok] video final:"
ls -la "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 "$DELIVER/boston_FULL_com_texto_e_trajetoria.mp4"
