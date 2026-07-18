"""Baixa o grupo 'videos' do ROADWork e extrai SO os 489 arquivos que faltam
(cruzando com workzone_annotations_full.json), em vez do dataset inteiro.

Por que nao usar o download_roadwork.py padrao: ele faz z.extractall() no zip
INTEIRO (milhares de videos, dezenas de cidades) quando so precisamos dos 520
anotados -- e so 31 desses ja estao em disco. Extrair tudo desperdicaria dezenas
de GB em videos que nunca vamos usar.

Pico de disco: ~87GB (partes brutas) + ~87GB (zip montado) = ~174GB, transitorio.
Estado estavel apos limpar os intermediarios: so os poucos GB dos ~489 videos
extraidos. Precisa do disco livre para o pico -- verificar antes de rodar.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data/roadwork"))
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")

from huggingface_hub import hf_hub_download

BASE = Path(__file__).resolve().parents[2] / "data/roadwork"
RAW = BASE / "raw"
DEST = BASE / "videos"
ANNOT = Path(__file__).resolve().parents[2] / "workzone_annotations_full.json"

PARTS = [
    "videos_compressed.zip", "videos_compressed.z01", "videos_compressed.z02",
    "videos_compressed.z03", "videos_compressed.z04", "videos_compressed.z05",
    "videos_compressed.z06", "videos_compressed.z07",
]
REPO_ID = "anuragxel/roadwork-dataset"

RAW.mkdir(parents=True, exist_ok=True)
DEST.mkdir(parents=True, exist_ok=True)

needed = set(json.load(open(ANNOT)).keys())
have = {p.name for p in DEST.glob("*.mp4")}
missing = needed - have
print(f"[0] anotados: {len(needed)} | ja em disco: {len(have & needed)} | faltam: {len(missing)}")
if not missing:
    print("nada a baixar.")
    sys.exit(0)

print("\n[1] baixando as partes do zip multi-parte (~87GB)...")
for f in PARTS:
    local = RAW / f
    if local.exists():
        print(f"  [skip] {f}")
        continue
    print(f"  [download] {f}")
    hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f,
                    local_dir=str(RAW), local_dir_use_symlinks=False)

assembled = RAW / "videos_assembled.zip"
if not assembled.exists():
    print("\n[2] montando o zip multi-parte ('zip -FF')...")
    subprocess.run(["zip", "-FF", str(RAW / "videos_compressed.zip"),
                    "--out", str(assembled)], check=True)

    print("\n[3] limpando as partes brutas (~87GB liberados)...")
    for f in PARTS:
        (RAW / f).unlink(missing_ok=True)
else:
    print("\n[2] zip ja montado, pulando.")

print(f"\n[4] extraindo so os {len(missing)} arquivos que faltam...")
extracted = 0
with zipfile.ZipFile(assembled, "r") as zf:
    names_in_zip = {os.path.basename(n): n for n in zf.namelist()}
    still_missing = []
    for name in sorted(missing):
        member = names_in_zip.get(name)
        if member is None:
            still_missing.append(name)
            continue
        with zf.open(member) as src, open(DEST / name, "wb") as dst:
            dst.write(src.read())
        extracted += 1
        if extracted % 50 == 0:
            print(f"  [{extracted}/{len(missing)}] extraidos", flush=True)

print(f"\n[5] extraidos: {extracted} | nao encontrados no zip: {len(still_missing)}")
if still_missing:
    print("  (nao encontrados, primeiros 10):", still_missing[:10])

print("\n[6] limpando o zip montado (~87GB liberados)...")
assembled.unlink(missing_ok=True)

have_now = {p.name for p in DEST.glob("*.mp4")}
print(f"\n=== FINAL: {len(have_now & needed)}/{len(needed)} videos anotados em disco ===")
