# Cosmos3-Edge — pacote operacional para Jetson

Gerado em 2026-07-26. Symlinks já resolvidos: este diretório é autocontido.

## Comece por aqui
`docs/DEPLOY_ORIN_GUIA.md` — passo a passo, comandos prontos, armadilhas de
ambiente e o estado de cada artefato.

## Configuração recomendada (medida no boston.mp4, A100)
`--resolution-tier 256 --steps 8` → **1,99 s/tick**, contra 15,06 s do pacote
original. Orçamento: 6,0 s (60 ações a 10 Hz = 6 s de controle por inferência,
o modelo usa *action chunking*; NÃO são 100 ms por tick).

## O que tem aqui
    onnx/vae-chunk-256/    VAE encoder por chunk (192x320) — 3 grafos
    onnx/mot-inverse-256/  transformer inverse dynamics (tier 256)
    onnx/reasoner-int4/    reasoner INT4 AWQ
    onnx/visual-fp16/      torre visual
    runtime/               scripts de execução
    deploy/                build dos engines NO Jetson (obrigatório: engines
                           TensorRT são travados ao hardware)
    export/                scripts para reexportar em outra configuração
    third_party/           fork Cosmos3 do diffusers (obrigatório no PYTHONPATH)
    docs/                  guia de deploy + relatório da investigação

## O que NÃO veio
`weights/` (17 GB, referência BF16/INT4 — não é necessário para o caminho ONNX)
e os grafos antigos em 480x832, substituídos pelos de tier 256.

## Primeiro comando no Jetson
    ./deploy/build_vae_chunk_on_jetson.sh onnx/vae-chunk-256 engines/vae
