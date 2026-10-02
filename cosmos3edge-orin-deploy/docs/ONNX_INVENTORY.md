# Inventário ONNX validado

Os 14 grafos abaixo passaram `onnx.checker.check_model` já dentro da árvore
final, com os arquivos de external data materializados e sem links simbólicos
ou hardlinks.

| Componente | Grafo | Precisão | Nós / observação |
|---|---|---|---|
| Reasoner | `onnx/reasoner-int4/model.onnx` | W4A16 AWQ | 1.024 nós; 168 `Int4GroupwiseGemmPlugin` |
| Visual | `onnx/visual-fp16/model.onnx` | FP16 | 1.455 nós; perfil de uma imagem Boston |
| MoT inverse AV | `onnx/mot-fp16-shared/onnx/inverse-av61-fp16.onnx` | FP16 | 14.469 nós |
| MoT policy AV | `onnx/mot-fp16-shared/onnx/policy-av60-fp16.onnx` | FP16 | 14.593 nós; rollout experimental do modelo-base, 60 ações 9D + 61 frames; não validado para controle |
| MoT forward UMI | `onnx/mot-fp16-shared/onnx/forward-umi16-fp16.onnx` | FP16 | 14.444 nós |
| MoT policy UMI | `onnx/mot-fp16-shared/onnx/policy-umi16-fp16.onnx` | FP16 | 14.532 nós; vídeo e ação conjuntos |
| MoT I2V condicional | `onnx/mot-fp16-shared/onnx/i2v-121-cond-fp16.onnx` | FP16 | 14.406 nós |
| MoT I2V incondicional | `onnx/mot-fp16-shared/onnx/i2v-121-uncond-fp16.onnx` | FP16 | 14.406 nós |
| VAE encoder AV61 | `onnx/vae-fp16-shared/onnx/encoder-av61-fp16.onnx` | FP16 | 44.366 nós |
| VAE decoder AV61 | `onnx/vae-fp16-shared/onnx/decoder-av61-fp16.onnx` | FP16 | 50.792 nós |
| VAE encoder UMI17 | `onnx/vae-fp16-shared/onnx/encoder-umi17-fp16.onnx` | FP16 | 13.170 nós |
| VAE decoder UMI17 | `onnx/vae-fp16-shared/onnx/decoder-umi17-fp16.onnx` | FP16 | 14.800 nós |
| VAE encoder I2V121 | `onnx/vae-fp16-shared/onnx/encoder-i2v121-fp16.onnx` | FP16 | 86.906 nós |
| VAE decoder I2V121 | `onnx/vae-fp16-shared/onnx/decoder-i2v121-fp16.onnx` | FP16 | 99.872 nós |

Scheduler UniPC, CFG, criação de masks/position IDs, processamento de mídia,
KV-cache e renderização ficam no runtime host. Eles são lógica de controle sem
pesos e não devem ser apresentados como mais um “ONNX gigante”.
