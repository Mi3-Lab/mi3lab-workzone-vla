# Mapa real de precisão

| Componente | Formato de implantação | Precisão |
|---|---|---|
| Reasoner | ONNX + TensorRT Edge-LLM | W4A16 AWQ INT4; `lm_head` e KV em FP16 |
| Encoder visual | ONNX/TensorRT | FP16 |
| MoT inverse/forward/policy/I2V | ONNX Runtime TensorRT EP | FP16 |
| Wan VAE encoder/decoder | ONNX Runtime TensorRT EP | FP16 |
| Scheduler, CFG, UniPC e mídia | host | FP32/inteiros conforme a operação |
| Referência nativa | safetensors | BF16 |
| Referência TorchAO | state dict | INT4 weight-only nos `nn.Linear` da torre de entendimento; embeddings, normas, `*_moe_gen*` e demais partes BF16 |

Não existe nesta pasta um falso “INT4 completo”. Convoluções, normalizações,
softmax, VAE e saídas sensíveis permanecem FP16 até que calibração INT8/INT4 e
paridade numérica sejam demonstradas no Orin.
