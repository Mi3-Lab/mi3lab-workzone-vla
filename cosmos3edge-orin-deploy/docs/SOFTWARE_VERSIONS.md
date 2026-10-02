# Versões e fronteira de portabilidade

## Ambiente usado nos testes A100

- Diffusers Cosmos3: commit `86e6dac5360703ddf09fe250db50be667eb93662`;
- TensorRT Edge-LLM: v0.9.0, base
  `1ac0f2b99642045125e1c5ac7b109434ba3b36c7`, com o patch desta pasta;
- PyTorch: 2.10.0+cu128;
- ONNX: 1.22.0;
- ONNX Runtime GPU: 1.27.0;
- Transformers: 4.57.6;
- Tokenizers: 0.22.2;
- TorchAO do benchmark opcional: 0.17.0.

TorchAO 0.17 emitiu aviso de que suas extensões C++ preferem PyTorch 2.11 ou
superior. O benchmark A100 em PyTorch 2.10 ainda executou os tensores
`Int4TilePackedTo4dTensor` pelo caminho disponível, mas ficou mais lento que
BF16. Esse resultado não deve ser extrapolado para o ONNX W4A16 nem para o
Jetson; use uma combinação PyTorch/TorchAO compatível com o JetPack e repita a
medição.

## Jetson AGX Orin

PyTorch, CUDA, TensorRT e ONNX Runtime precisam ser versões aarch64 compatíveis
com o JetPack instalado. Não copie wheels x86_64 nem engines serializados no
A100. Antes de executar, confirme que `onnxruntime.get_available_providers()`
contém `TensorrtExecutionProvider` e `CUDAExecutionProvider`.

O pacote fixa código, ONNX e pesos, mas não inventa uma combinação universal de
JetPack. A validação física final deve registrar versão do JetPack/TensorRT,
clocks, potência, temperatura, memória e latências warm p50/p95 no dispositivo.
