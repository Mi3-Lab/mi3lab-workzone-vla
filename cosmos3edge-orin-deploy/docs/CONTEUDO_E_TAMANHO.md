# O que ocupa espaço na entrega

Há duas entregas porque pesos executáveis ONNX e checkpoints de referência têm
finalidades diferentes.

## ZIP operacional do Orin

O runtime ocupa exatamente **10.759.761.656 bytes (10,0208 GiB)**,
incluindo as evidências finais:

| Parte | Tamanho aparente | Motivo |
|---|---:|---|
| MoT FP16 compartilhado | 5,94 GiB | Um conjunto atende inverse AV, policy AV, forward/policy UMI e os dois ramos I2V |
| Reasoner W4A16 INT4 | 1,70 GiB | Grafo, 168 plugins GEMM INT4 e pesos externos |
| VAE FP16 compartilhado | 1,37 GiB | Encoders/decoders AV, UMI e I2V compartilham pesos |
| Visual SigLIP2 FP16 | 0,92 GiB | Encoder visual e projector do Reasoner |
| Configuração, código e fixtures | ~0,05 GiB | Diffusers fixado, tokenizer/config, scripts e testes |

Esses aproximadamente 10 GiB **já são os pesos usados pela inferência ONNX**;
um pacote de 17 MiB conteria apenas grafos/configurações e não poderia executar
o modelo. O armazenamento lógico dos seis MoT seria 37,39 GB se cada grafo
duplicasse os pesos; a deduplicação reduz essa parte a 6,38 GB: economia de
82,93% e razão de compartilhamento de 5,86x.

Ele ainda pode ser um pouco maior que o checkpoint BF16 de 8,57 GiB porque a
entrega guarda simultaneamente o MoT/VAE/visual FP16 e uma cópia quantizada
separada do reasoner (1,70 GiB), além dos grafos de interface de cada perfil.
Isso não significa que o checkpoint inteiro foi duplicado: os grandes blobs
MoT e VAE são compartilhados por conteúdo entre as modalidades.

### Comparação com o checkpoint original

O checkpoint BF16 original tem `9.198.075.487` bytes (`8,566 GiB`). O runtime
operacional completo fica aproximadamente `16,8%` maior (`1,168x`) porque
também inclui reasoner INT4, visual, VAE, interfaces ONNX e código executável.
Essa relação não deve ser chamada de taxa de compressão do MoT: são escopos
diferentes. A taxa comparável dentro dos seis perfis MoT é a deduplicação de
82,93% indicada acima.

## ZIP opcional de pesos de referência

O conjunto opcional ocupa exatamente **18.105.005.161 bytes (16,8616 GiB)**:

| Parte | Tamanho aparente | Uso |
|---|---:|---|
| Checkpoint BF16 completo | 8,57 GiB | Comparação nativa e reexportação |
| TorchAO INT4 salvo | 3,96 GiB | Comparação PyTorch INT4 da language model |
| Wan VAE `.pth` nativo | 2,63 GiB | Referência/reexportação VAE |
| Fonte AWQ do Reasoner | 1,71 GiB | Reconstrução/reexportação do ONNX W4A16 |

O Orin não precisa desse segundo ZIP para executar os ONNX. Ele existe porque
foi solicitado preservar também os pesos BF16 e INT4; quem preferir pode obter
as referências novamente da fonte oficial.

Engines TensorRT não estão incluídos. Eles são dependentes da versão do
TensorRT/JetPack e da GPU e devem ser construídos no próprio AGX Orin.
