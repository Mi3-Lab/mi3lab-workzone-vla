"""TrainableAlpamayoR1 ajustado pra rodar sob DeepSpeed (necessario no cotrain).

O problema:
  PerWaypointActionInProjV2.forward faz upcast EXPLICITO pra fp32 (action_in_proj.py
  linhas 162-163: `x = x.float()`, `timesteps = timesteps.float()`) -- proposital,
  pras features de Fourier terem estabilidade numerica -- e logo em seguida joga
  isso num MLP de pesos bf16. Ou seja: o modelo da NVIDIA foi ESCRITO contando com
  a torch.autocast pra reconciliar fp32-de-entrada com peso-bf16.

  Sem DeepSpeed, o HF Trainer (bf16=True) liga a autocast e tudo funciona -- foi o
  caso dos treinos anteriores (deepspeed: null). Com DeepSpeed a autocast NAO e'
  aplicada (o DeepSpeed gerencia a precisao por conta propria), e o upcast interno
  vira "mat1 and mat2 must have the same dtype, but got Float and BFloat16".

A correcao:
  Reativar a autocast em volta do forward -- isto e', devolver a condicao que o
  codigo original assume -- em vez de sair castando tensores e brigando com o
  upcast interno (que existe por um motivo). Nao mexe no sft_alpamayo_r1.py nem no
  action_in_proj.py, que sao vendored da NVIDIA.
"""
from typing import Any

import torch

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import ReasoningVLAOutput


class CotrainAlpamayoR1(TrainableAlpamayoR1):
    def forward(self, *args: Any, **kwargs: Any) -> ReasoningVLAOutput:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return super().forward(*args, **kwargs)
