"""PAIDataset com MULTIPLAS janelas temporais por clipe (caminho A).

Descoberta que motiva isto: todos os nossos treinos usaram use_default_keyframe
=True, ou seja t0 FIXO em 5.1s (pai.py:29, DEFAULT_T0_US) — extraimos UMA janela
de 8s (1.6s historia + 6.4s futuro) de cada clipe de ~20s. O clipe comporta ~8+
janelas com stride de 1.5s.

Com 3644 clipes x 8 janelas ~= 29k amostras, um salto de 8x SEM baixar um byte.
Contexto: a NVIDIA reporta ganhos de dados continuando ate 2M exemplos; nossos
7 experimentos com 3.6k amostras empataram em minADE 4.57-5.12. Este e' o teste
mais barato da hipotese de que o gap e' escala de dados.

Ressalva honesta (vale para interpretar o resultado): janelas do mesmo clipe sao
CORRELACIONADAS (mesma via, mesma cena, mesma iluminacao). A diversidade efetiva
e' menor que 8x — isto testa "mais amostras do mesmo mundo", nao "mundo maior".

Implementacao: o __getitem__ do pai le t0 de self.DEFAULT_T0_US quando
use_default_keyframe=True. Atribuir na instancia sombreia o atributo de classe,
e cada worker do DataLoader tem sua PROPRIA copia do dataset (processos), entao
nao ha corrida. Se uma janela cair fora do clipe (clipe curto), cai de volta no
t0 canonico de 5.1s, que e' valido para todos os clipes por construcao.
"""
from typing import Any

from alpamayo.data.pai import PAIDataset

# t0 em microssegundos: 8 janelas, stride 1.5s, cobrindo 2.0s..12.5s.
# 5.1s (o t0 canonico da NVIDIA) esta incluido — o conjunto original de treino
# e' um SUBCONJUNTO deste, o que mantem a comparacao com os baselines limpa.
DEFAULT_T0_LIST_US = [2_000_000, 3_500_000, 5_100_000, 6_600_000,
                      8_100_000, 9_600_000, 11_100_000, 12_600_000]
_CANONICAL_T0_US = 5_100_000


class MultiWindowPAIDataset(PAIDataset):
    def __init__(self, *args: Any, t0_list_us: list[int] | None = None, **kwargs: Any):
        kwargs["use_default_keyframe"] = True   # t0 vem SEMPRE de DEFAULT_T0_US
        super().__init__(*args, **kwargs)
        self.t0_list_us = list(t0_list_us) if t0_list_us else list(DEFAULT_T0_LIST_US)
        print(f"[MultiWindowPAIDataset] {super().__len__()} clipes x "
              f"{len(self.t0_list_us)} janelas = {len(self)} amostras")

    def __len__(self) -> int:
        return super().__len__() * len(self.t0_list_us)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        clip_idx, w = divmod(idx, len(self.t0_list_us))
        self.DEFAULT_T0_US = self.t0_list_us[w]        # sombreia o atributo de classe
        try:
            return super().__getitem__(clip_idx)
        except Exception:
            # janela fora do clipe (clipe mais curto que o esperado):
            # usa o t0 canonico, valido pra todo clipe. Gera leve duplicacao
            # do sample padrao, que e' ruido aceitavel.
            self.DEFAULT_T0_US = _CANONICAL_T0_US
            try:
                return super().__getitem__(clip_idx)
            except Exception:
                return None
