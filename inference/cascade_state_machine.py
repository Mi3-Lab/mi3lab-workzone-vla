"""Maquina de estados 'evidence-first' para deteccao de zona de obra.

Redesign motivado pelo negative-control de 2026-07-08 (BDD100k: 33% de
falso-positivo por frame; video comma2k19 sem obra: 99% do tempo preso em
APPROACHING). Tres defeitos corrigidos em relacao ao filtro Bayesiano do
v16 (render_trt_video.py):

1. PRESSUPOSICAO: a pergunta EGO ("Regarding the road work zone, ...")
   pressupoe que a obra existe -- leading question, modo de falha
   documentado em LVLMs (arXiv 2504.20468, 2310.05338, 2604.21911).
   -> Camada 1: pergunta de GATE neutra e binaria decide se ha evidencia;
      as perguntas de estado so sao consideradas com o gate ativo.

2. CANAL UNICO SEQUESTRA O ESTADO: fusao multiplicativa deixa o EGO
   (emissao peaked) dominar mesmo com ACTIVE/SIGN/TRAFFIC contra.
   -> Camada 3: transicao para dentro de obra exige gate ativo em >=K dos
      ultimos N frames E corroboracao de um segundo canal (DESC citando
      objeto especifico de obra, ou SIGN com keyword de obra).

3. HISTERESE DE SAIDA: sair exige gate inativo em >=K_EXIT dos ultimos N
   (nao basta um frame ruidoso).

O filtro Bayesiano original continua rodando DENTRO do modo ativo (entre
APPROACHING/INSIDE/EXITING), onde ele e' comprovadamente bom (validado nos
4 videos de obra) -- ele so perde o papel de guardiao da entrada.
"""
import re
from collections import deque
from enum import Enum

import numpy as np


class WZState(Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    INSIDE = "inside"
    EXITING = "exiting"


STATES = [WZState.OUTSIDE, WZState.APPROACHING, WZState.INSIDE, WZState.EXITING]
EGO_SHORT = ["OUTSIDE", "APPROACHING", "INSIDE"]

# Objetos ESPECIFICOS de obra (corroboracao). Generico demais ("road work"
# solto em frase) fica de fora de proposito: corroboracao exige objeto.
WZ_OBJECT_RE = re.compile(
    r"\bcone|\bbarrier|\bbarricade|\bdrum\b|tubular marker|TTC sign|"
    r"\bworker|\bflagger|work vehicle|construction (?:crew|equipment|site)|excavat",
    re.I)
SIGN_WZ_KEYWORDS = [
    "ROAD WORK", "WORK ZONE", "DETOUR", "LANE CLOSED", "ROAD CLOSED",
    "SIDEWALK CLOSED", "UTILITY WORK", "SHOULDER WORK", "FLAGGER",
    "BE PREPARED TO STOP", "LANE ENDS", "LANE SHIFT",
]

PEAK = 0.80
EGO_EMIT = np.array([
    [0.85, 0.12, 0.03], [0.10, 0.75, 0.15], [0.05, 0.55, 0.40], [0.50, 0.25, 0.25],
])
TRANS = np.array([
    [0.92, 0.08, 0.00, 0.00], [0.02, 0.88, 0.10, 0.00],
    [0.00, 0.00, 0.98, 0.02], [0.06, 0.00, 0.06, 0.88],
])
PRIOR = np.array([0.55, 0.20, 0.20, 0.05])


def peaked_dist(idx, n):
    p = np.full(n, (1.0 - PEAK) / max(n - 1, 1))
    p[idx] = PEAK
    return p


import re as _re

_GATE_COUNT_RE = _re.compile(r"^(\d+)\s+temporary", _re.I)


def parse_gate(text):
    """Resposta da pergunta de gate -> True/False/None.

    O modelo nem sempre responde 'Yes'/'No' literal -- frequentemente cai no
    template do SFT ('2 temporary signs(s) detected (...)'). Interpreta:
      - 'yes...'                          -> True
      - 'no...' / 'none...'               -> False
      - 'N temporary ... detected', N>0   -> True (N==0 -> False)
    """
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith(("no", "none")):
        return False
    m = _GATE_COUNT_RE.match(text.strip())
    if m:
        return int(m.group(1)) > 0
    return None


def classify_ego(answer):
    a = answer.lower()
    for i, lab in enumerate(EGO_SHORT):
        if lab.lower() in a:
            return i
    return None


def has_corroboration(desc_txt, sign_txt):
    if desc_txt and WZ_OBJECT_RE.search(desc_txt):
        return True
    if sign_txt and any(k in sign_txt.upper() for k in SIGN_WZ_KEYWORDS):
        return True
    return False


def sign_announces(sign_txt):
    """Placa de obra lida ('ROAD WORK AHEAD' etc.) = anuncio formal da zona."""
    return bool(sign_txt) and any(k in sign_txt.upper() for k in SIGN_WZ_KEYWORDS)


class CascadeStateMachine:
    """Camadas 1+3 do redesign. Janela N, entrada K_ENTER, saida K_EXIT.

    Sign latch (ajuste 2026-07-08, caso Boston): uma placa de obra lida e'
    o ANUNCIO da zona — o significado de APPROACHING e' exatamente o trecho
    entre o anuncio e a zona, mesmo sem elementos visiveis no meio (o
    modelo dizia 'nada visivel' no vao entre a placa e a obra e a cascade
    voltava/segurava OUTSIDE). Ao ler a placa: entra em APPROACHING
    imediatamente e segura por LATCH_N amostras, imune ao gate-clear.
    Seguro contra falso-positivo: 0/329 leituras falsas de placa nos
    controles negativos (BDD100k + comma2k19, stage7_1).

    Gate efetivo: SIGN/DESC citando obra no frame conta como gate Yes
    (corrige contradicao interna do modelo — ex: SIGN le 'ROAD WORK AHEAD'
    e o gate do mesmo frame diz No).
    """

    N = 5
    K_ENTER = 3
    K_EXIT = 4
    LATCH_N = 25  # ~15s na cadencia de 0.6s

    def __init__(self):
        self.state = WZState.OUTSIDE
        self.gate_window = deque(maxlen=self.N)
        self.corr_window = deque(maxlen=self.N)
        self.sign_latch = 0
        self.belief = PRIOR.copy()

    def _gate_active(self):
        return sum(1 for g in self.gate_window if g) >= self.K_ENTER

    def _gate_clear(self):
        return sum(1 for g in self.gate_window if not g) >= self.K_EXIT

    def _corroborated(self):
        return any(self.corr_window)

    def update(self, gate, ego_txt, desc_txt="", sign_txt=""):
        """Um passo (um frame amostrado). Retorna o estado exibido."""
        corr = has_corroboration(desc_txt, sign_txt)
        gate_eff = bool(gate) or corr
        self.gate_window.append(gate_eff)
        self.corr_window.append(corr)

        if sign_announces(sign_txt):
            self.sign_latch = self.LATCH_N
        elif self.sign_latch > 0:
            self.sign_latch -= 1

        if self.state == WZState.OUTSIDE:
            # Placa de obra lida = anuncio formal -> APPROACHING imediato.
            if self.sign_latch > 0:
                self.state = WZState.APPROACHING
                self.belief = np.array([0.10, 0.70, 0.15, 0.05])
                return self.state
            # Entrada padrao: gate ativo em >=K_ENTER dos ultimos N frames
            # E corroboracao de segundo canal em pelo menos 1 deles.
            if self._gate_active() and self._corroborated():
                self.state = WZState.APPROACHING
                self.belief = np.array([0.10, 0.70, 0.15, 0.05])
            return self.state

        # -------- modo ativo: filtro Bayesiano (so canal EGO) ----------
        ego_i = classify_ego(ego_txt)
        p_ego = peaked_dist(ego_i, 3) if ego_i is not None else np.ones(3) / 3
        emission = EGO_EMIT @ p_ego
        b = (self.belief @ TRANS) * emission
        self.belief = b / b.sum()

        # Saida global: evidencia sumiu de forma sustentada -> OUTSIDE.
        # O sign latch suprime a saida (anuncio ainda vigente).
        if self._gate_clear() and self.sign_latch == 0:
            self.state = WZState.OUTSIDE
            self.belief = PRIOR.copy()
            return self.state

        # Transicoes internas suavizadas pelo belief — somente entre estados
        # ativos (APPROACHING/INSIDE/EXITING). Saida para OUTSIDE acontece
        # exclusivamente pela histerese do gate acima; sem isso, o EGO
        # dizendo OUTSIDE no vao entre a placa e a zona derrubava o estado
        # que o sign latch deveria sustentar.
        target = STATES[int(self.belief.argmax())]
        if (target != self.state and target != WZState.OUTSIDE
                and self.belief.max() >= 0.60):
            self.state = target
        return self.state
