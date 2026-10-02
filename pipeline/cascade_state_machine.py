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


# Qualificadores no texto livre do DESC que tornam o objeto irrelevante pro
# trajeto do veiculo (calibrado contra o ground truth humano ROADWork/CMU):
# - "sidewalk": obra de calcada nao poe o carro numa work zone de via
# - "parked": veiculo de obra estacionado != zona ativa (o modelo qualifica
#   "parked" espontaneamente quando e o caso, e o GT marca outside)
IRRELEVANT_QUALIFIERS = ("sidewalk", "parked")


def has_corroboration(desc_txt, sign_txt):
    """desc_txt/sign_txt citing a specific work-zone object corroborates the
    gate -- EXCEPT sentences whose location/status qualifier marks the object
    as irrelevant to the vehicle's path (see IRRELEVANT_QUALIFIERS).
    Calibrating against human ground truth (workzone_annotations_full.json,
    CMU ROADWork) showed these are the dominant false-positive sources: dense
    urban footage often has real sidewalk-level barriers/tubular markers
    (unrelated sidewalk construction) or a lone parked work vehicle that DESC
    correctly describes, but which don't put the vehicle anywhere near a road
    work zone the way the outside/approaching/inside/exiting ground truth is
    defined. A sentence naming an object with no such qualifier is trusted.

    Only the free-text object-location prefix is checked (up to the first
    "Scene:" marker of the fine-tuned template). The trailing structured
    fields ("Scene: ...", "Lane status: ...", "Detected objects: N Type(s).")
    restate the same objects WITHOUT the location qualifier and would defeat
    the exclusion above otherwise -- e.g. "Barriers on right sidewalk.
    Scene: ... Detected objects: 2 Barrier(s)." has to be excluded based on
    the first sentence, but "Detected objects: 2 Barrier(s)." alone would
    match WZ_OBJECT_RE with no qualifier to filter on.
    """
    if desc_txt:
        object_prefix = desc_txt.split("Scene:", 1)[0]
        for sentence in re.split(r"(?<=[.!?])\s+", object_prefix):
            low = sentence.lower()
            if WZ_OBJECT_RE.search(sentence) and not any(q in low for q in IRRELEVANT_QUALIFIERS):
                return True
    if sign_txt and any(k in sign_txt.upper() for k in SIGN_WZ_KEYWORDS):
        return True
    return False


class CascadeStateMachine:
    """Camadas 1+3 do redesign. Janela N, entrada K_ENTER, saida K_EXIT."""

    N = 5
    K_ENTER = 3
    K_EXIT = 4
    K_FADING = 2  # gate inativo em >=K_FADING de N -> aviso precoce (EXITING)

    def __init__(self, restrict_active=True, N=None, k_enter=None,
                 k_exit=None, k_fading=None, use_ego=True):
        # restrict_active=False restaura o argmax original sobre os 4 estados
        # (inclusive OUTSIDE) no filtro Bayesiano -- usado apenas como
        # ablacao; ver comentario na secao de transicoes internas.
        #
        # N/k_* e use_ego permitem calibrar a maquina de estados por modelo: os
        # defaults foram fixados no split de calibracao para o nosso 2B, e
        # herda-los para outro modelo mede os hiperparametros, nao o modelo.
        self.restrict_active = restrict_active
        if N is not None:
            self.N = N
        if k_enter is not None:
            self.K_ENTER = k_enter
        if k_exit is not None:
            self.K_EXIT = k_exit
        if k_fading is not None:
            self.K_FADING = k_fading
        self.use_ego = use_ego
        self.state = WZState.OUTSIDE
        self.gate_window = deque(maxlen=self.N)
        self.corr_window = deque(maxlen=self.N)
        self.belief = PRIOR.copy()

    def _gate_active(self):
        return sum(1 for g in self.gate_window if g) >= self.K_ENTER

    def _gate_clear(self):
        return sum(1 for g in self.gate_window if not g) >= self.K_EXIT

    def _gate_fading(self):
        return sum(1 for g in self.gate_window if not g) >= self.K_FADING

    def _corroborated(self):
        return any(self.corr_window)

    def update(self, gate, ego_txt, desc_txt="", sign_txt="", fast_entry=False):
        """Um passo (um frame amostrado). Retorna o estado exibido.

        fast_entry: True quando ALGUM canal deu evidencia de alta confianca
        NESTE MESMO frame -- uma placa transcrita com keyword de obra, ou
        gate ativo + DESC citando um objeto especifico (cone/barricada/
        marcador/veiculo de obra) na mesma passada. Diferente do GATE sozinho
        (resposta binaria de uma pergunta composta, ruidosa frame-a-frame),
        essas sao leituras verificaveis -- nao se beneficiam de voto
        majoritario numa janela de N frames. Sem esse atalho, esperar
        K_ENTER=3 de N=5 com um ciclo mais lento (SIGN roda todo ciclo em
        OUTSIDE, ~300ms/ciclo) atrasa a entrada em segundos de mundo real: o
        carro ja esta dentro da obra quando o estado finalmente sai de
        OUTSIDE (sintoma observado: EGO ja responde "INSIDE" no primeiro
        ciclo logo apos a transicao). Quando gate=Yes sem corroboracao
        especifica (possivel falso-positivo do gate sozinho), fast_entry fica
        False e a entrada continua exigindo a janela suavizada, preservando a
        protecao original contra ruido de frame unico.
        """
        return self.update_evidence(bool(gate),
                                    has_corroboration(desc_txt, sign_txt),
                                    classify_ego(ego_txt) if self.use_ego else None,
                                    fast_entry)

    def update_evidence(self, gate, corroborated, ego_i, fast_entry=False):
        """Same step, but taking already-parsed evidence instead of raw text.

        update() is the text-facing entry point; this one lets the offline
        calibration (calibrate_cascade.py) replay recorded evidence through the
        *exact* state logic that runs live, so the thresholds we search over are
        the thresholds we deploy.
        """
        self.gate_window.append(bool(gate))
        self.corr_window.append(bool(corroborated))

        if self.state == WZState.OUTSIDE:
            # Entrada exige: gate ativo em >=K_ENTER dos ultimos N frames
            # E corroboracao de segundo canal em pelo menos 1 deles -- OU
            # evidencia de alta confianca no mesmo frame, que dispensa a
            # janela.
            if fast_entry or (self._gate_active() and self._corroborated()):
                self.state = WZState.APPROACHING
                self.belief = np.array([0.10, 0.70, 0.15, 0.05])
            return self.state

        # -------- modo ativo: filtro Bayesiano (so canal EGO) ----------
        # ego_i=None (use_ego=False) deixa a emissao uniforme: usado quando o
        # canal EGO do modelo nao discrimina (medido para o Cosmos3-Edge base),
        # caso em que o estado passa a ser conduzido so por gate+corroboracao.
        p_ego = peaked_dist(ego_i, 3) if ego_i is not None else np.ones(3) / 3
        emission = EGO_EMIT @ p_ego
        b = (self.belief @ TRANS) * emission
        self.belief = b / b.sum()

        # Saida global: evidencia sumiu de forma sustentada -> OUTSIDE.
        if self._gate_clear():
            self.state = WZState.OUTSIDE
            self.belief = PRIOR.copy()
            return self.state

        # EXITING via perda parcial de evidencia do GATE, nao do EGO. Testado
        # empiricamente contra frames de "exiting" do ground truth (mesmo
        # oferecendo a palavra "EXITING" como opcao explicita no prompt): o
        # EGO sempre responde "INSIDE" -- nao foi fine-tuned para produzir
        # essa resposta, entao o filtro Bayesiano nunca alcanca EXITING via
        # emissao do EGO. O proprio GATE comecando a ficar negativo (mas
        # ainda nao o suficiente pra _gate_clear) e' o sinal disponivel.
        if self.state == WZState.INSIDE and self._gate_fading():
            self.state = WZState.EXITING
            return self.state

        # Transicoes internas suavizadas pelo belief -- APENAS entre os
        # estados ativos (APPROACHING/INSIDE/EXITING). Sair para OUTSIDE e'
        # papel exclusivo do _gate_clear() acima (histerese K_EXIT=4 de N=5).
        # Sem essa restricao, um unico ciclo com EGO="OUTSIDE" logo apos uma
        # entrada por placa (onde o veiculo ainda nao alcancou os cones, mas
        # ja avistou o aviso) derruba o belief.argmax() de volta pra OUTSIDE
        # na hora, sem passar pela histerese de saida -- foi exatamente o bug
        # reportado (placa -> APPROACHING -> OUTSIDE quase instantaneo).
        if self.restrict_active:
            active_belief = self.belief[1:]
            target = STATES[1 + int(active_belief.argmax())]
            if target != self.state and active_belief.max() >= 0.60:
                self.state = target
        else:  # ablacao: comportamento original (argmax sobre os 4 estados)
            target = STATES[int(self.belief.argmax())]
            if target != self.state and self.belief.max() >= 0.60:
                self.state = target
        return self.state
