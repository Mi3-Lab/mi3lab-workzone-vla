"""Modelo compacto de trajetoria (~485M): ResNet-101 + decoder DETR multimodal.

Inspirado em github.com/mu-hashmi/alpamayo-r1-distilled (MIT), que destila o
Alpamayo num preditor de trajetoria sem VLM. NAO e' uma "cabeca de acao" que se
pluga no nosso VLM -- e' um modelo completo e independente, que le pixels direto.
Aqui e' reimplementado sobre a NOSSA infra (PAIDataset + action space do
alpamayo_r1), pra que as metricas saiam diretamente comparaveis aos baselines.

Por que tentar isto:
    Nosso VLA tem um expert de 1.77B que nasce ALEATORIO e ve 3644 clipes
    (~500k params por exemplo). Quatro experimentos com alvos diferentes deram
    todos minADE ~4.6. Aqui o backbone visual ja vem pre-treinado (ImageNet) e o
    decoder e' ~4x menor -- muito mais eficiente em dados.

Diferencas deliberadas em relacao ao repo de referencia:
    - Multimodal (M=6 modos + winner-takes-all) em vez de deterministico. O repo
      original preve UMA trajetoria; nossos baselines sao medidos em minADE
      (melhor de 6 amostras). Comparar 1 contra best-of-6 seria enganoso, entao
      o modelo preve 6 modos e a loss otimiza justamente o melhor deles.
    - Reusa UnicycleAccelCurvatureActionSpace do alpamayo_r1 (o MESMO que o nosso
      VLA e o professor usam) em vez de reimplementar o unicycle. Mesmos limites,
      mesma integracao => a comparacao mede a arquitetura, nao o action space.
    - 224x384 em vez de 224x224: as cameras sao 1080x1920 (9:16); espremer pra
      quadrado distorce a geometria lateral, que e' exatamente o que precisamos
      prever.
"""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace

H_IN, W_IN = 224, 384        # preserva ~9:16; divisivel por 32 -> grade 7x12
D_MODEL, N_HEADS, N_LAYERS, FFN = 1280, 20, 16, 5120
N_WAYPOINTS, N_MODES = 64, 6


def _sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Positional encoding 2D sinusoidal -> [h*w, d]."""
    assert d % 4 == 0
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(d // 4) / (d // 4)))
    out = []
    for pos in (y.flatten(), x.flatten()):
        ang = pos[:, None].float() * omega[None, :]
        out += [ang.sin(), ang.cos()]
    return torch.cat(out, dim=1)  # [h*w, d]


class TrajStudent485M(nn.Module):
    def __init__(self, action_space_cfg: dict[str, Any], n_cams: int = 4, n_frames: int = 4):
        super().__init__()
        self.n_cams, self.n_frames = n_cams, n_frames

        # --- encoder visual: ResNet-101 ImageNet, COMPARTILHADO entre as cameras ---
        net = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        )
        # 4 frames x 3 RGB = 12 canais. Inflamos o conv1 pre-treinado repetindo os
        # pesos e dividindo por n_frames: preserva a magnitude da ativacao, entao o
        # resto do backbone continua no regime em que foi treinado.
        w = net.conv1.weight.data                                   # [64,3,7,7]
        conv1 = nn.Conv2d(3 * n_frames, 64, 7, 2, 3, bias=False)
        conv1.weight.data = w.repeat(1, n_frames, 1, 1) / n_frames
        net.conv1 = conv1
        self.backbone = nn.Sequential(*list(net.children())[:-2])   # -> [B,2048,7,12]

        self.proj = nn.Conv2d(2048, D_MODEL, 1)
        self.cam_embed = nn.Parameter(torch.zeros(n_cams, D_MODEL))
        gh, gw = H_IN // 32, W_IN // 32
        self.register_buffer("pos2d", _sincos_2d(gh, gw, D_MODEL), persistent=False)

        # --- ego history -> um token de contexto (da a velocidade/curvatura atual) ---
        self.ego_mlp = nn.Sequential(
            nn.Linear(16 * 6, 512), nn.ReLU(), nn.Linear(512, D_MODEL)
        )

        # --- decoder DETR (pre-LN), queries = M modos x 64 waypoints ---
        layer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, FFN, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, N_LAYERS)
        self.queries = nn.Parameter(torch.randn(N_MODES, N_WAYPOINTS, D_MODEL) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(N_WAYPOINTS, D_MODEL) * 0.02)

        # --- cabecas: acao (accel, curvatura) normalizada + confianca do modo ---
        self.action_head = nn.Sequential(
            nn.Linear(D_MODEL, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        self.mode_head = nn.Linear(D_MODEL, 1)

        # MESMO action space do VLA e do professor -> metricas comparaveis
        self.action_space = UnicycleAccelCurvatureActionSpace(**action_space_cfg)

    def _encode_images(self, image_frames: torch.Tensor) -> torch.Tensor:
        """image_frames [B, cam, frm, 3, H, W] -> memoria [B, cam*gh*gw, D]."""
        B, C, T = image_frames.shape[:3]
        x = image_frames.flatten(3, 3) if image_frames.dim() == 7 else image_frames
        x = x.view(B * C, T * 3, *image_frames.shape[-2:])          # [B*C, 12, H, W]
        x = F.interpolate(x, size=(H_IN, W_IN), mode="bilinear", align_corners=False)
        f = self.proj(self.backbone(x))                              # [B*C, D, gh, gw]
        f = f.flatten(2).transpose(1, 2)                             # [B*C, gh*gw, D]
        f = f + self.pos2d[None].to(f.dtype)
        f = f.view(B, C, -1, D_MODEL) + self.cam_embed[None, :, None, :]
        return f.reshape(B, -1, D_MODEL)

    def _ego_token(self, xyz: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """xyz [B,16,3], rot [B,16,3,3] -> [B,1,D]."""
        yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
        feats = torch.cat([xyz, yaw.sin()[..., None], yaw.cos()[..., None],
                           torch.zeros_like(yaw)[..., None]], dim=-1)   # [B,16,6]
        return self.ego_mlp(feats.flatten(1))[:, None, :]

    def forward(self, image_frames, ego_history_xyz, ego_history_rot):
        B = image_frames.shape[0]
        mem = torch.cat([self._encode_images(image_frames),
                         self._ego_token(ego_history_xyz, ego_history_rot)], dim=1)

        q = (self.queries + self.time_embed[None]).reshape(1, N_MODES * N_WAYPOINTS, D_MODEL)
        q = q.expand(B, -1, -1)
        h = self.decoder(q, mem).view(B, N_MODES, N_WAYPOINTS, D_MODEL)

        action = self.action_head(h)                                   # [B,M,64,2]
        logits = self.mode_head(h.mean(2)).squeeze(-1)                 # [B,M]

        # integra cada modo pelo unicycle -> trajetoria cinematicamente factivel
        hx = ego_history_xyz[:, None].expand(-1, N_MODES, -1, -1).reshape(B * N_MODES, -1, 3)
        hr = ego_history_rot[:, None].expand(-1, N_MODES, -1, -1, -1).reshape(B * N_MODES, -1, 3, 3)
        xyz, rot = self.action_space.action_to_traj(
            action.reshape(B * N_MODES, N_WAYPOINTS, 2), hx, hr
        )
        return (xyz.view(B, N_MODES, N_WAYPOINTS, 3),
                rot.view(B, N_MODES, N_WAYPOINTS, 3, 3),
                logits)


def wta_loss(pred_xyz, logits, gt_xyz):
    """Winner-takes-all: so o melhor modo recebe gradiente de regressao.

    E' o alvo certo aqui: nossos baselines sao medidos em minADE (melhor de 6),
    entao a loss otimiza exatamente a metrica. Alem disso a WTA e' o que faz os
    modos se ESPECIALIZAREM (seguir reto / curvar / frear) em vez de todos
    colapsarem na media -- que e' a patologia oposta do "leque" do nosso VLA.
    """
    d = (pred_xyz[..., :2] - gt_xyz[:, None, :, :2]).norm(dim=-1).mean(-1)  # [B,M]
    best = d.argmin(1)
    reg = d.gather(1, best[:, None]).squeeze(1).mean()
    cls = F.cross_entropy(logits, best)          # aprende a rankear os modos
    return reg + 0.5 * cls, reg.detach()
