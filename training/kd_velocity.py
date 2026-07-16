"""KD de campo de velocidade (flow-matching) do professor 10B pro aluno 2B.

Por que isto e diferente da destilacao anterior:
  - antes (output-level): o professor dava UM alvo de trajetoria por clipe, e
    ainda filtrado pra ficar perto do GT -- ou seja, quase o proprio GT. Ganho
    medido: ~0.7% de minADE. Praticamente um no-op.
  - agora (velocity-field): a trajetoria neste modelo NAO sai de tokens
    discretos, sai de um policy de flow-matching (o expert preve a velocidade
    v = x - noise a partir de um ponto ruidoso). O professor entao pode ser
    consultado em QUALQUER ponto do caminho de difusao, dando supervisao densa
    -- muitos (noisy_x, t) por clipe em vez de um unico ponto final.

Requisito central: professor e aluno tem que ver o MESMO ponto de ruido, senao
o MSE entre as velocidades nao tem sentido. Por isso o ruido e sorteado UMA vez
(_process_traj_future_training) e reusado nos dois experts.

Viabilidade (verificada, nao suposta):
  - mesmo vocabulario (155697) e mesmo traj_token_start_idx (151669);
  - mesmo preprocessamento de imagem (720 patches, grid 20x36) nos dois;
    => um unico batch serve os dois modelos.
  - o expert dos dois e identico (hidden 2048, head_dim 128) e os dois VLMs tem
    kv_heads=8/head_dim=128, entao cada expert atende (via GQA) ao KV-cache do
    SEU proprio VLM, mesmo o texto sendo 2048 (aluno) vs 4096 (professor).
"""
from typing import Any

import torch
import torch.nn.functional as F

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import ReasoningVLAOutput
from alpamayo_r1.models.base_model import IGNORE_INDEX


def _move(x: Any, device: torch.device) -> Any:
    """Move tensores pra `device`, recursivamente (o batch tem dicts aninhados)."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, device) for v in x)
    return x


def _expert_velocity(
    model: TrainableAlpamayoR1,
    vlm_outputs: Any,
    input_ids: torch.Tensor,
    future_traj_data: dict[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """Velocidade prevista pelo expert de `model`, condicionada no KV-cache do
    VLM de `model`, no ponto de ruido dado. Replica sft_alpamayo_r1.forward
    (linhas 138-174), so que parametrizado pelo modelo -- pra rodar identico no
    professor e no aluno.
    """
    future_start_token_id = model.config.traj_token_ids["future_start"]
    last_future_start = (input_ids == future_start_token_id).nonzero(as_tuple=False)[-1, 1] + 1

    action_embeds = model.action_in_proj(
        future_traj_data["noisy_x"], future_traj_data["timesteps"]
    )
    kv_cache = vlm_outputs.past_key_values
    kv_cache.crop(last_future_start)
    if model.stop_grad_from_vlm:
        for layer in kv_cache.layers:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()

    position_ids = model._process_position_ids_qwen2_5_vl(
        vlm_outputs, batch_size, action_embeds.shape[1], action_embeds.device
    )
    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    expert_outputs = model.expert(
        inputs_embeds=action_embeds,
        position_ids=position_ids,
        past_key_values=kv_cache,
        attention_mask=None,
        use_cache=True,
        **forward_kwargs,
    )
    diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
    pred = model.action_out_proj(diffusion_out)
    return pred.view(-1, *model.action_space.get_action_space_dims())


class VelocityKDAlpamayoR1(TrainableAlpamayoR1):
    """Aluno 2B treinado com flow-matching contra o GT + KD de velocidade do professor."""

    def __init__(
        self,
        *args: Any,
        teacher_path: str,
        kd_weight: float = 1.0,
        teacher_device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.kd_weight = kd_weight
        self.teacher_path = teacher_path
        # Numa L40S (48GB) o aluno+optimizer (~26GB) e o professor (~22GB) nao
        # cabem juntos. Como o professor e' so inferencia (sem grads/optimizer),
        # ele vai pra OUTRA GPU do mesmo no: GPU0 treina, GPU1 so hospeda o
        # professor. O que atravessa entre elas e' so o batch e a velocidade
        # prevista -- tensores pequenos.
        self.teacher_device = teacher_device
        # guardado numa lista pra NAO virar submodulo: assim o professor fica fora
        # do optimizer, fora do state_dict salvo e fora do wrap do DDP.
        self._teacher: list[TrainableAlpamayoR1] = []

    def _get_teacher(self, student_device: torch.device) -> TrainableAlpamayoR1:
        """Carrega o professor no 1o forward, nao no __init__.

        Este __init__ roda DENTRO do from_pretrained do aluno, que ativa o
        contexto de meta-device do HF (low_cpu_mem_usage). Um from_pretrained
        aninhado ali dentro nasce em `meta` e qualquer .to(device) depois estoura
        "Cannot copy out of meta tensor". No 1o forward ja estamos fora desse
        contexto.
        """
        if not self._teacher:
            device = (
                torch.device(self.teacher_device)
                if self.teacher_device is not None
                else student_device
            )
            teacher = TrainableAlpamayoR1.from_pretrained(
                self.teacher_path, dtype=torch.bfloat16
            )
            teacher.to(device).eval()
            teacher.requires_grad_(False)
            self._teacher.append(teacher)
        return self._teacher[0]

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        input_ids = tokenized_data.pop("input_ids")
        batch_size = input_ids.shape[0]
        teacher = self._get_teacher(input_ids.device)
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # UM sorteio de ruido, compartilhado: professor e aluno avaliados no mesmo
        # (noisy_x, t). Sem isto o MSE entre as velocidades compara pontos diferentes.
        future_traj_data = self._process_traj_future_training(traj_data)

        with torch.no_grad():
            student_vlm_outputs = self.vlm(
                input_ids=input_ids, labels=labels, use_cache=True, **tokenized_data
            )
        pred_student = _expert_velocity(
            self, student_vlm_outputs, input_ids, future_traj_data, batch_size
        )

        # o professor pode estar em outra GPU (ver teacher_device): leva o batch
        # ate ele e traz so a velocidade prevista de volta.
        t_device = next(teacher.parameters()).device
        with torch.no_grad():
            t_input_ids = input_ids.to(t_device)
            teacher_vlm_outputs = teacher.vlm(
                input_ids=t_input_ids,
                labels=labels.to(t_device),
                use_cache=True,
                **_move(tokenized_data, t_device),
            )
            pred_teacher = _expert_velocity(
                teacher,
                teacher_vlm_outputs,
                t_input_ids,
                _move(future_traj_data, t_device),
                batch_size,
            )
            pred_teacher = pred_teacher.to(pred_student.device)

        flow_loss = self.diffusion.compute_loss_from_pred(
            training_data=future_traj_data, pred=pred_student
        )
        kd_loss = F.mse_loss(pred_student, pred_teacher.detach().to(pred_student.dtype))
        loss = flow_loss + self.kd_weight * kd_loss

        return ReasoningVLAOutput(loss=loss)
