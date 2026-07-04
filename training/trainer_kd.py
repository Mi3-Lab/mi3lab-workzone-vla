"""Knowledge Distillation trainer for Alpamayo Stage 3 (10B → 2B).

Loss: L = (1 - λ) * L_SFT + λ * τ² * KL(teacher || student)
Reference: LLAVADI (arxiv:2407.19409) — τ=0.7 optimal for VLMs
"""

import logging as _logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.utils import is_sagemaker_mp_enabled

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer, TrainingArguments

_logger = _logging.getLogger(__name__)


@dataclass
class KDTrainingArguments(TrainingArguments):
    kd_lambda: float = field(
        default=0.5,
        metadata={"help": "Weight of KD loss. L = (1-λ)*L_SFT + λ*L_KD"},
    )
    kd_temperature: float = field(
        default=0.7,
        metadata={"help": "Softmax temperature for KD (τ=0.7 per LLAVADI paper)"},
    )


class KDTrainer(ReasoningVLA_Trainer):
    """Trainer that distills knowledge from a frozen teacher into a student model.

    The teacher is kept frozen and on the same device as the student.
    KD loss: τ² * KL(softmax(logits_T/τ) || log_softmax(logits_S/τ))
    """

    def __init__(self, teacher_model: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self._teacher_placed = False

    def _place_teacher(self, device: torch.device) -> None:
        if not self._teacher_placed:
            self.teacher = self.teacher.to(device=device, dtype=torch.bfloat16)
            self._teacher_placed = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Find device from inputs
        device = next(
            (v.device for v in inputs.values() if isinstance(v, torch.Tensor)), None
        )
        if device is not None:
            self._place_teacher(device)

        # sft_base_model.forward() receives tokenized_data as a named arg and calls
        # .pop("input_ids") on it — mutating the inner dict. We must give each
        # forward pass its own copy of that nested dict so keys survive both calls.
        td_orig = inputs.get("tokenized_data", {})
        student_inputs = {**inputs, "tokenized_data": dict(td_orig)}
        teacher_inputs = {**inputs, "tokenized_data": dict(td_orig)}

        # Student forward
        student_out = model(**student_inputs)
        loss_sft = student_out.loss
        logits_s = student_out.logits  # [B, L, V]

        # Teacher forward (no grad, no loss computation needed)
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs)
        logits_t = teacher_out.logits.to(logits_s.device, dtype=torch.float32)

        # Shift logits (predict next token, same convention as _compute_next_token_loss)
        shift_s = logits_s[..., :-1, :].float()  # [B, L-1, V]
        shift_t = logits_t[..., :-1, :]

        # Restrict KD to the same supervised positions as loss_sft (labels_mask,
        # e.g. only traj_future tokens) — without this, kl_div sums over the
        # full sequence (image/prompt tokens included), inflating loss_kd by
        # ~2 orders of magnitude relative to loss_sft (empirically 24 vs 11855
        # on a 3204-token sequence with ~128 supervised tokens) and drowning out
        # the SFT signal once combined with kd_lambda.
        labels_mask = inputs.get("labels_mask")
        if labels_mask is not None:
            mask = labels_mask[:, 1:].to(shift_s.device)
            shift_s = shift_s[mask]
            shift_t = shift_t[mask]

        τ = self.args.kd_temperature
        λ = self.args.kd_lambda

        # KL(T||S): KL divergence with T as target distribution
        p_t = F.softmax(shift_t / τ, dim=-1)
        log_p_s = F.log_softmax(shift_s / τ, dim=-1)
        # kl_div expects (input=log_probs, target=probs); shift_s/shift_t are now
        # [N_valid_tokens, V] (mask flattens batch+seq), so batchmean divides by
        # N_valid_tokens — a proper per-token average, comparable in scale to loss_sft.
        loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (τ ** 2)

        loss = (1.0 - λ) * loss_sft + λ * loss_kd

        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            _logger.info(
                f"step={self.state.global_step} "
                f"loss={loss.item():.4f} "
                f"loss_sft={loss_sft.item():.4f} "
                f"loss_kd={loss_kd.item():.4f}"
            )

        return (loss, student_out) if return_outputs else loss
