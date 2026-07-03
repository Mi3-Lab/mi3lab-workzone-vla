"""Student model builder for Knowledge Distillation (Stage 3).

Creates a TrainableReasoningVLA with Cosmos-Reason2-2B as the VLM backbone,
using the trajectory tokenizer config from the base Alpamayo-1.5-10B model.
The student starts from Cosmos-Reason2-2B pretrained weights — no fine-tuned
VLM weights are loaded (those come from the teacher via KD).
"""

import json
from pathlib import Path
from typing import Any

from hydra.utils import instantiate

from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo_r1.models.base_model import ReasoningVLAConfig
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def build_student_model(
    alpamayo_base_path: str,
    student_vlm_path: str,
) -> TrainableReasoningVLA:
    """Build student model for KD using Alpamayo architecture + Cosmos-Reason2-2B.

    Reads trajectory tokenizer config from the base Alpamayo-1.5-10B model,
    but uses student_vlm_path as the VLM backbone. No fine-tuned weights are
    loaded — the student starts from Cosmos-Reason2-2B pretrained weights.

    Args:
        alpamayo_base_path: Path to Alpamayo-1.5-10B-A1-format (for traj config).
        student_vlm_path: Path to Cosmos-Reason2-2B (student VLM backbone).

    Returns:
        Student TrainableReasoningVLA with Cosmos-Reason2-2B backbone.
    """
    config_path = Path(alpamayo_base_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        base_config = json.load(f)

    config_kwargs: dict[str, Any] = {
        "vlm_name_or_path": student_vlm_path,
        "vlm_backend": base_config.get("vlm_backend", "qwenvl3"),
        "traj_tokenizer_cfg": base_config.get("traj_tokenizer_cfg"),
        "hist_traj_tokenizer_cfg": base_config.get("hist_traj_tokenizer_cfg"),
        "traj_vocab_size": base_config.get("traj_vocab_size"),
        "tokens_per_history_traj": base_config.get("tokens_per_history_traj"),
        "tokens_per_future_traj": base_config.get("tokens_per_future_traj"),
        "model_dtype": base_config.get("model_dtype", "bfloat16"),
        "attn_implementation": base_config.get("attn_implementation", "flash_attention_2"),
        "min_pixels": base_config.get("min_pixels"),
        "max_pixels": base_config.get("max_pixels"),
        "add_special_tokens": base_config.get("add_special_tokens", True),
    }

    config = instantiate(
        {
            "_target_": "alpamayo_r1.models.base_model.ReasoningVLAConfig",
            "_recursive_": False,
            "_convert_": "all",
            **config_kwargs,
        }
    )

    pretrained_modules = {}
    if config.traj_tokenizer_cfg is not None:
        pretrained_modules["traj_tokenizer"] = instantiate(config.traj_tokenizer_cfg)

    logger.info(f"Building student model with VLM: {student_vlm_path}")
    model = TrainableReasoningVLA(
        config, pretrained_modules=pretrained_modules or None
    )
    logger.info("Student model built — starting from Cosmos-Reason2-2B pretrained weights")
    return model
