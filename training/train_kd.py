"""Stage 3 — Knowledge Distillation training: Alpamayo 10B → 2B.

Teacher: TrainableReasoningVLA loaded from Stage 1 v2 checkpoint (frozen).
Student: TrainableReasoningVLA with Cosmos-Reason2-2B backbone (trains).

Usage:
    python train_kd.py --config-name sft_stage3_kd
"""

import os
import sys
import glob

import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers.trainer_utils import get_last_checkpoint

from alpamayo_r1.common import logging
from alpamayo.common import misc, config_utils, wandb_utils
from alpamayo_r1.common.logging import setup_logging

from alpamayo1_5_sft.trainer_kd import KDTrainer, KDTrainingArguments
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_sft.models.kd_model import build_student_model

setup_logging()
logger = logging.RankedLogger("train_kd", rank_zero_only=True)
logger.setLevel("INFO")

BASE = "/data/wesleyferreiramaia/wokzone-alpamayo"


def load_teacher(cfg: DictConfig) -> TrainableReasoningVLA:
    """Load and freeze Stage 1 v2 teacher model."""
    teacher = TrainableReasoningVLA.from_alpamayo_checkpoint(
        checkpoint_path=cfg.kd.teacher_checkpoint,
        vlm_name_or_path=cfg.kd.teacher_vlm_path,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info(f"Teacher loaded from {cfg.kd.teacher_checkpoint} (frozen)")
    return teacher


@hydra.main(version_base=None, config_path=None, config_name="config")
def train(cfg: DictConfig) -> None:
    misc.seed_everything(42)

    training_args = KDTrainingArguments(
        **OmegaConf.to_container(cfg.trainer, resolve=True)
    )
    logger.info("Config:\n" + misc.pformat(OmegaConf.to_container(cfg, resolve=True)))

    # ── Teacher (frozen 10B) ──────────────────────────────────────────────────
    teacher = load_teacher(cfg)

    # ── Student (fresh 2B) ───────────────────────────────────────────────────
    student = build_student_model(
        alpamayo_base_path=cfg.kd.alpamayo_base_path,
        student_vlm_path=cfg.kd.student_vlm_path,
    )

    # ── Dataset (same as Stage 1) ─────────────────────────────────────────────
    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=student.config
    )
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=student.config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=student.config
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = []
    for cb_name, cb_cfg in cfg.get("callbacks", {}).items():
        logger.info(f"Initializing callback {cb_name}")
        callbacks.append(hyu.instantiate(cb_cfg, _convert_="partial"))

    # ── W&B ───────────────────────────────────────────────────────────────────
    if cfg.get("wandb", None) is not None:
        wandb_utils.init_wandb(cfg)

    # ── KD Trainer ────────────────────────────────────────────────────────────
    trainer = KDTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    # Resume from checkpoint if exists
    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            logger.info(f"Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(training_args.output_dir)
    logger.info(f"Student saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
