"""Compact prompt-only checkpoint serialization and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .prompt_learner import ObjectAgnosticPromptLearner


CHECKPOINT_SCHEMA_VERSION = 1
PROMPT_KEYS = ("normal_context", "abnormal_context")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_state(learner: ObjectAgnosticPromptLearner) -> dict[str, Tensor]:
    return {
        "normal_context": learner.normal_context.detach().cpu().clone(),
        "abnormal_context": learner.abnormal_context.detach().cpu().clone(),
    }


def save_prompt_checkpoint(
    path: str | Path,
    learner: ObjectAgnosticPromptLearner,
    *,
    dataset: str,
    epoch: int,
    seed: int,
    prompt_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    sample_manifest_sha256: str,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset": dataset,
        "epoch": epoch,
        "seed": seed,
        "prompt_config": dict(prompt_config),
        "training_config": dict(training_config),
        "sample_manifest_sha256": sample_manifest_sha256,
        "prompt_state": prompt_state(learner),
    }
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return sha256_file(target)


def load_prompt_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt checkpoint is not a mapping: {checkpoint_path}")
    required = {
        "schema_version",
        "dataset",
        "epoch",
        "seed",
        "prompt_config",
        "training_config",
        "sample_manifest_sha256",
        "prompt_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Prompt checkpoint is missing keys: {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload['schema_version']}")
    state = payload["prompt_state"]
    if not isinstance(state, dict) or set(state) != set(PROMPT_KEYS):
        raise ValueError("checkpoint prompt_state must contain only normal/abnormal contexts")
    normal, abnormal = state["normal_context"], state["abnormal_context"]
    if not isinstance(normal, Tensor) or not isinstance(abnormal, Tensor):
        raise TypeError("checkpoint contexts must be tensors")
    if normal.ndim != 2 or normal.shape != abnormal.shape:
        raise ValueError("checkpoint prompt contexts must be equal-shape 2D tensors")
    return payload


def restore_prompt_checkpoint(
    learner: ObjectAgnosticPromptLearner,
    path: str | Path,
) -> dict[str, Any]:
    payload = load_prompt_checkpoint(path)
    state = payload["prompt_state"]
    expected = learner.normal_context.shape
    if state["normal_context"].shape != expected:
        raise ValueError(
            f"Checkpoint prompt shape {tuple(state['normal_context'].shape)} does not "
            f"match learner shape {tuple(expected)}"
        )
    with torch.no_grad():
        learner.normal_context.copy_(
            state["normal_context"].to(
                device=learner.normal_context.device,
                dtype=learner.normal_context.dtype,
            )
        )
        learner.abnormal_context.copy_(
            state["abnormal_context"].to(
                device=learner.abnormal_context.device,
                dtype=learner.abnormal_context.dtype,
            )
        )
    return payload
