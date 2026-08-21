"""End-to-end per-source-dataset prompt fitting and artifact packaging."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Callable, Sequence

import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm

from .checkpoint import load_prompt_checkpoint, save_prompt_checkpoint, sha256_file
from .config import ExperimentConfig
from .data import (
    PromptTrainingDataset,
    PromptTrainingSample,
    automatic_attack_train_split,
    canonical_manifest,
    discover_dataset,
    load_attack_train_manifest,
)
from .losses import AnomalyCLIPPromptLoss
from .model import PublicCLIPPromptModel, build_public_clip_prompt_model


@dataclass(frozen=True)
class EpochHistory:
    epoch: int
    total_loss: float
    image_loss: float
    pixel_loss: float
    pixel_loss_weighted: float
    learning_rate: float
    batches: int
    samples: int


@dataclass(frozen=True)
class DatasetRunResult:
    dataset: str
    output_directory: Path
    checkpoint: Path
    sample_count: int
    history: tuple[EpochHistory, ...]


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def prepare_training_samples(
    config: ExperimentConfig,
    dataset: str,
) -> tuple[list[PromptTrainingSample], list[dict[str, object]], str, str | None]:
    root_value = config.data.root_for(dataset)
    if not root_value:
        raise ValueError(f"data.{dataset}_root is required")
    root = Path(root_value).expanduser().resolve()
    discovered = discover_dataset(dataset, root)
    manifest_value = config.data.manifest_for(dataset)
    if manifest_value:
        selected = load_attack_train_manifest(
            manifest_value,
            dataset=dataset,
            root=root,
            discovered=discovered,
        )
        source_manifest_sha = sha256_file(Path(manifest_value).expanduser().resolve())
    else:
        selected = automatic_attack_train_split(
            discovered,
            seed=config.training.seed,
            evaluation_fraction=config.data.automatic_evaluation_fraction,
        )
        source_manifest_sha = None
    records, selected_sha = canonical_manifest(selected, root)
    return selected, records, selected_sha, source_manifest_sha


def _optimizer_matches_prompts(
    optimizer: torch.optim.Optimizer,
    model: PublicCLIPPromptModel,
) -> bool:
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.trainable_parameters()}
    return optimized == expected


def train_prompt_model(
    model: PublicCLIPPromptModel,
    samples: Sequence[PromptTrainingSample],
    config: ExperimentConfig,
) -> list[EpochHistory]:
    training = config.training
    seed_everything(training.seed)
    model.assert_parameter_boundary()
    dataset = PromptTrainingDataset(samples, config.model.image_size)
    generator = torch.Generator().manual_seed(training.seed)
    loader = DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=model.device.type == "cuda",
        generator=generator,
        worker_init_fn=_seed_worker if training.num_workers else None,
    )
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=training.learning_rate,
        betas=training.betas,
    )
    if not _optimizer_matches_prompts(optimizer, model):
        raise RuntimeError("Optimizer parameter set is not exactly the two prompt contexts")
    criterion = AnomalyCLIPPromptLoss(pixel_weight=training.pixel_loss_weight)
    history: list[EpochHistory] = []
    dataset_name = samples[0].dataset if samples else "dataset"
    epoch_iterator = tqdm(
        range(1, training.epochs + 1),
        desc=f"{dataset_name} epochs",
        unit="epoch",
        disable=not training.show_progress,
    )
    for epoch in epoch_iterator:
        model.clip_model.eval()
        model.text_encoder.eval()
        model.prompt_learner.train()
        totals = {"total": 0.0, "image": 0.0, "pixel": 0.0, "weighted": 0.0}
        seen = 0
        batches = 0
        batch_iterator = tqdm(
            loader,
            desc=f"{dataset_name} epoch {epoch:02d}",
            unit="batch",
            leave=False,
            disable=not training.show_progress,
        )
        for batch in batch_iterator:
            images = batch["image"].to(model.device, non_blocking=True)
            masks = batch["mask"].to(model.device, non_blocking=True)
            labels = batch["label"].to(model.device, non_blocking=True)
            global_features, patch_features = model.encode_visual(images)
            image_logits, similarity_maps = model.predictions(
                global_features,
                patch_features,
                output_size=tuple(masks.shape[-2:]),
            )
            breakdown = criterion(image_logits, labels, similarity_maps, masks)
            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            if any(parameter.grad is not None for parameter in model.clip_model.parameters()):
                raise RuntimeError("A frozen CLIP parameter unexpectedly received a gradient")
            optimizer.step()
            batch_size = int(labels.shape[0])
            seen += batch_size
            batches += 1
            totals["total"] += float(breakdown.total.detach()) * batch_size
            totals["image"] += float(breakdown.image.detach()) * batch_size
            totals["pixel"] += float(breakdown.pixel.detach()) * batch_size
            totals["weighted"] += float(breakdown.pixel_weighted.detach()) * batch_size
            batch_iterator.set_postfix(
                total=f"{float(breakdown.total.detach()):.4f}",
                image=f"{float(breakdown.image.detach()):.4f}",
                pixel=f"{float(breakdown.pixel.detach()):.4f}",
                refresh=False,
            )
        if not seen:
            raise RuntimeError("Prompt-training loader produced no samples")
        row = EpochHistory(
            epoch=epoch,
            total_loss=totals["total"] / seen,
            image_loss=totals["image"] / seen,
            pixel_loss=totals["pixel"] / seen,
            pixel_loss_weighted=totals["weighted"] / seen,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            batches=batches,
            samples=seen,
        )
        history.append(row)
        epoch_iterator.set_postfix(
            total=f"{row.total_loss:.4f}",
            image=f"{row.image_loss:.4f}",
            pixel=f"{row.pixel_loss:.4f}",
        )
    return history


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git_revision(repository: Path) -> tuple[str | None, bool | None]:
    repository = repository.expanduser().resolve()
    safe = f"safe.directory={repository.as_posix()}"
    try:
        revision = subprocess.run(
            ["git", "-c", safe, "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", safe, "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_config(config: ExperimentConfig, dataset: str) -> dict[str, object]:
    resolved = config.to_dict()

    def absolute(value: str | None) -> str | None:
        return str(Path(value).expanduser().resolve()) if value else None

    resolved["model"]["anomalyclip_root"] = absolute(config.model.anomalyclip_root)
    resolved["model"]["clip_download_root"] = absolute(config.model.clip_download_root)
    for name in ("mvtec", "visa"):
        resolved["data"][f"{name}_root"] = absolute(config.data.root_for(name))
        resolved["data"][f"{name}_training_manifest"] = absolute(
            config.data.manifest_for(name)
        )
    resolved["artifacts"]["output_root"] = absolute(config.artifacts.output_root)
    resolved["active_dataset"] = dataset
    return resolved


def save_dataset_artifacts(
    model: PublicCLIPPromptModel,
    *,
    dataset: str,
    samples: Sequence[PromptTrainingSample],
    sample_records: list[dict[str, object]],
    sample_manifest_sha: str,
    source_manifest_sha: str | None,
    history: Sequence[EpochHistory],
    config: ExperimentConfig,
) -> DatasetRunResult:
    output = Path(config.artifacts.output_root).expanduser().resolve() / dataset
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"prompts_epoch{config.training.selected_epoch}.pt"
    checkpoint_sha = save_prompt_checkpoint(
        checkpoint,
        model.prompt_learner,
        dataset=dataset,
        epoch=config.training.selected_epoch,
        seed=config.training.seed,
        prompt_config=asdict(config.prompt),
        training_config=asdict(config.training),
        sample_manifest_sha256=sample_manifest_sha,
    )
    # Validate the just-written artifact before publishing its manifest.
    loaded = load_prompt_checkpoint(checkpoint)
    if loaded["dataset"] != dataset or loaded["epoch"] != config.training.selected_epoch:
        raise RuntimeError("Checkpoint round-trip metadata validation failed")

    history_path = output / "training_history.csv"
    fieldnames = list(asdict(history[0]).keys())
    temporary_history = history_path.with_name(f".{history_path.name}.tmp")
    with temporary_history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)
    temporary_history.replace(history_path)

    resolved = _resolved_config(config, dataset)
    resolved_config_path = output / "resolved_config.yaml"
    _atomic_text(resolved_config_path, yaml.safe_dump(resolved, sort_keys=False))

    project_root = Path(__file__).resolve().parents[2]
    revision, dirty = _git_revision(project_root)
    anomalyclip_revision, anomalyclip_dirty = (
        _git_revision(Path(config.model.anomalyclip_root))
        if config.model.anomalyclip_root
        else (None, None)
    )
    label_counts = {
        str(label): sum(sample.label == label for sample in samples) for label in (0, 1)
    }
    category_counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        bucket = category_counts.setdefault(sample.category, {"normal": 0, "abnormal": 0})
        bucket["normal" if sample.label == 0 else "abnormal"] += 1
    prompt_state = loaded["prompt_state"]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "training_mode": config.data.training_mode,
        "sample_count": len(samples),
        "label_counts": label_counts,
        "category_counts": category_counts,
        "sample_manifest_sha256": sample_manifest_sha,
        "source_manifest_sha256": source_manifest_sha,
        "training_samples": sample_records,
        "seed": config.training.seed,
        "selected_epoch": config.training.selected_epoch,
        "checkpoint_selection": config.training.checkpoint_selection,
        "source_revision": revision,
        "source_dirty": dirty,
        "anomalyclip_revision": anomalyclip_revision,
        "anomalyclip_dirty": anomalyclip_dirty,
        "clip_model_name": config.model.clip_model_name,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "pyyaml": yaml.__version__,
        },
        "config_sha256": _json_hash(resolved),
        "checkpoint": {
            "filename": checkpoint.name,
            "sha256": checkpoint_sha,
            "tensor_shapes": {key: list(value.shape) for key, value in prompt_state.items()},
            "tensor_dtypes": {key: str(value.dtype) for key, value in prompt_state.items()},
        },
        "files": {
            history_path.name: sha256_file(history_path),
            resolved_config_path.name: sha256_file(resolved_config_path),
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return DatasetRunResult(
        dataset=dataset,
        output_directory=output,
        checkpoint=checkpoint,
        sample_count=len(samples),
        history=tuple(history),
    )


def run_dataset_training(
    config: ExperimentConfig,
    dataset: str,
    *,
    model_factory: Callable[..., PublicCLIPPromptModel] = build_public_clip_prompt_model,
) -> DatasetRunResult:
    output = Path(config.artifacts.output_root).expanduser().resolve() / dataset
    expected_outputs = {
        output / f"prompts_epoch{config.training.selected_epoch}.pt",
        output / "training_history.csv",
        output / "resolved_config.yaml",
        output / "manifest.json",
    }
    existing = sorted(path for path in expected_outputs if path.exists())
    if existing and not config.artifacts.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing prompt artifacts; set "
            f"artifacts.overwrite=true only for an intentional rerun: {existing}"
        )
    samples, records, sample_sha, source_sha = prepare_training_samples(config, dataset)
    print(f"[{dataset}] prompt-training samples: {len(samples)}")
    seed_everything(config.training.seed)
    model = model_factory(config.prompt, config.model)
    try:
        history = train_prompt_model(model, samples, config)
        return save_dataset_artifacts(
            model,
            dataset=dataset,
            samples=samples,
            sample_records=records,
            sample_manifest_sha=sample_sha,
            source_manifest_sha=source_sha,
            history=history,
            config=config,
        )
    finally:
        model.release_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_dataset_inputs(config: ExperimentConfig, dataset: str) -> dict[str, object]:
    samples, _records, sample_sha, source_sha = prepare_training_samples(config, dataset)
    return {
        "dataset": dataset,
        "sample_count": len(samples),
        "normal_count": sum(sample.label == 0 for sample in samples),
        "abnormal_count": sum(sample.label == 1 for sample in samples),
        "category_count": len({sample.category for sample in samples}),
        "sample_manifest_sha256": sample_sha,
        "source_manifest_sha256": source_sha,
    }


def run_training_pipeline(
    config: ExperimentConfig,
    datasets: Sequence[str] | None = None,
) -> list[DatasetRunResult]:
    selected = tuple(datasets or config.data.datasets)
    unknown = sorted(set(selected) - set(config.data.datasets))
    if unknown:
        raise ValueError(f"Datasets are not enabled by the config: {unknown}")
    return [run_dataset_training(config, dataset) for dataset in selected]
