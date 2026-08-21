"""Validated configuration for per-dataset shallow prompt training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


def _section(cls, value: Mapping[str, Any] | None, name: str):
    data = dict(value or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {unknown}")
    return cls(**data)


@dataclass(frozen=True)
class PromptConfig:
    n_ctx: int = 12
    normal_suffix: str = "object."
    abnormal_suffix: str = "damaged object."
    init_std: float = 0.02
    context_length: int = 77
    category_specific: bool = False
    deep_text_prompt_tuning: bool = False

    def __post_init__(self) -> None:
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if not self.normal_suffix.strip() or not self.abnormal_suffix.strip():
            raise ValueError("normal_suffix and abnormal_suffix cannot be empty")
        if self.normal_suffix.strip() == self.abnormal_suffix.strip():
            raise ValueError("normal and abnormal suffixes must differ")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if self.context_length < self.n_ctx + 3:
            raise ValueError(
                "context_length is too short for SOS, context, suffix, and EOS"
            )
        if self.category_specific:
            raise ValueError("category-specific prompts are deferred in this phase")
        if self.deep_text_prompt_tuning:
            raise ValueError("deep text-prompt tuning is forbidden by this protocol")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "anomalyclip_public_clip"
    anomalyclip_root: str | None = None
    clip_model_name: str = "ViT-L/14@336px"
    clip_download_root: str | None = None
    image_size: int = 518
    feature_layers: tuple[int, ...] = (6, 12, 18, 24)
    feature_map_indices: tuple[int, ...] = (0, 1, 2, 3)
    temperature: float = 0.07
    use_dpam: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_layers", tuple(self.feature_layers))
        object.__setattr__(self, "feature_map_indices", tuple(self.feature_map_indices))
        if self.backend != "anomalyclip_public_clip":
            raise ValueError("backend must be 'anomalyclip_public_clip'")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not self.feature_layers or any(layer <= 0 for layer in self.feature_layers):
            raise ValueError("feature_layers must contain positive one-based indices")
        if len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError("feature_layers cannot contain duplicates")
        if not self.feature_map_indices:
            raise ValueError("feature_map_indices cannot be empty")
        if any(index < 0 or index >= len(self.feature_layers) for index in self.feature_map_indices):
            raise ValueError("feature_map_indices must index feature_layers")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.use_dpam:
            raise ValueError("DPAM changes the public CLIP visual path and is disabled")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[str, ...] = ("mvtec", "visa")
    training_mode: str = "per_source_dataset"
    mvtec_root: str | None = None
    visa_root: str | None = None
    mvtec_training_manifest: str | None = None
    visa_training_manifest: str | None = None
    automatic_evaluation_fraction: float = 0.5

    def __post_init__(self) -> None:
        datasets = tuple(str(value).lower() for value in self.datasets)
        object.__setattr__(self, "datasets", datasets)
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError("datasets must contain unique values")
        unknown = sorted(set(datasets) - {"mvtec", "visa"})
        if unknown:
            raise ValueError(f"Unsupported datasets: {unknown}")
        if self.training_mode != "per_source_dataset":
            raise ValueError("only per_source_dataset training is enabled")
        if not 0 < self.automatic_evaluation_fraction < 1:
            raise ValueError("automatic_evaluation_fraction must be between 0 and 1")

    def root_for(self, dataset: str) -> str | None:
        return self.mvtec_root if dataset == "mvtec" else self.visa_root

    def manifest_for(self, dataset: str) -> str | None:
        return (
            self.mvtec_training_manifest
            if dataset == "mvtec"
            else self.visa_training_manifest
        )


@dataclass(frozen=True)
class TrainingConfig:
    objective: str = "anomalyclip_image_ce_pixel_focal_dice"
    image_loss: str = "cross_entropy"
    pixel_losses: tuple[str, ...] = ("focal", "dice_abnormal", "dice_normal")
    pixel_loss_weight: float = 4.0
    use_pixel_masks: bool = True
    normal_mask_policy: str = "all_zero"
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.5, 0.999)
    epochs: int = 15
    batch_size: int = 8
    num_workers: int = 0
    show_progress: bool = True
    seed: int = 111
    checkpoint_selection: str = "fixed_epoch"
    selected_epoch: int = 15
    freeze_clip: bool = True
    train_prompt_parameters_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixel_losses", tuple(self.pixel_losses))
        object.__setattr__(self, "betas", tuple(float(value) for value in self.betas))
        if self.objective != "anomalyclip_image_ce_pixel_focal_dice":
            raise ValueError("the confirmed objective is the AnomalyCLIP loss")
        if self.image_loss != "cross_entropy":
            raise ValueError("image_loss must be cross_entropy")
        if self.pixel_losses != ("focal", "dice_abnormal", "dice_normal"):
            raise ValueError("pixel_losses must match the confirmed AnomalyCLIP terms")
        if self.pixel_loss_weight <= 0 or not self.use_pixel_masks:
            raise ValueError("the confirmed objective requires positively weighted masks")
        if self.normal_mask_policy != "all_zero":
            raise ValueError("normal samples must use all-zero masks")
        if self.optimizer.lower() != "adam":
            raise ValueError("optimizer must be Adam")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.epochs <= 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
        if self.checkpoint_selection != "fixed_epoch":
            raise ValueError("checkpoint_selection must be fixed_epoch")
        if self.selected_epoch != self.epochs:
            raise ValueError("selected_epoch must equal epochs for the fixed final checkpoint")
        if not self.freeze_clip or not self.train_prompt_parameters_only:
            raise ValueError("CLIP must stay frozen and only prompts may be trained")


@dataclass(frozen=True)
class ArtifactConfig:
    output_root: str = "artifacts/prompts"
    save_prompt_checkpoint: bool = True
    save_training_history_csv: bool = True
    save_resolved_config: bool = True
    save_manifest: bool = True
    save_all_epoch_checkpoints: bool = False
    save_clip_backbone: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not all((self.save_prompt_checkpoint, self.save_training_history_csv,
                    self.save_resolved_config, self.save_manifest)):
            raise ValueError("all four compact reproducibility artifacts are required")
        if self.save_all_epoch_checkpoints or self.save_clip_backbone:
            raise ValueError("per-epoch checkpoints and the frozen backbone must not be saved")


@dataclass(frozen=True)
class ExperimentConfig:
    prompt: PromptConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    artifacts: ArtifactConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        allowed = {"prompt", "model", "data", "training", "artifacts", "deferred"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown top-level configuration keys: {unknown}")
        return cls(
            prompt=_section(PromptConfig, value.get("prompt"), "prompt"),
            model=_section(ModelConfig, value.get("model"), "model"),
            data=_section(DataConfig, value.get("data"), "data"),
            training=_section(TrainingConfig, value.get("training"), "training"),
            artifacts=_section(ArtifactConfig, value.get("artifacts"), "artifacts"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return ExperimentConfig.from_mapping(value)
