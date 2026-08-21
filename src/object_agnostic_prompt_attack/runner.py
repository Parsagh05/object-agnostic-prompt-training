"""Backward-compatible imports for the prompt-training runner."""

from .training import run_dataset_training, run_training_pipeline

__all__ = ["run_dataset_training", "run_training_pipeline"]
