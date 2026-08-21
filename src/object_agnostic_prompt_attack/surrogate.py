"""Backward-compatible imports for the frozen public-CLIP training backend."""

from .model import PublicCLIPPromptModel, build_public_clip_prompt_model

__all__ = ["PublicCLIPPromptModel", "build_public_clip_prompt_model"]
