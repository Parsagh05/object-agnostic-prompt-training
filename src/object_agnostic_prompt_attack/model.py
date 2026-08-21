"""Frozen public-CLIP backend with shallow learnable text prompts."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig, PromptConfig
from .prompt_learner import ObjectAgnosticPromptLearner
from .text_encoder import ShallowTextEncoder


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def normalize_clip(images: Tensor) -> Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def load_anomalyclip_public_clip(
    config: ModelConfig,
    device: torch.device,
) -> tuple[nn.Module, Callable[[list[str]], Tensor], str]:
    if not config.anomalyclip_root:
        raise ValueError(
            "model.anomalyclip_root is required and must point to the official "
            "AnomalyCLIP repository"
        )
    root = Path(config.anomalyclip_root).expanduser().resolve()
    if not (root / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(
            f"Expected AnomalyCLIP_lib under official repository: {root}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for module_name in ("AnomalyCLIP_lib", "prompt_ensemble"):
        module = sys.modules.get(module_name)
        module_file = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_file.startswith(root_text):
            sys.modules.pop(module_name, None)
    library = importlib.import_module("AnomalyCLIP_lib")
    prompt_module = importlib.import_module("prompt_ensemble")
    load_kwargs: dict[str, object] = {"device": device}
    if config.clip_download_root:
        load_kwargs["download_root"] = str(
            Path(config.clip_download_root).expanduser().resolve()
        )
    # No design_details: this intentionally constructs ordinary public CLIP,
    # not AnomalyCLIP's compound/deep prompt transformer.
    clip_model, _ = library.load(config.clip_model_name, **load_kwargs)
    tokenize = getattr(prompt_module, "tokenize", None)
    if not callable(tokenize):
        raise AttributeError("AnomalyCLIP prompt_ensemble.py must expose tokenize")
    return clip_model, tokenize, root_text


class PublicCLIPPromptModel(nn.Module):
    """Frozen visual/text trunk plus two shallow trainable context tensors."""

    def __init__(
        self,
        clip_model: nn.Module,
        tokenize: Callable[[list[str]], Tensor],
        prompt_config: PromptConfig,
        model_config: ModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model.to(device).eval()
        self.model_config = model_config
        self.device = device
        self.clip_model.requires_grad_(False)
        self.prompt_learner = ObjectAgnosticPromptLearner(
            self.clip_model, tokenize, prompt_config
        ).to(device)
        self.text_encoder = ShallowTextEncoder(self.clip_model).to(device)
        self.text_encoder.requires_grad_(False)
        # Re-enable only the two intended shallow contexts.
        for parameter in self.prompt_learner.prompt_parameters():
            parameter.requires_grad_(True)

        visual = getattr(self.clip_model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise TypeError("public CLIP visual transformer must expose resblocks")
        self._captured: dict[int, Tensor] = {}
        self._capture_enabled = False
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer in model_config.feature_layers:
            if layer > len(blocks):
                raise ValueError(
                    f"feature layer {layer} is outside the visual transformer "
                    f"range [1, {len(blocks)}]"
                )

            def hook(_module, _inputs, output, *, layer_index=layer):
                if self._capture_enabled:
                    value = output[0] if isinstance(output, tuple) else output
                    self._captured[layer_index] = value

            self._hook_handles.append(blocks[layer - 1].register_forward_hook(hook))

    def trainable_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.prompt_learner.prompt_parameters()

    def assert_parameter_boundary(self) -> None:
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            "prompt_learner.normal_context",
            "prompt_learner.abnormal_context",
        }
        if trainable != expected:
            raise RuntimeError(
                f"Only shallow prompt contexts may be trainable; got {sorted(trainable)}"
            )

    def encode_prompts(self) -> Tensor:
        prompt_batch = self.prompt_learner()
        features = self.text_encoder(prompt_batch.embeddings, prompt_batch.token_ids).float()
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def encode_visual(self, images_01: Tensor) -> tuple[Tensor, list[Tensor]]:
        images = normalize_clip(images_01.to(self.device))
        self._captured.clear()
        self._capture_enabled = True
        try:
            visual_output = self.clip_model.encode_image(images)
        finally:
            self._capture_enabled = False
        if visual_output.ndim == 3:
            global_features = visual_output[:, 0]
        elif visual_output.ndim == 2:
            global_features = visual_output
        else:
            raise RuntimeError(
                f"Unexpected public CLIP image output shape: {tuple(visual_output.shape)}"
            )
        missing = [layer for layer in self.model_config.feature_layers if layer not in self._captured]
        if missing:
            raise RuntimeError(f"Visual hooks did not capture feature layers: {missing}")
        visual = self.clip_model.visual
        patch_features: list[Tensor] = []
        for layer in self.model_config.feature_layers:
            feature = self._captured.pop(layer)
            if feature.ndim != 3:
                raise RuntimeError(f"Visual layer {layer} did not return a 3D token tensor")
            # OpenAI CLIP block outputs are [tokens, batch, width].
            if feature.shape[1] == images.shape[0]:
                feature = feature.permute(1, 0, 2)
            elif feature.shape[0] != images.shape[0]:
                raise RuntimeError(f"Cannot identify batch dimension at visual layer {layer}")
            feature = visual.ln_post(feature)
            if visual.proj is not None:
                feature = feature @ visual.proj
            patch_features.append(feature.float())
        return F.normalize(global_features.float(), dim=-1), patch_features

    def predictions(
        self,
        global_features: Tensor,
        patch_features: Sequence[Tensor],
        output_size: tuple[int, int],
    ) -> tuple[Tensor, list[Tensor]]:
        text_features = self.encode_prompts()
        image_logits = global_features @ text_features.t() / self.model_config.temperature
        similarity_maps: list[Tensor] = []
        selected = set(self.model_config.feature_map_indices)
        for index, patch in enumerate(patch_features):
            if index not in selected:
                continue
            patch = F.normalize(patch.float(), dim=-1)
            probabilities = (patch @ text_features.t() / self.model_config.temperature).softmax(dim=-1)
            token_count = probabilities.shape[1]
            without_cls = token_count - 1
            if int(math.isqrt(without_cls)) ** 2 == without_cls:
                probabilities = probabilities[:, 1:]
                token_count = without_cls
            side = int(math.isqrt(token_count))
            if side * side != token_count:
                raise RuntimeError(f"Patch-token count is not square: {token_count}")
            similarity_map = probabilities.reshape(-1, side, side, 2).permute(0, 3, 1, 2)
            similarity_maps.append(
                F.interpolate(
                    similarity_map,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if not similarity_maps:
            raise RuntimeError("No configured feature maps were selected")
        return image_logits, similarity_maps

    def release_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._captured.clear()


def build_public_clip_prompt_model(
    prompt_config: PromptConfig,
    model_config: ModelConfig,
) -> PublicCLIPPromptModel:
    device = resolve_device(model_config.device)
    clip_model, tokenize, _ = load_anomalyclip_public_clip(model_config, device)
    model = PublicCLIPPromptModel(
        clip_model=clip_model,
        tokenize=tokenize,
        prompt_config=prompt_config,
        model_config=model_config,
        device=device,
    )
    model.assert_parameter_boundary()
    return model
