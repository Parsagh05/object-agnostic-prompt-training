"""CoOp-style shallow prompts without category-specific text or deep tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .config import PromptConfig


@dataclass(frozen=True)
class PromptBatch:
    """Embedded prompts and their original CLIP token IDs.

    Row order is always ``normal, abnormal``.
    """

    embeddings: Tensor
    token_ids: Tensor


class ObjectAgnosticPromptLearner(nn.Module):
    """Learn one shallow context for normality and one for abnormality.

    ``clip_model`` must expose a CLIP-compatible ``token_embedding`` module.
    ``tokenize`` must accept a list of strings and return ``[N, L]`` token IDs.
    No class name is accepted by this API, preventing accidental object-aware
    prompt construction.
    """

    concept_names = ("normal", "abnormal")

    def __init__(
        self,
        clip_model: nn.Module,
        tokenize: Callable[[list[str]], Tensor],
        config: PromptConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PromptConfig()

        token_embedding = clip_model.token_embedding
        if not hasattr(token_embedding, "weight"):
            raise TypeError("clip_model.token_embedding must expose a weight tensor")
        context_dim = int(token_embedding.weight.shape[1])
        dtype = token_embedding.weight.dtype

        normal = torch.empty(self.config.n_ctx, context_dim, dtype=dtype)
        abnormal = torch.empty(self.config.n_ctx, context_dim, dtype=dtype)
        nn.init.normal_(normal, std=self.config.init_std)
        nn.init.normal_(abnormal, std=self.config.init_std)
        self.normal_context = nn.Parameter(normal)
        self.abnormal_context = nn.Parameter(abnormal)

        placeholder = " ".join(["X"] * self.config.n_ctx)
        prompt_text = [
            f"{placeholder} {self.config.normal_suffix.strip()}",
            f"{placeholder} {self.config.abnormal_suffix.strip()}",
        ]
        token_ids = tokenize(prompt_text).to(token_embedding.weight.device)
        if token_ids.ndim != 2 or token_ids.shape[0] != 2:
            raise ValueError("tokenize must return shape [2, context_length]")
        if token_ids.shape[1] != self.config.context_length:
            raise ValueError(
                f"tokenizer returned length {token_ids.shape[1]}, expected "
                f"{self.config.context_length}"
            )

        with torch.no_grad():
            embedded = token_embedding(token_ids).detach()
        suffix_start = 1 + self.config.n_ctx
        self.register_buffer("token_ids", token_ids)
        self.register_buffer("token_prefix", embedded[:, :1, :])
        self.register_buffer("token_suffix", embedded[:, suffix_start:, :])

    def forward(self) -> PromptBatch:
        context = torch.stack(
            (self.normal_context, self.abnormal_context), dim=0
        )
        embeddings = torch.cat(
            (self.token_prefix, context, self.token_suffix), dim=1
        )
        return PromptBatch(embeddings=embeddings, token_ids=self.token_ids)

    def prompt_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return the complete and only intended optimization parameter set."""

        return self.normal_context, self.abnormal_context
