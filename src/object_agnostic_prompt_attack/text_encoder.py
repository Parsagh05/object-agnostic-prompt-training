"""Ordinary CLIP text encoding for already embedded shallow prompts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ShallowTextEncoder(nn.Module):
    """Use CLIP's existing text path without transformer-layer prompt injection."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompt_embeddings: Tensor, token_ids: Tensor) -> Tensor:
        if prompt_embeddings.ndim != 3:
            raise ValueError("prompt_embeddings must have shape [N, L, D]")
        if token_ids.shape != prompt_embeddings.shape[:2]:
            raise ValueError("token_ids must match the first two prompt dimensions")

        length = prompt_embeddings.shape[1]
        positional = self.positional_embedding[:length].to(prompt_embeddings.dtype)
        x = prompt_embeddings + positional
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # OpenAI CLIP assigns the EOT token the largest token ID.
        eot_positions = token_ids.argmax(dim=-1)
        row = torch.arange(x.shape[0], device=x.device)
        projection = self.text_projection
        return x[row, eot_positions] @ projection

