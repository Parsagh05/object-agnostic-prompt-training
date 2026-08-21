"""AnomalyCLIP image and mask-supervised prompt losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ProbabilityFocalLoss(nn.Module):
    """AnomalyCLIP-compatible focal loss for two-channel probabilities."""

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, probabilities: Tensor, target: Tensor) -> Tensor:
        if probabilities.ndim != 4 or probabilities.shape[1] != 2:
            raise ValueError("probabilities must have shape [B, 2, H, W]")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.shape != probabilities.shape[:1] + probabilities.shape[2:]:
            raise ValueError("target must have shape [B, H, W]")
        labels = target.long()
        one_hot = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).to(probabilities.dtype)
        one_hot = one_hot.clamp(self.smooth, 1.0 - self.smooth)
        # Match AnomalyCLIP's reference implementation, including the small
        # additive smoothing term after selecting the target probability.
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        pt = pt.clamp_min(self.smooth)
        return (-((1.0 - pt) ** self.gamma) * pt.log()).mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target Dice tensors must have equal shapes")
        batch = target.shape[0]
        prediction = prediction.reshape(batch, -1)
        target = target.reshape(batch, -1)
        intersection = (prediction * target).sum(dim=1)
        score = (2 * intersection + self.smooth) / (
            prediction.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - score.mean()


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    image: Tensor
    pixel: Tensor
    pixel_weighted: Tensor


class AnomalyCLIPPromptLoss(nn.Module):
    def __init__(self, pixel_weight: float = 4.0) -> None:
        super().__init__()
        self.pixel_weight = pixel_weight
        self.focal = ProbabilityFocalLoss()
        self.dice = BinaryDiceLoss()

    def forward(
        self,
        image_logits: Tensor,
        labels: Tensor,
        similarity_maps: list[Tensor],
        masks: Tensor,
    ) -> LossBreakdown:
        if not similarity_maps:
            raise ValueError("at least one patch similarity map is required")
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        masks = (masks > 0.5).to(image_logits.dtype)
        image_loss = F.cross_entropy(image_logits, labels.long())
        pixel_loss = image_logits.new_zeros(())
        for similarity_map in similarity_maps:
            pixel_loss = pixel_loss + self.focal(similarity_map, masks)
            pixel_loss = pixel_loss + self.dice(similarity_map[:, 1], masks)
            pixel_loss = pixel_loss + self.dice(similarity_map[:, 0], 1.0 - masks)
        weighted = self.pixel_weight * pixel_loss
        return LossBreakdown(
            total=image_loss + weighted,
            image=image_loss,
            pixel=pixel_loss,
            pixel_weighted=weighted,
        )
