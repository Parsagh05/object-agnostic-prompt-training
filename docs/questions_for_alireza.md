# Prompt-training decisions

This project only trains, validates, and saves shallow object-agnostic prompt
checkpoints.

## Confirmed

- Train one prompt pair per source dataset: MVTec and VisA.
- Each pair contains separate normal and abnormal shallow context tensors.
- Use 12 context tokens with random initialization (`std=0.02`).
- Use fixed suffixes `"object."` and `"damaged object."`.
- Do not insert category names.
- Do not use deep or compound prompt tokens inside text-transformer layers.
- Keep the CLIP image encoder and text encoder frozen; update prompt tensors
  only.
- Keep the previous public-CLIP visual path and do not apply DPAM visual
  surgery; frozen intermediate patch tokens provide pixel supervision.
- Use the AnomalyCLIP prompt-training objective: image-level normal/abnormal
  cross-entropy plus mask-supervised pixel-level focal loss, abnormal-region
  Dice loss, and normal-region Dice loss. Weight the combined pixel objective
  by `4`, matching AnomalyCLIP.
- Use dataset defect masks for abnormal samples and an all-zero mask for normal
  samples.
- Train for 15 epochs and save the fixed final epoch-15 prompt checkpoint.
- Save one compact training history and reproducibility manifest per dataset.
- Joint-dataset and per-category prompt training are deferred.

There are no remaining design confirmations for the current per-dataset
prompt-training phase.
