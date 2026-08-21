# Implemented prompt-training pipeline

## Phase 1: configure confirmed training supervision

- Use AnomalyCLIP's image-level normal/abnormal cross-entropy plus pixel-level
  focal and Dice losses from defect masks. Use all-zero masks for normal
  samples and weight the combined pixel objective by `4`.
- Keep the architecture fixed: two shallow object-agnostic contexts, 12 tokens,
  random initialization with `std=0.02`, and the suffixes `"object."` and
  `"damaged object."`.
- Use per-source-dataset training only.

## Phase 2: prepare training data

- Build one deterministic MVTec training manifest and one VisA training
  manifest.
- Record dataset roots, sample IDs, labels, mask availability, seed, and
  manifest hashes.
- Do not mix the two datasets in this phase.

## Phase 3: train prompts

- Load the same public CLIP implementation and image preprocessing expected by
  the downstream consumer.
- Extract intermediate frozen public-CLIP patch tokens with forward hooks; do
  not apply DPAM visual surgery.
- Freeze the complete CLIP model.
- Enable gradients only for `normal_context` and `abnormal_context`.
- Do not enable AnomalyCLIP's deep or compound text-prompt tuning.
- Train one prompt pair on MVTec and one on VisA.
- Run exactly 15 epochs and retain the final epoch-15 state.
- Record one row per epoch in `training_history.csv`.

## Phase 4: save compact artifacts

For each dataset, save:

- one prompt-only epoch-15 checkpoint;
- one resolved configuration;
- one training-history CSV;
- one manifest containing revisions, sample hashes, seed, tensor shapes, dtype,
  and checkpoint checksum.

Keep the deliverable compact: omit the CLIP backbone, per-epoch optimizer
snapshots, raw console logs, and caches.

## Verification

- Validate prompt and token shapes.
- Verify normal and abnormal contexts are independent.
- Prove that only prompt tensors receive gradients and optimizer updates.
- Verify checkpoint round-trip equality.
- Verify deterministic initialization and data ordering from seed `111`.
- Validate manifest and checksum integrity.

All phases above are implemented by `scripts/train_prompts.py` and the
`train-object-agnostic-prompts` installed command. `--validate-only` performs
data/manifest preflight without loading CLIP.
