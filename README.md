# Object-Agnostic Learnable-Prompt Training Pipeline

This project has one responsibility: train shallow object-agnostic textual
prompts and save compact prompt-only checkpoints for later use elsewhere. The
pipeline ends after validating and packaging those checkpoints.

The implementation is complete and executable. It uses the official
AnomalyCLIP repository to load ordinary public CLIP, but it does not enable
AnomalyCLIP's compound text transformer or DPAM visual surgery.

## Prompt architecture

Each source dataset receives one normal prompt and one abnormal prompt:

```text
[SOS] [N1] ... [N12] object.         [EOS] [PAD] ...
[SOS] [A1] ... [A12] damaged object. [EOS] [PAD] ...
```

- `N*` and `A*` are learnable embedding parameters.
- Category names are never inserted.
- The fixed suffixes are `"object."` and `"damaged object."`.
- Contexts are initialized from a normal distribution with `std=0.02`.
- Prompt learning is shallow: no tokens are injected into internal text
  transformer layers.
- CLIP remains frozen. Only the normal and abnormal context tensors update.

## Training scope

The current phase uses per-source-dataset prompt training only:

```text
MVTec training data -> mvtec_prompts_epoch15.pt
VisA training data  -> visa_prompts_epoch15.pt
```

Joint-dataset and per-category prompt training are deferred until these
per-dataset prompt results are reviewed.

Training runs for 15 epochs and always retains the final epoch-15 checkpoint,
matching the earlier project convention.

## Training objective

Prompt training uses the AnomalyCLIP loss: image-level normal/abnormal
cross-entropy plus mask-supervised pixel-level focal loss, abnormal-region
Dice loss, and normal-region Dice loss. The combined pixel objective has
weight `4`. Abnormal samples use their dataset defect masks; normal samples use
all-zero masks.

Only the two shallow prompt tensors are optimized. AnomalyCLIP's deep or
compound text-prompt tuning is explicitly disabled.

Patch features are read from frozen public-CLIP transformer layers
`6, 12, 18, 24` using forward hooks. Their normal/abnormal similarity maps
provide the pixel supervision without modifying the visual architecture.

## Data protocol

For exact parity with the earlier experiment, point each
`*_training_manifest` setting at its existing `attack_train_indices.csv`.
Only rows whose partition is `attack_train` are accepted; evaluation rows are
never used for prompt fitting. A shared protocol CSV containing both datasets
may be supplied to both options; each run selects only its own dataset rows.

If no training manifest is supplied, the pipeline reconstructs the earlier
deterministic protocol from each dataset's labeled test set. Within every
category it balances normal and abnormal counts, shuffles each label stratum
with seed `111`, reserves 50% for evaluation, and trains only on the remaining
half. This fallback is necessary because the official MVTec training split has
no anomalous images or defect masks.

## Installation and use

Install this project and the requirements of the official AnomalyCLIP checkout:

```bash
python -m pip install -e .
```

First validate the dataset roots and selected training cohorts without loading
CLIP:

```bash
train-object-agnostic-prompts \
  --config configs/experiment.example.yaml \
  --datasets all \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --visa-root /path/to/VisA_20220922 \
  --mvtec-training-manifest /path/to/mvtec/attack_train_indices.csv \
  --visa-training-manifest /path/to/visa/attack_train_indices.csv \
  --validate-only
```

Then train and package both independent prompt pairs:

```bash
train-object-agnostic-prompts \
  --config configs/experiment.example.yaml \
  --datasets all \
  --anomalyclip-root /path/to/AnomalyCLIP \
  --clip-download-root /path/to/clip-cache \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --visa-root /path/to/VisA_20220922 \
  --mvtec-training-manifest /path/to/mvtec/attack_train_indices.csv \
  --visa-training-manifest /path/to/visa/attack_train_indices.csv \
  --output-root artifacts/prompts \
  --device cuda
```

Use `--datasets mvtec` or `--datasets visa` for a single run. All configuration
values can instead be written directly into the YAML file. Existing artifacts
are protected by default; pass `--overwrite` only for an intentional rerun.

## Saved artifacts

Each dataset writes one compact directory:

```text
artifacts/prompts/
|-- mvtec/
|   |-- prompts_epoch15.pt
|   |-- training_history.csv
|   |-- resolved_config.yaml
|   `-- manifest.json
`-- visa/
    |-- prompts_epoch15.pt
    |-- training_history.csv
    |-- resolved_config.yaml
    `-- manifest.json
```

The checkpoint contains prompt tensors only. It does not duplicate the frozen
CLIP backbone. The manifest records dataset identity, sample-manifest hash,
seed, source revision, configuration hash, and checkpoint checksum.

## Project layout

```text
object_agnostic_prompt_attack_pipeline/
|-- configs/experiment.example.yaml
|-- docs/
|   |-- implementation_plan.md
|   |-- questions_for_alireza.md
|   `-- artifact_contract.md
|-- scripts/train_prompts.py
|-- src/object_agnostic_prompt_attack/
|   |-- data.py
|   |-- losses.py
|   |-- model.py
|   |-- config.py
|   |-- prompt_learner.py
|   |-- text_encoder.py
|   |-- checkpoint.py
|   `-- training.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

## References

- Previous project: https://github.com/Parsagh05/adversarial-robustness
- CoOp: https://github.com/KaiyangZhou/CoOp
- AnomalyCLIP: https://github.com/zqhang/AnomalyCLIP
