# Prompt artifact contract

This project saves prompt-training artifacts only.

## Directory structure

```text
artifacts/prompts/<split_protocol>[_train<NN>]/
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

`<split_protocol>` is `balanced` or `full`, followed by the attack-train
fraction when it is below 1.0 (`full_train25`). No two cohorts share a
directory, so a run under one cannot overwrite another's checkpoints.

## Checkpoint contents

Each `prompts_epoch15.pt` contains:

- `normal_context`;
- `abnormal_context`;
- prompt architecture metadata, including `split_protocol` and
  `attack_train_fraction`;
- dataset name;
- epoch and seed.

It must not contain the frozen CLIP backbone.

The checkpoint uses schema version `1`. The reusable tensors are stored under
`prompt_state.normal_context` and `prompt_state.abnormal_context`; the package
function `restore_prompt_checkpoint` validates and restores them into a
matching shallow prompt learner.

## Manifest contents

Each `manifest.json` records:

- dataset and training-manifest identity;
- `split_protocol`, its `label_balance_policy` string, and
  `attack_train_fraction`;
- sample-manifest SHA-256;
- prompt configuration;
- optimizer configuration;
- source-code and dependency revisions;
- selected epoch (`15`);
- checkpoint filename, tensor shapes, dtype, and SHA-256.

## Training history

`training_history.csv` contains one row per epoch with the prompt-training loss
and its image, unweighted pixel, and weighted pixel components, plus batch and
sample counts. Raw console logs and per-batch files are not part of the
deliverable.
