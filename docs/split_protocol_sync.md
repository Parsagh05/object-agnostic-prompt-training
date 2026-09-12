# Split-protocol sync contract

Prompt training and attack generation live in two repositories
([object-agnostic-prompt-training](https://github.com/Parsagh05/object-agnostic-prompt-training)
trains the prompts,
[adversarial-perturbation-generation](https://github.com/Parsagh05/adversarial-perturbation-generation)
consumes them). Both independently derive the same train/evaluation split from
the same public test sets. If their settings disagree, prompts are fitted on
images the attack pipeline holds out for evaluation. That is contamination, and
it produces no error on either side.

Four values must agree.

| | attack pipeline | this repository | enforced? |
| --- | --- | --- | --- |
| seed | `SPLIT_SEED=111` | `training.seed: 111` | when the manifest carries `split_seed` |
| fraction | `EVALUATION_FRACTION=0.50` | `data.automatic_evaluation_fraction: 0.5` | when the manifest carries `evaluation_fraction` |
| protocol | `SPLIT_PROTOCOL` | `data.split_protocol` | when the manifest carries `label_balance_policy` |
| label policy | derived from the protocol | derived from the protocol | derived, never set by hand |
| cohort size | `ATTACK_TRAIN_FRACTION` | `data.attack_train_fraction` | no -- see below |

## How the check works

The attack pipeline stamps `split_seed`, `evaluation_fraction` and
`label_balance_policy` onto every row of `attack_train_indices.csv`. When one of
those CSVs is passed as `*_training_manifest`, `load_attack_train_manifest`
compares all three against this run's configuration and raises on a mismatch.

The policy strings are the link between the two protocol names:

| protocol | policy string |
| --- | --- |
| `balanced` | `per_dataset_category_equal_labels_v1` |
| `full` | `per_dataset_category_all_images_v1` |

## Where it is still unenforced

**No manifest supplied.** The automatic split reconstructs the protocol from the
dataset roots, so there is nothing to compare against. Seed, fraction and
protocol must be set to match the attack run by hand. This is the path the
committed epoch-15 checkpoints were produced on
(`source_manifest_sha256: null`).

**Manifests without the provenance columns** are accepted unchecked, so an older
CSV cannot be validated.

**`attack_train_fraction` is applied, not checked.** It subsets the training
half after the split, so nothing about it is recorded in the manifest rows to
compare against. Set it to the attack run's `ATTACK_TRAIN_FRACTION` by hand.
It is folded into the output directory (`full_train25`), so at least two runs
at different fractions cannot be confused for one another. Unlike the other
three, a mismatch here is not contamination -- the cohort stays inside the
training half either way -- it just means the prompts saw more images than the
perturbation did.

Supplying the attack pipeline's own `attack_train_indices.csv` is therefore the
safer route: it removes the duplicated derivation entirely and turns the
remaining agreement into a checked one.

## Cohort sizes

`ATTACK_TRAIN_FRACTION` keeps every image whose `attack_train_rank` is at most
`ceil(attack_train_stratum_size x fraction)`. Rank is position in the shuffled
order, so cohorts nest and the same rule reproduces them from either the
manifest or the automatic split. Verified image-for-image against
`select_attack_train_fraction` for 0.05/0.10/0.25/0.50/1.00 on both datasets
and both protocols.

## The failure this prevents

Under `full`, deriving the held-out count from `min(normal, abnormal)` rather
than from the images actually kept selects 1,279 MVTec images for prompt
training instead of 864 — 415 of them reserved for evaluation. The `basis`
line in `automatic_attack_train_split` exists for this, and
`tests/test_split_protocol.py` pins both totals.
