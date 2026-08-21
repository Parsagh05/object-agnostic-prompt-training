from pathlib import Path

import yaml


def load_config():
    path = Path(__file__).parents[1] / "configs" / "experiment.example.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_prompt_architecture_contract():
    config = load_config()
    prompt = config["prompt"]

    assert prompt["n_ctx"] == 12
    assert prompt["normal_suffix"] == "object."
    assert prompt["abnormal_suffix"] == "damaged object."
    assert prompt["category_specific"] is False
    assert prompt["deep_text_prompt_tuning"] is False


def test_per_source_dataset_training_contract():
    config = load_config()
    assert config["data"]["datasets"] == ["mvtec", "visa"]
    assert config["data"]["training_mode"] == "per_source_dataset"

    training = config["training"]
    assert training["objective"] == "anomalyclip_image_ce_pixel_focal_dice"
    assert training["image_loss"] == "cross_entropy"
    assert training["pixel_losses"] == [
        "focal",
        "dice_abnormal",
        "dice_normal",
    ]
    assert training["pixel_loss_weight"] == 4.0
    assert training["use_pixel_masks"] is True
    assert training["normal_mask_policy"] == "all_zero"
    assert training["epochs"] == 15
    assert training["checkpoint_selection"] == "fixed_epoch"
    assert training["selected_epoch"] == 15
    assert training["freeze_clip"] is True
    assert training["train_prompt_parameters_only"] is True


def test_compact_prompt_artifact_contract():
    artifacts = load_config()["artifacts"]
    assert artifacts["output_root"] == "artifacts/prompts"
    assert artifacts["save_prompt_checkpoint"] is True
    assert artifacts["save_training_history_csv"] is True
    assert artifacts["save_all_epoch_checkpoints"] is False
    assert artifacts["save_clip_backbone"] is False
