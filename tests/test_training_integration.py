from dataclasses import replace
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from object_agnostic_prompt_attack.checkpoint import load_prompt_checkpoint
from object_agnostic_prompt_attack.config import (
    ArtifactConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PromptConfig,
    TrainingConfig,
)
from object_agnostic_prompt_attack.model import PublicCLIPPromptModel
from object_agnostic_prompt_attack.training import (
    prepare_training_samples,
    run_dataset_training,
)


class MixingTextTransformer(nn.Module):
    def forward(self, value):
        return value + value.mean(dim=0, keepdim=True)


class VisualBlock(nn.Module):
    def forward(self, value):
        return value + 0.01


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([VisualBlock(), VisualBlock()])
        self.ln_post = nn.Identity()
        self.proj = nn.Parameter(torch.eye(4))


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(128, 4)
        self.transformer = MixingTextTransformer()
        self.positional_embedding = nn.Parameter(torch.zeros(10, 4))
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(4))
        self.visual = FakeVisual()

    def encode_image(self, images):
        base = images.mean(dim=(2, 3))
        base = torch.cat((base, base[:, :1]), dim=1)
        tokens = torch.stack([base, base, base, base, base], dim=0)
        for block in self.visual.transformer.resblocks:
            tokens = block(tokens)
        return tokens.permute(1, 0, 2)


def fake_tokenize(texts):
    tokens = torch.zeros(len(texts), 10, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:4] = torch.tensor([2, 3, 4])
    tokens[:, 5] = 127
    return tokens


def write_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 6, 3), value, dtype=np.uint8)).save(path)


def write_mask(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    Image.fromarray(mask).save(path)


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        dataset_root = self.root / "mvtec"
        for index in range(4):
            write_image(dataset_root / "bottle" / "test" / "good" / f"{index:03d}.png", 40 + index)
            write_image(dataset_root / "bottle" / "test" / "crack" / f"{index:03d}.png", 180 + index)
            write_mask(dataset_root / "bottle" / "ground_truth" / "crack" / f"{index:03d}_mask.png")
        self.config = ExperimentConfig(
            prompt=PromptConfig(n_ctx=3, context_length=10),
            model=ModelConfig(
                image_size=4,
                feature_layers=(1, 2),
                feature_map_indices=(0, 1),
                device="cpu",
            ),
            data=DataConfig(
                datasets=("mvtec",),
                mvtec_root=str(dataset_root),
                automatic_evaluation_fraction=0.5,
            ),
            training=TrainingConfig(
                epochs=2,
                selected_epoch=2,
                batch_size=2,
                num_workers=0,
            ),
            artifacts=ArtifactConfig(output_root=str(self.root / "artifacts")),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def model_factory(self, prompt_config, model_config):
        return PublicCLIPPromptModel(
            FakeClip(), fake_tokenize, prompt_config, model_config, torch.device("cpu")
        )

    def test_automatic_split_is_balanced_and_deterministic(self):
        first, _, first_hash, _ = prepare_training_samples(self.config, "mvtec")
        second, _, second_hash, _ = prepare_training_samples(self.config, "mvtec")
        self.assertEqual([sample.protocol_id for sample in first], [sample.protocol_id for sample in second])
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(sum(sample.label == 0 for sample in first), 2)
        self.assertEqual(sum(sample.label == 1 for sample in first), 2)

    def test_shared_prior_manifest_selects_only_requested_attack_train_rows(self):
        discovered, _, _, _ = prepare_training_samples(self.config, "mvtec")
        manifest_path = self.root / "shared_protocol.csv"
        rows = []
        manifest_samples = [
            next(sample for sample in discovered if sample.label == label)
            for label in (0, 1)
        ]
        for sample in manifest_samples:
            rows.append(
                {
                    "protocol_id": sample.protocol_id,
                    "dataset": "mvtec",
                    "category": sample.category,
                    "label": sample.label,
                    "partition": "attack_train",
                }
            )
        rows.extend(
            [
                {
                    "protocol_id": "test/visa/fake/normal/000",
                    "dataset": "visa",
                    "category": "fake",
                    "label": 0,
                    "partition": "attack_train",
                },
                {
                    "protocol_id": "test/visa/fake/anomaly/001",
                    "dataset": "visa",
                    "category": "fake",
                    "label": 1,
                    "partition": "evaluation",
                },
            ]
        )
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest_config = replace(
            self.config,
            data=replace(
                self.config.data,
                mvtec_training_manifest=str(manifest_path),
            ),
        )
        selected, _, _, source_hash = prepare_training_samples(
            manifest_config, "mvtec"
        )
        self.assertEqual({sample.dataset for sample in selected}, {"mvtec"})
        self.assertEqual({sample.label for sample in selected}, {0, 1})
        self.assertIsNotNone(source_hash)

    def test_end_to_end_training_saves_only_compact_artifacts(self):
        result = run_dataset_training(
            self.config,
            "mvtec",
            model_factory=self.model_factory,
        )
        self.assertEqual(len(result.history), 2)
        self.assertEqual(
            {path.name for path in result.output_directory.iterdir()},
            {"prompts_epoch2.pt", "training_history.csv", "resolved_config.yaml", "manifest.json"},
        )
        payload = load_prompt_checkpoint(result.checkpoint)
        self.assertEqual(set(payload["prompt_state"]), {"normal_context", "abnormal_context"})
        self.assertNotIn("clip_model", payload)
        with self.assertRaises(FileExistsError):
            run_dataset_training(
                self.config,
                "mvtec",
                model_factory=self.model_factory,
            )


if __name__ == "__main__":
    unittest.main()
