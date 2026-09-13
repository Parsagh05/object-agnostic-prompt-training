"""Scoring an arbitrary prompt pair through the training-time path.

`predictions(text_features=...)` is what lets a frozen ensemble and a learned
checkpoint be compared without either taking a different route to its score.
"""

import unittest

import torch
import torch.nn.functional as F

from object_agnostic_prompt_attack.config import ModelConfig
from object_agnostic_prompt_attack.model import PublicCLIPPromptModel


class Stub:
    """Only the attributes `predictions` reads, so no CLIP download is needed."""

    model_config = ModelConfig(feature_layers=(6, 12), feature_map_indices=(0, 1))
    device = torch.device("cpu")

    def __init__(self):
        self.encode_prompts_calls = 0

    def encode_prompts(self):
        self.encode_prompts_calls += 1
        return F.normalize(torch.ones(2, 8), dim=-1)


def inputs(batch=3, dim=8, tokens=16):
    glob = F.normalize(torch.randn(batch, dim), dim=-1)
    patches = [torch.randn(batch, 1 + tokens, dim) for _ in range(2)]
    return glob, patches


class ExternalTextScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.stub = Stub()
        self.glob, self.patches = inputs()

    def predict(self, text=None, size=(12, 12)):
        return PublicCLIPPromptModel.predictions(
            self.stub, self.glob, self.patches, size, text
        )

    def test_supplied_text_bypasses_the_learnable_contexts(self):
        self.predict(torch.randn(2, 8))
        self.assertEqual(self.stub.encode_prompts_calls, 0)

    def test_omitted_text_falls_back_to_the_learnable_contexts(self):
        self.predict()
        self.assertEqual(self.stub.encode_prompts_calls, 1)

    def test_shapes(self):
        logits, maps = self.predict(torch.randn(2, 8))
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(len(maps), 2)
        self.assertEqual(tuple(maps[0].shape), (3, 2, 12, 12))

    def test_maps_are_two_class_probabilities(self):
        _, maps = self.predict(torch.randn(2, 8))
        for layer in maps:
            self.assertTrue(
                torch.allclose(layer.sum(dim=1), torch.ones(3, 12, 12), atol=1e-5)
            )

    def test_input_is_normalised_so_scale_cannot_change_the_score(self):
        text = torch.randn(2, 8)
        a, _ = self.predict(text)
        b, _ = self.predict(F.normalize(text, dim=-1))
        c, _ = self.predict(text * 17.0)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.allclose(a, c, atol=1e-6))

    def test_row_order_is_normal_then_abnormal(self):
        """Row 1 is the abnormal channel: aligning it must raise that score."""

        text = torch.zeros(2, 8)
        text[0, 0] = 1.0          # normal prototype
        text[1] = self.glob[0]    # abnormal prototype = the first image exactly
        logits, _ = self.predict(text)
        self.assertGreater(logits[0, 1].item(), logits[0, 0].item())

    def test_rejects_anything_that_is_not_a_prompt_pair(self):
        for bad in (torch.randn(1, 8), torch.randn(3, 8), torch.randn(154, 8)):
            with self.assertRaises(ValueError):
                self.predict(bad)


if __name__ == "__main__":
    unittest.main()
