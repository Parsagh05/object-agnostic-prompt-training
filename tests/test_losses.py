import unittest

import torch

from object_agnostic_prompt_attack.losses import AnomalyCLIPPromptLoss


class AnomalyCLIPLossTests(unittest.TestCase):
    def test_combines_image_focal_and_both_dice_terms(self):
        image_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        masks = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 0.0]]]])
        probabilities = torch.tensor(
            [
                [[[0.9, 0.9], [0.9, 0.9]], [[0.1, 0.1], [0.1, 0.1]]],
                [[[0.1, 0.9], [0.9, 0.9]], [[0.9, 0.1], [0.1, 0.1]]],
            ],
            requires_grad=True,
        )
        result = AnomalyCLIPPromptLoss(pixel_weight=4.0)(
            image_logits, labels, [probabilities], masks
        )
        self.assertAlmostEqual(
            float(result.total.detach()),
            float((result.image + 4.0 * result.pixel).detach()),
            places=6,
        )
        result.total.backward()
        self.assertIsNotNone(image_logits.grad)
        self.assertIsNotNone(probabilities.grad)


if __name__ == "__main__":
    unittest.main()
