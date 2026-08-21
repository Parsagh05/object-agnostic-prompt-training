import torch
from torch import nn

from object_agnostic_prompt_attack import ObjectAgnosticPromptLearner, PromptConfig


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(100, 8)
        self.visual_weight = nn.Parameter(torch.ones(1))


def fake_tokenize(texts):
    tokens = torch.zeros(len(texts), 10, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 4] = 99
    return tokens


def test_optimizer_can_be_restricted_to_prompt_parameters():
    clip = FakeClip().requires_grad_(False)
    learner = ObjectAgnosticPromptLearner(
        clip, fake_tokenize, PromptConfig(n_ctx=3, context_length=10)
    )
    optimizer_params = list(learner.prompt_parameters())

    assert all(parameter.requires_grad for parameter in optimizer_params)
    assert all(not parameter.requires_grad for parameter in clip.parameters())
    assert sum(parameter.numel() for parameter in optimizer_params) == 2 * 3 * 8

