import torch
from torch import nn

from object_agnostic_prompt_attack import ObjectAgnosticPromptLearner, PromptConfig


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(100, 8)


def fake_tokenize(texts):
    tokens = torch.zeros(len(texts), 10, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:4] = torch.tensor([2, 3, 4])
    tokens[:, 4] = 99
    return tokens


def test_builds_two_object_agnostic_shallow_prompts():
    config = PromptConfig(n_ctx=3, context_length=10)
    learner = ObjectAgnosticPromptLearner(FakeClip(), fake_tokenize, config)

    batch = learner()

    assert batch.embeddings.shape == (2, 10, 8)
    assert batch.token_ids.shape == (2, 10)
    assert learner.concept_names == ("normal", "abnormal")
    assert set(dict(learner.named_parameters())) == {
        "normal_context",
        "abnormal_context",
    }


def test_normal_and_abnormal_contexts_are_independent():
    config = PromptConfig(n_ctx=3, context_length=10)
    learner = ObjectAgnosticPromptLearner(FakeClip(), fake_tokenize, config)

    assert learner.normal_context.data_ptr() != learner.abnormal_context.data_ptr()

