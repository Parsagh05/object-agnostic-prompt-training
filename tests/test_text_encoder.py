import torch
from torch import nn

from object_agnostic_prompt_attack import ShallowTextEncoder


class IdentityTransformer(nn.Module):
    def forward(self, value):
        return value


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = IdentityTransformer()
        self.positional_embedding = nn.Parameter(torch.zeros(10, 4))
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(4))


def test_selects_eot_embedding_without_deep_prompt_path():
    encoder = ShallowTextEncoder(FakeClip())
    embeddings = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)
    token_ids = torch.zeros(2, 10, dtype=torch.long)
    token_ids[0, 3] = 99
    token_ids[1, 5] = 99

    encoded = encoder(embeddings, token_ids)

    assert torch.equal(encoded[0], embeddings[0, 3])
    assert torch.equal(encoded[1], embeddings[1, 5])

