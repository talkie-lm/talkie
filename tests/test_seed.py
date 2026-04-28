import torch

from talkie.sampling import sample_gumbel


def test_sample_gumbel_with_seed_is_deterministic():
    device = torch.device("cpu")
    g1 = torch.Generator(device=device)
    g1.manual_seed(42)
    g2 = torch.Generator(device=device)
    g2.manual_seed(42)
    a = sample_gumbel((4, 8), device, generator=g1)
    b = sample_gumbel((4, 8), device, generator=g2)
    assert torch.equal(a, b)


def test_sample_gumbel_without_seed_varies():
    device = torch.device("cpu")
    a = sample_gumbel((4, 8), device)
    b = sample_gumbel((4, 8), device)
    assert not torch.equal(a, b)
