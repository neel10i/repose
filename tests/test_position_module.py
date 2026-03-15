"""
tests for RepoPositionModule.

verifies: output shape, gradient flow, positions are learnable (vary with input).
"""

import pytest
import torch

from src.position_module import RepoPositionModule


def test_forward_output_shape():
    """returns [batch, seq_len, 1] for given hidden_states shape."""
    batch, seq_len, hidden_dim = 2, 10, 256
    module = RepoPositionModule(hidden_dim=hidden_dim)
    hidden_states = torch.randn(batch, seq_len, hidden_dim)

    out = module(hidden_states)

    assert out.shape == (batch, seq_len, 1)


def test_forward_gradients_flow():
    """preserves gradients so module params can be trained."""
    module = RepoPositionModule(hidden_dim=128)
    hidden_states = torch.randn(1, 5, 128, requires_grad=True)

    out = module(hidden_states)
    loss = out.sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert module.position_proj.weight.grad is not None


def test_positions_differ_across_tokens():
    """different token hidden states produce different position values."""
    module = RepoPositionModule(hidden_dim=64)
    h1 = torch.randn(1, 1, 64)
    h2 = torch.randn(1, 1, 64) * 2 + 1

    z1 = module(h1).squeeze()
    z2 = module(h2).squeeze()

    assert not torch.allclose(z1, z2)


def test_positions_are_learnable_not_constant():
    """output changes when input is perturbed; not constant."""
    module = RepoPositionModule(hidden_dim=32)
    x = torch.randn(1, 4, 32)
    x_perturb = x.clone()
    x_perturb[0, 2, :] += 0.1

    z = module(x)
    z_perturb = module(x_perturb)

    assert not torch.allclose(z, z_perturb)
