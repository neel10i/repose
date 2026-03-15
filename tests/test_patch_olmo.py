"""
tests for OLMo-2 patching with REPO modules.

verifies: repo_modules attached for layers >= start_layer, patched forward runs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.patch_olmo import patch_model_with_repo


class _MinimalAttention(nn.Module):
    """minimal attention with q/k/v proj and rotary_emb for testing patch.

    mimics OLMo-2 attention structure without loading the full model.
    """

    def __init__(self, hidden_size: int = 8, num_heads: int = 2, head_dim: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

        # inv_freq is the RoPE frequency tensor
        # REPO needs this to compute rotation angles from learned positions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.rotary_emb = self

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(seq, device=hidden_states.device).unsqueeze(0).expand(batch, -1)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class _MinimalLayer(nn.Module):
    """single transformer layer with self_attn.

    matches the structure expected by patch_model_with_repo.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = _MinimalAttention(hidden_size, num_heads, head_dim)


class _MinimalModel(nn.Module):
    """inner model container with layers.

    mirrors OLMo-2's model.model.layers structure.
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers: nn.ModuleList = layers


class _MinimalOlmoLike(nn.Module):
    """minimal model with model.model.layers and self_attn for patch tests.

    this avoids downloading the real 1B model during testing.
    """

    def __init__(self, num_layers: int = 2, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        head_dim = hidden_size // num_heads
        layer_list = nn.ModuleList([
            _MinimalLayer(hidden_size, num_heads, head_dim) for _ in range(num_layers)
        ])
        self.model = _MinimalModel(layer_list)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = torch.randn(1, 4, 8, device=next(self.parameters()).device)
        for layer in self.model.layers:
            assert isinstance(layer, nn.Module)
            attn = getattr(layer, "self_attn", None)
            assert isinstance(attn, nn.Module)
            x = x + attn(x, attention_mask=attention_mask)
        return type("Out", (), {"logits": x[:, :, :1].expand(1, 4, 100352)})()


def test_patch_attaches_repo_modules():
    """after patch, layers >= start_layer have self_attn.repo_modules and use_repo.

    verifies the patcher correctly injects RepoPositionModule per head
    and marks layers with use_repo=True.
    """
    model = _MinimalOlmoLike(num_layers=3, hidden_size=8, num_heads=2)
    start_layer = 1
    patched, repo_modules = patch_model_with_repo(model, start_layer=start_layer, device="cpu")

    # repo_modules should only contain entries for patched layers (1 and 2)
    assert isinstance(patched, nn.Module)
    inner = getattr(patched, "model", None)
    assert isinstance(inner, nn.Module)
    layers = getattr(inner, "layers", None)
    assert isinstance(layers, (list, nn.ModuleList))

    assert len(repo_modules) == 2
    assert 1 in repo_modules
    assert 2 in repo_modules

    # verify each patched layer has the expected structure
    for layer_idx in range(start_layer, 3):
        layer = layers[layer_idx]
        assert isinstance(layer, nn.Module)
        attn = getattr(layer, "self_attn", None)
        assert isinstance(attn, nn.Module)

        # REPO-specific attributes added by patcher
        assert getattr(attn, "use_repo", None) is True
        assert hasattr(attn, "repo_modules")
        mods = getattr(attn, "repo_modules", {})

        # one module per attention head
        assert len(mods) == 2


def test_patched_forward_runs():
    """patched model forward runs without error.

    verifies the wrapped forward function works end-to-end,
    computing learned positions and applying REPO RoPE.
    """
    model = _MinimalOlmoLike(num_layers=2, hidden_size=8, num_heads=2)
    patched, _ = patch_model_with_repo(model, start_layer=0, device="cpu")
    assert isinstance(patched, nn.Module)

    # forward should work without crashing
    out = patched(input_ids=torch.zeros(1, 4, dtype=torch.long))

    # output shape should match expected logits format
    assert hasattr(out, "logits")
    assert out.logits.shape == (1, 4, 100352)
