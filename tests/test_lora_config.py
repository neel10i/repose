"""
tests for LoRA configuration with REPO.

verifies: lora config targets attention projections, repo modules train fully.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.lora_config import get_repo_lora_config
from src.patch_olmo import patch_model_with_repo


class _MinimalAttention(nn.Module):
    """minimal attention for testing lora + repo interaction."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.rotary_emb = self

    def forward(self, hidden_states, **kwargs):
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return self.o_proj(v)


class _MinimalLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self_attn = _MinimalAttention(hidden_size, num_heads)


class _MinimalModel(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers


class _MinimalOlmoLike(nn.Module):
    """minimal model for lora + repo tests."""

    def __init__(self, num_layers: int = 2, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        layer_list = nn.ModuleList([
            _MinimalLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.model = _MinimalModel(layer_list)


def test_lora_config_targets_attention():
    """lora config targets q/k/v/o projections."""
    from collections.abc import Collection

    config = get_repo_lora_config()

    # target_modules is stored as a set internally
    targets = config.target_modules
    assert isinstance(targets, Collection)

    # verify attention projections are targeted
    assert "q_proj" in targets
    assert "k_proj" in targets
    assert "v_proj" in targets
    assert "o_proj" in targets


def test_lora_config_is_causal_lm():
    """lora config is for causal language modeling."""
    from peft import TaskType

    config = get_repo_lora_config()
    assert config.task_type == TaskType.CAUSAL_LM


def test_lora_config_excludes_repo_modules():
    """repo module names are not in lora target_modules (they train fully)."""
    from collections.abc import Collection

    config = get_repo_lora_config()

    # target_modules is stored as a set internally
    targets = config.target_modules
    assert isinstance(targets, Collection)

    # repo modules have names like "repo_modules.0.gate_proj"
    # these should not match any target_modules patterns
    repo_patterns = ["repo_modules", "gate_proj", "content_proj", "position_proj"]
    for pattern in repo_patterns:
        assert pattern not in targets
