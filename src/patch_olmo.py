"""
monkey-patch causal LM to inject REPO position modules.

strategy: for layers >= start_layer, inject RepoPositionModule per head,
wrap attention forward to compute z and apply RoPE with z_j - z_i.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.position_module import RepoPositionModule

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any


def patch_olmo2_with_repo(
    model_name: str = "allenai/OLMo-2-0425-1B",
    start_layer: int = 5,
    position_dim: int | None = None,
    device: str = "cuda",
) -> tuple[nn.Module, dict[int, dict[int, RepoPositionModule]]]:
    """
    loads OLMo-2 from HuggingFace and patches it with REPO.

    returns:
        (model, repo_modules). use device="cpu" for testing without GPU.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    return patch_model_with_repo(model, start_layer, position_dim, device)


def patch_model_with_repo(
    model: nn.Module,
    start_layer: int = 5,
    position_dim: int | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[int, dict[int, RepoPositionModule]]]:
    """
    patches an already-loaded model (with model.model.layers and .self_attn) with REPO.

    returns:
        (model, repo_modules) with repo_modules[layer_idx][head_idx] = RepoPositionModule.
    """
    repo_modules: dict[int, dict[int, RepoPositionModule]] = {}

    # REPO only modifies upper layers (default: layer 5+)
    # lower layers keep standard RoPE to preserve local pattern extraction
    inner = getattr(model, "model", None)
    if inner is None:
        return model, repo_modules
    layers = getattr(inner, "layers", None)
    if layers is None or not isinstance(layers, (list, nn.ModuleList)):
        return model, repo_modules

    for layer_idx, layer in enumerate(layers):
        if not isinstance(layer, nn.Module):
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None or not isinstance(attn, nn.Module):
            continue
        if layer_idx < start_layer:
            continue

        # each attention head gets its own position predictor
        # paper shows REPO works best with per-head modules (not shared)
        num_heads: int = getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 1))
        if num_heads is None:
            num_heads = 1
        if hasattr(attn, "q_proj") and isinstance(attn.q_proj, nn.Module):
            hidden = attn.q_proj.out_features
            num_heads = getattr(attn, "num_heads", 1) or num_heads
        hidden_dim: int = getattr(attn, "hidden_size", 0) or (
            getattr(attn.q_proj, "in_features", 256) if hasattr(attn, "q_proj") else 256
        )
        head_dim: int = getattr(attn, "head_dim", 0) or (hidden_dim // num_heads if num_heads else 64)

        layer_repo: dict[int, RepoPositionModule] = {}
        for head_idx in range(int(num_heads)):
            layer_repo[head_idx] = RepoPositionModule(int(hidden_dim), position_dim).to(device)
        repo_modules[layer_idx] = layer_repo
        attn.repo_modules = layer_repo  # type: ignore[attr-defined]
        attn.use_repo = True  # type: ignore[attr-defined]

        # find the rotary embedding module to extract inv_freq
        # different model variants store it under different names
        rotary_emb: nn.Module = getattr(attn, "rotary_emb", None) or getattr(attn, "rope", attn)
        if not hasattr(rotary_emb, "inv_freq") and hasattr(attn, "inv_freq"):
            rotary_emb = attn
        _wrap_attention_forward(attn, rotary_emb)

    return model, repo_modules


def _wrap_attention_forward(attn_module: nn.Module, rotary_emb: nn.Module) -> None:
    """wraps attention forward to use learned positions z for RoPE."""

    def repo_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert isinstance(hidden_states, torch.Tensor)
        batch_size, seq_len, _ = hidden_states.shape

        # compute learned position z_i for each token per head
        # z will drive RoPE instead of fixed indices
        repo_modules: MutableMapping[int, RepoPositionModule] = getattr(attn_module, "repo_modules", {})
        z_list = [repo_modules[h](hidden_states) for h in repo_modules]
        z = torch.cat(z_list, dim=-1)

        q: torch.Tensor = attn_module.q_proj(hidden_states)  # type: ignore[operator]
        k: torch.Tensor = attn_module.k_proj(hidden_states)  # type: ignore[operator]
        v: torch.Tensor = attn_module.v_proj(hidden_states)  # type: ignore[operator]

        num_heads: int = getattr(attn_module, "num_heads", getattr(attn_module, "num_attention_heads", 1)) or 1
        head_dim: int = getattr(attn_module, "head_dim", q.shape[-1] // num_heads)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # apply REPO-modified RoPE: uses (z_j - z_i) instead of (j - i)
        # this is the core of the REPO mechanism
        inv_freq: torch.Tensor | None = getattr(rotary_emb, "inv_freq", None)
        if inv_freq is not None:
            q, k = _apply_repo_rope(q, k, z, inv_freq)

        scale = float(head_dim) ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        if hasattr(attn_module, "o_proj"):
            out = attn_module.o_proj(out)  # type: ignore[operator]
        return out

    attn_module.forward = repo_forward  # type: ignore[method-assign]


def _apply_repo_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    applies RoPE using learned positions z (per head).

    q, k: [batch, num_heads, seq_len, head_dim]
    z: [batch, seq_len, num_heads]
    inv_freq: [head_dim//2]
    """
    z_per_head = z.transpose(1, 2)
    inv_freq = inv_freq.to(z.device)

    # angles are computed from learned position differences
    # standard RoPE uses (position_j - position_i) * inv_freq
    # REPO uses (z_j - z_i) * inv_freq where z is learned per token
    angles = z_per_head.unsqueeze(-1) * inv_freq
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)

    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        if cos.shape[-1] != x_even.shape[-1]:
            return x
        # standard RoPE rotation on even/odd pairs
        return torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
            dim=-1,
        ).flatten(-2)

    q = _rotate(q, cos_t, sin_t)
    k = _rotate(k, cos_t, sin_t)
    return q, k
