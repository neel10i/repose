"""
REPO position prediction module.

maps hidden states to scalar positions via SwiGLU + linear.
used per attention head; output z_i feeds into RoPE as (z_j - z_i).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepoPositionModule(nn.Module):
    """
    lightweight module that assigns a scalar position z_i to each token from its hidden state.

    architecture: hidden_state -> SwiGLU (position representation) -> linear -> scalar z.
    """

    def __init__(self, hidden_dim: int, position_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_dim = position_dim if position_dim is not None else hidden_dim // 4

        self.gate_proj = nn.Linear(hidden_dim, self.position_dim, bias=False)
        self.content_proj = nn.Linear(hidden_dim, self.position_dim, bias=False)
        self.position_proj = nn.Linear(self.position_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        args:
            hidden_states: [batch, seq_len, hidden_dim]
        returns:
            positions: [batch, seq_len, 1]
        """
        gate = F.silu(self.gate_proj(hidden_states))
        content = self.content_proj(hidden_states)
        rep = gate * content
        return self.position_proj(rep)
