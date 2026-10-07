"""Feature-only decoder that never receives Tanner-graph connectivity."""

from __future__ import annotations

import torch
from torch import nn


class FeatureOnlyMLPDecoder(nn.Module):
    """Decode an entire received word from LLRs and SNR only.

    Flattening the word gives this baseline access to every channel observation
    and bit position, making it stronger than an independent per-bit MLP. Its
    API intentionally has no adjacency or edge argument.
    """

    uses_graph_structure = False

    def __init__(self, code_length: int, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        modules: list[nn.Module] = [nn.Linear(code_length + 1, hidden_dim), nn.GELU()]
        for _ in range(layers - 2):
            modules.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        modules.append(nn.Linear(hidden_dim, code_length))
        self.network = nn.Sequential(*modules)

    def forward(self, llr: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        scaled_snr = snr_db.reshape(llr.shape[0], 1) / 10.0
        return self.network(torch.cat((llr, scaled_snr), dim=1))
