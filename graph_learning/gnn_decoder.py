"""A compact PyTorch Geometric Tanner-graph decoder."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import MessagePassing
except ImportError as exc:  # pragma: no cover - gives a useful import failure
    raise ImportError("TannerGNNDecoder requires torch-geometric") from exc

from ldpc.tanner_graph import TannerGraph


class TannerLayer(MessagePassing):
    """Mean aggregation followed by a residual node update."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(x_j)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregated = self.propagate(edge_index, x=x)
        return x + self.update_mlp(torch.cat((x, aggregated), dim=-1))


class TannerGNNDecoder(nn.Module):
    """Decode variable bits using the same LLR/SNR observations plus edges."""

    uses_graph_structure = True

    def __init__(self, hidden_dim: int = 32, layers: int = 3):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList(TannerLayer(hidden_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _batch_edges(edge_index: torch.Tensor, batch_size: int, nodes: int) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=edge_index.device) * nodes
        return (edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)

    def forward(
        self,
        llr: torch.Tensor,
        snr_db: torch.Tensor,
        graph: TannerGraph,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index = graph.edge_index if edge_index is None else edge_index
        edge_index = edge_index.to(llr.device)
        features = graph.node_features(llr, snr_db)
        batch_size, nodes, _ = features.shape
        x = self.input_projection(features).reshape(batch_size * nodes, -1)
        batched_edges = self._batch_edges(edge_index, batch_size, nodes)
        for layer in self.layers:
            x = layer(x, batched_edges)
        x = x.reshape(batch_size, nodes, -1)
        variables = graph.metadata["num_variables"]
        return self.output(x[:, :variables]).squeeze(-1)
