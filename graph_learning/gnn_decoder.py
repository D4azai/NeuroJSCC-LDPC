"""A compact PyTorch Geometric Tanner-graph decoder."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.utils import to_dense_adj
except ImportError as exc:  # pragma: no cover - gives a useful import failure
    raise ImportError("TannerGNNDecoder requires torch-geometric") from exc

from ldpc.tanner_graph import TannerGraph


class TannerLayer(nn.Module):
    """Dense mean aggregation followed by a residual node update.

    Tanner graphs in this project contain only tens of nodes. A shared dense
    adjacency multiplication avoids materializing one disconnected PyG graph per
    sample and is substantially faster on CPU than scatter-based mini-batching.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        messages = self.message_mlp(x)
        aggregated = torch.einsum("ij,bjh->bih", adjacency, messages)
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
    def _normalized_adjacency(
        edge_index: torch.Tensor, nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        # PyG stores A[source, target]. Transpose so rows index destinations,
        # then normalize rows to reproduce mean message aggregation exactly.
        adjacency = to_dense_adj(edge_index, max_num_nodes=nodes)[0].T.to(dtype=dtype)
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

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
        x = self.input_projection(features)
        adjacency = self._normalized_adjacency(edge_index, nodes, x.dtype)
        for layer in self.layers:
            x = layer(x, adjacency)
        variables = graph.metadata["num_variables"]
        return self.output(x[:, :variables]).squeeze(-1)
