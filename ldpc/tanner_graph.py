"""Conversion of a parity-check matrix to a bipartite Tanner graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TannerGraph:
    """Tensor representation with variables first and checks second."""

    parity_check: torch.Tensor
    edge_index: torch.Tensor
    variable_indices: torch.Tensor
    check_indices: torch.Tensor
    node_type: torch.Tensor
    metadata: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())

    def node_features(self, llr: torch.Tensor, snr_db: float | torch.Tensor) -> torch.Tensor:
        """Build non-structural features: LLR, SNR, and a node-type flag.

        Check nodes have no channel observation, so their LLR feature is zero.
        No degree, adjacency statistic, or structural embedding is included.
        """
        if llr.shape[-1] != self.metadata["num_variables"]:
            raise ValueError("LLR width does not match the number of variable nodes")
        batch_shape = llr.shape[:-1]
        node_count = self.num_nodes
        features = torch.zeros(*batch_shape, node_count, 3, device=llr.device, dtype=llr.dtype)
        features[..., : llr.shape[-1], 0] = llr
        snr = torch.as_tensor(snr_db, device=llr.device, dtype=llr.dtype) / 10.0
        if snr.ndim == 0:
            features[..., :, 1] = snr
        else:
            features[..., :, 1] = snr.reshape(*batch_shape, 1)
        features[..., :, 2] = self.node_type.to(llr.device, llr.dtype)
        return features

    def to_pyg_data(self, x: torch.Tensor | None = None):
        """Create a PyG ``Data`` object without making PyG a hidden import."""
        try:
            from torch_geometric.data import Data
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError("Install torch-geometric to create PyG Data objects") from exc
        return Data(x=x, edge_index=self.edge_index, node_type=self.node_type)


def build_tanner_graph(h: np.ndarray | torch.Tensor) -> TannerGraph:
    """Create a bidirectional edge index exactly matching nonzero entries of H."""
    parity_check = torch.as_tensor(h, dtype=torch.uint8)
    if parity_check.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix")
    checks, variables = parity_check.shape
    check_rows, variable_cols = torch.nonzero(parity_check, as_tuple=True)
    check_nodes = check_rows + variables
    forward = torch.stack((variable_cols, check_nodes))
    reverse = torch.stack((check_nodes, variable_cols))
    edge_index = torch.cat((forward, reverse), dim=1).to(torch.long)
    variable_indices = torch.arange(variables, dtype=torch.long)
    check_indices = torch.arange(variables, variables + checks, dtype=torch.long)
    node_type = torch.cat((torch.zeros(variables), torch.ones(checks))).to(torch.long)
    metadata = {
        "num_variables": variables,
        "num_checks": checks,
        "num_undirected_edges": int(parity_check.sum()),
        "density": float(parity_check.float().mean()),
        "check_node_offset": variables,
    }
    return TannerGraph(parity_check, edge_index, variable_indices, check_indices, node_type, metadata)


def assert_matches_parity_check(graph: TannerGraph) -> None:
    """Raise if a Tanner graph's variable-to-check edges differ from H."""
    variables = graph.metadata["num_variables"]
    recovered = torch.zeros_like(graph.parity_check)
    source, target = graph.edge_index
    mask = (source < variables) & (target >= variables)
    recovered[target[mask] - variables, source[mask]] = 1
    if not torch.equal(recovered, graph.parity_check):
        raise AssertionError("Tanner graph does not correspond to its parity-check matrix")
