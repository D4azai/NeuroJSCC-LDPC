import numpy as np
import torch

from ldpc.parity_check import hamming_7_4
from ldpc.tanner_graph import assert_matches_parity_check, build_tanner_graph


def test_tanner_graph_exactly_matches_h() -> None:
    h = hamming_7_4()
    graph = build_tanner_graph(h)
    assert_matches_parity_check(graph)
    assert graph.metadata["num_variables"] == 7
    assert graph.metadata["num_checks"] == 3
    assert graph.edge_index.shape == (2, 2 * int(h.sum()))
    assert torch.equal(graph.variable_indices, torch.arange(7))
    assert torch.equal(graph.check_indices, torch.arange(7, 10))


def test_node_features_contain_no_degree_or_adjacency_columns() -> None:
    graph = build_tanner_graph(hamming_7_4())
    features = graph.node_features(torch.ones(2, 7), torch.full((2, 1), 2.0))
    assert features.shape == (2, 10, 3)
    assert torch.all(features[:, 7:, 0] == 0)
    assert np.allclose(features[:, :, 1].numpy(), 0.2)
