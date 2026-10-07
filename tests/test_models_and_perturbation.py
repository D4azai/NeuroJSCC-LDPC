import inspect

import numpy as np
import torch

from graph_learning.gnn_decoder import TannerGNNDecoder
from graph_learning.mlp_decoder import FeatureOnlyMLPDecoder
from graph_learning.perturbation import edge_change_fraction, perturb_parity_check
from ldpc.parity_check import cyclic_ldpc
from ldpc.tanner_graph import build_tanner_graph


def test_mlp_api_has_no_connectivity_and_models_output_bits() -> None:
    h = cyclic_ldpc(6, 12, 2)
    graph = build_tanner_graph(h)
    mlp = FeatureOnlyMLPDecoder(12, hidden_dim=16, layers=2)
    gnn = TannerGNNDecoder(hidden_dim=8, layers=2)
    assert list(inspect.signature(mlp.forward).parameters) == ["llr", "snr_db"]
    llr = torch.randn(3, 12)
    snr = torch.full((3, 1), 2.0)
    assert mlp(llr, snr).shape == (3, 12)
    assert gnn(llr, snr, graph).shape == (3, 12)


def test_rewiring_preserves_bipartite_degrees_and_changes_edges() -> None:
    h = cyclic_ldpc(8, 16, 3)
    perturbed = perturb_parity_check(h, 0.25, seed=1)
    assert np.array_equal(h.sum(axis=0), perturbed.sum(axis=0))
    assert np.array_equal(h.sum(axis=1), perturbed.sum(axis=1))
    assert int(h.sum()) == int(perturbed.sum())
    assert edge_change_fraction(h, perturbed) >= 0.25
