import numpy as np

from graph_learning.negative_sampling import (
    candidate_pairs,
    positional_node_features,
    sample_negative_pairs,
    validate_pair_labels,
)
from ldpc.parity_check import hamming_7_4


def test_uniform_and_hard_negatives_are_true_non_edges() -> None:
    h = hamming_7_4()
    positives, non_edges = candidate_pairs(h)
    sample_count = min(len(positives), len(non_edges))
    positives = positives[:sample_count]
    variables = positional_node_features(h.shape[1], 0)
    checks = positional_node_features(h.shape[0], 1)
    for strategy in ("uniform", "hard"):
        negatives = sample_negative_pairs(h, sample_count, strategy, variables, checks, 0)
        validate_pair_labels(h, positives, negatives)
        assert len(np.unique(negatives, axis=0)) == len(negatives)
