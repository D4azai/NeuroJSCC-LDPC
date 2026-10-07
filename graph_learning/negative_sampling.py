"""Leakage-aware non-edge sampling for Tanner edge diagnostics."""

from __future__ import annotations

import numpy as np


def positional_node_features(count: int, node_type: int, dimensions: int = 6) -> np.ndarray:
    """Return graph-independent identity/position attributes in a common space.

    The normalized index and Fourier coordinates describe node labels only. They
    do not use degree, neighborhoods, or any value computed from existing edges.
    """
    if dimensions < 6:
        raise ValueError("dimensions must be at least 6")
    position = np.arange(count, dtype=np.float64) / max(count - 1, 1)
    features = np.zeros((count, dimensions), dtype=np.float64)
    features[:, 0] = position
    features[:, 1] = np.sin(2 * np.pi * position)
    features[:, 2] = np.cos(2 * np.pi * position)
    features[:, 3] = np.sin(4 * np.pi * position)
    features[:, 4] = np.cos(4 * np.pi * position)
    features[:, 5] = float(node_type)
    return features


def candidate_pairs(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = np.argwhere(h == 1)
    negatives = np.argwhere(h == 0)
    return positives[:, [1, 0]], negatives[:, [1, 0]]  # variable, check


def sample_negative_pairs(
    h: np.ndarray,
    count: int,
    strategy: str,
    variable_features: np.ndarray,
    check_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample uniform non-edges or the most feature-similar non-edges."""
    _, non_edges = candidate_pairs(h)
    if count > len(non_edges):
        raise ValueError("Requested more negative pairs than available non-edges")
    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        return non_edges[rng.choice(len(non_edges), size=count, replace=False)]
    if strategy == "hard":
        variable = variable_features[non_edges[:, 0], :5]
        check = check_features[non_edges[:, 1], :5]
        similarity = np.sum(variable * check, axis=1) / (
            np.linalg.norm(variable, axis=1) * np.linalg.norm(check, axis=1) + 1e-12
        )
        jitter = rng.uniform(0.0, 1e-10, len(non_edges))
        return non_edges[np.argsort(-(similarity + jitter))[:count]]
    raise ValueError(f"Unknown negative-sampling strategy: {strategy}")


def validate_pair_labels(h: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> None:
    if not np.all(h[positives[:, 1], positives[:, 0]] == 1):
        raise AssertionError("Positive edge sample contains a non-edge")
    if not np.all(h[negatives[:, 1], negatives[:, 0]] == 0):
        raise AssertionError("Negative edge sample contains an edge")
