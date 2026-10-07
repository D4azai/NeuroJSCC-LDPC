"""Controlled bipartite Tanner-graph structural ablations."""

from __future__ import annotations

import numpy as np


def perturb_parity_check(h: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Rewire a fraction of edges using degree-preserving bipartite swaps.

    ``fraction=1`` requests extensive randomization while retaining both degree
    sequences and edge count. A perturbed matrix is a structural ablation and is
    not assumed to define the code that generated evaluation words.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    h = np.asarray(h, dtype=np.uint8)
    result = h.copy()
    if fraction == 0.0:
        return result
    rng = np.random.default_rng(seed)
    edge_count = int(h.sum())
    target_removed = max(1, int(round(edge_count * fraction)))
    successful_swaps = 0
    attempts = 0
    randomization_swaps = edge_count * 10
    max_attempts = max(1000, edge_count * 1000)
    while attempts < max_attempts:
        removed_before = int(np.logical_and(h == 1, result == 0).sum())
        if fraction < 1.0 and removed_before >= target_removed:
            break
        if fraction == 1.0 and successful_swaps >= randomization_swaps:
            break
        edges = np.argwhere(result == 1)
        first, second = edges[rng.choice(len(edges), size=2, replace=False)]
        c1, v1 = map(int, first)
        c2, v2 = map(int, second)
        attempts += 1
        if c1 == c2 or v1 == v2 or result[c1, v2] or result[c2, v1]:
            continue
        result[c1, v1] = result[c2, v2] = 0
        result[c1, v2] = result[c2, v1] = 1
        removed_after = int(np.logical_and(h == 1, result == 0).sum())
        if fraction < 1.0 and removed_after <= removed_before:
            result[c1, v1] = result[c2, v2] = 1
            result[c1, v2] = result[c2, v1] = 0
            continue
        successful_swaps += 1
    achieved = edge_change_fraction(h, result)
    if fraction < 1.0 and achieved + 1.0 / edge_count < fraction:
        raise RuntimeError("Could not achieve requested rewiring fraction")
    return result


def edge_change_fraction(original: np.ndarray, perturbed: np.ndarray) -> float:
    """Fraction of original edges that are absent after perturbation."""
    removed = np.logical_and(original == 1, perturbed == 0).sum()
    return float(removed / max(int(np.asarray(original).sum()), 1))
