"""Small reproducible parity-check matrices for research-scale experiments."""

from __future__ import annotations

import numpy as np


def hamming_7_4() -> np.ndarray:
    """Return a full-rank (7, 4) Hamming-code parity-check matrix."""
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def cyclic_ldpc(rows: int = 12, cols: int = 24, column_weight: int = 3) -> np.ndarray:
    """Construct a deterministic sparse binary matrix using cyclic row offsets.

    This is a compact research fixture, rather than a standards-based LDPC code.
    Each variable connects to ``column_weight`` checks. Duplicate offsets are
    rejected so every column has exactly the requested weight.
    """
    if rows < 2 or cols <= rows:
        raise ValueError("Require 2 <= rows < cols")
    if not 1 <= column_weight < rows:
        raise ValueError("column_weight must be in [1, rows)")
    h = np.zeros((rows, cols), dtype=np.uint8)
    strides = [1, 5, 7, 11, 13][:column_weight]
    if len(strides) < column_weight:
        strides.extend(range(17, 17 + column_weight - len(strides)))
    for variable in range(cols):
        used: set[int] = set()
        for layer, stride in enumerate(strides):
            check = (variable * stride + layer * (rows // column_weight + 1)) % rows
            while check in used:
                check = (check + 1) % rows
            used.add(check)
            h[check, variable] = 1
    return h


def make_parity_check(name: str = "cyclic_24_12", **kwargs: int) -> np.ndarray:
    """Build one of the supported parity-check matrices by name."""
    if name == "hamming_7_4":
        return hamming_7_4()
    if name == "cyclic_24_12":
        defaults = {"rows": 12, "cols": 24, "column_weight": 3}
        defaults.update(kwargs)
        return cyclic_ldpc(**defaults)
    if name == "cyclic_48_24":
        defaults = {"rows": 24, "cols": 48, "column_weight": 3}
        defaults.update(kwargs)
        return cyclic_ldpc(**defaults)
    raise ValueError(f"Unknown code configuration: {name}")
