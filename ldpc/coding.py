"""GF(2) utilities and binary linear-code sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compute reduced row-echelon form over GF(2)."""
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    pivots: list[int] = []
    row = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        for other in range(a.shape[0]):
            if other != row and a[other, col]:
                a[other] ^= a[row]
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots


def nullspace_generator(h: np.ndarray) -> np.ndarray:
    """Return rows spanning the null space of ``h`` over GF(2)."""
    rref, pivots = gf2_rref(h)
    free = [col for col in range(h.shape[1]) if col not in pivots]
    if not free:
        raise ValueError("Parity-check matrix has a zero-dimensional null space")
    basis = np.zeros((len(free), h.shape[1]), dtype=np.uint8)
    for index, free_col in enumerate(free):
        basis[index, free_col] = 1
        for row, pivot_col in enumerate(pivots):
            basis[index, pivot_col] = rref[row, free_col]
    if np.any((np.asarray(h, dtype=np.uint8) @ basis.T) % 2):
        raise RuntimeError("Failed to construct a valid GF(2) generator")
    return basis


@dataclass(frozen=True)
class BinaryLinearCode:
    """A binary linear code represented by parity-check and generator matrices."""

    h: np.ndarray
    g: np.ndarray

    @classmethod
    def from_parity_check(cls, h: np.ndarray) -> "BinaryLinearCode":
        h = np.asarray(h, dtype=np.uint8) % 2
        return cls(h=h, g=nullspace_generator(h))

    @property
    def n(self) -> int:
        return int(self.h.shape[1])

    @property
    def k(self) -> int:
        return int(self.g.shape[0])

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, messages: torch.Tensor) -> torch.Tensor:
        """Encode a ``[batch, k]`` tensor of binary messages."""
        generator = torch.as_tensor(self.g, device=messages.device, dtype=torch.int64)
        return (messages.to(torch.int64) @ generator) % 2

    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        messages = torch.randint(0, 2, (count, self.k), generator=generator)
        return self.encode(messages)
