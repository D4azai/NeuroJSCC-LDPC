"""Deterministic synthetic codeword datasets shared by every decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from ldpc.channel import transmit_bpsk
from ldpc.coding import BinaryLinearCode


@dataclass(frozen=True)
class DatasetSplits:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


def _make_split(code: BinaryLinearCode, count: int, snr_db: float, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    bits = code.sample(count, generator)
    _, llr = transmit_bpsk(bits, snr_db, code.rate, generator)
    snr = torch.full((count, 1), float(snr_db), dtype=torch.float32)
    return TensorDataset(llr, snr, bits.to(torch.float32))


def make_dataset_splits(
    code: BinaryLinearCode,
    snr_db: float,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> DatasetSplits:
    """Generate disjoint, repeatable partitions used unchanged by MLP and GNN."""
    return DatasetSplits(
        train=_make_split(code, train_size, snr_db, seed * 10 + 1),
        validation=_make_split(code, validation_size, snr_db, seed * 10 + 2),
        test=_make_split(code, test_size, snr_db, seed * 10 + 3),
    )
