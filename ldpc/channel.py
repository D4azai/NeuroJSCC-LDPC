"""BPSK over AWGN helpers used by all decoding baselines."""

from __future__ import annotations

import math

import torch


def noise_variance_from_ebn0(ebn0_db: float, code_rate: float) -> float:
    """Return real AWGN variance for unit-energy BPSK at an Eb/N0 value."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0)


def transmit_bpsk(
    bits: torch.Tensor,
    ebn0_db: float,
    code_rate: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transmit bits and return received symbols and exact channel LLRs."""
    symbols = 1.0 - 2.0 * bits.to(torch.float32)
    variance = noise_variance_from_ebn0(ebn0_db, code_rate)
    noise = torch.randn(symbols.shape, generator=generator, dtype=symbols.dtype)
    received = symbols + noise * math.sqrt(variance)
    llr = 2.0 * received / variance
    return received, llr
