"""Classical normalized min-sum decoding for binary LDPC codes."""

from __future__ import annotations

import torch


def min_sum_decode(
    llr: torch.Tensor,
    h: torch.Tensor,
    iterations: int = 20,
    normalization: float = 0.8,
) -> torch.Tensor:
    """Decode a batch of LLR vectors with normalized min-sum message passing."""
    h = torch.as_tensor(h, dtype=torch.bool, device=llr.device)
    batch, variables = llr.shape
    checks = h.shape[0]
    if h.shape[1] != variables:
        raise ValueError("LLR width and parity-check matrix disagree")
    v_to_c = llr[:, None, :].expand(batch, checks, variables).clone()
    c_to_v = torch.zeros_like(v_to_c)
    edge_mask = h[None, :, :]
    for _ in range(iterations):
        for check in range(checks):
            neighbours = torch.nonzero(h[check], as_tuple=False).flatten()
            messages = v_to_c[:, check, neighbours]
            signs = torch.where(messages >= 0, 1.0, -1.0)
            abs_messages = messages.abs()
            for local, variable in enumerate(neighbours):
                others = torch.arange(neighbours.numel(), device=llr.device) != local
                if others.any():
                    sign = signs[:, others].prod(dim=1)
                    magnitude = abs_messages[:, others].min(dim=1).values
                    c_to_v[:, check, variable] = normalization * sign * magnitude
        posterior = llr + c_to_v.sum(dim=1)
        decisions = (posterior < 0).to(torch.int64)
        syndrome = (decisions @ h.to(torch.int64).T) % 2
        if not syndrome.any():
            break
        totals = c_to_v.sum(dim=1)
        v_to_c = (llr[:, None, :] + totals[:, None, :] - c_to_v) * edge_mask
    return decisions
