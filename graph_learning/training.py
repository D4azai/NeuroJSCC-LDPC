"""Fair, shared-budget training and evaluation for neural decoders."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ldpc.belief_propagation import min_sum_decode
from ldpc.tanner_graph import TannerGraph
from research_utils import bit_error_rate, block_error_rate


@dataclass(frozen=True)
class DecoderMetrics:
    ber: float
    bler: float
    loss: float


def _forward(model: nn.Module, batch: tuple[torch.Tensor, ...], graph: TannerGraph) -> torch.Tensor:
    llr, snr, _ = batch
    if getattr(model, "uses_graph_structure", False):
        return model(llr, snr, graph)
    return model(llr, snr)


def evaluate_decoder(
    model: nn.Module, dataset: TensorDataset, graph: TannerGraph, batch_size: int, device: torch.device
) -> DecoderMetrics:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    loss = 0.0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = tuple(item.to(device) for item in batch)
            logits = _forward(model, batch, graph)
            targets = batch[2]
            loss += float(criterion(logits, targets).item())
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    return DecoderMetrics(
        ber=bit_error_rate(logits, targets),
        bler=block_error_rate(logits, targets),
        loss=loss / targets.numel(),
    )


def train_decoder(
    model: nn.Module,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    graph: TannerGraph,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Train with a fixed budget and select the lowest validation-loss epoch."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    loader_generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=loader_generator
        )
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward(model, batch, graph), batch[2])
            loss.backward()
            optimizer.step()
        validation = evaluate_decoder(model, validation_data, graph, batch_size, device)
        if validation.loss < best_loss:
            best_loss = validation.loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def evaluate_bp(
    dataset: TensorDataset,
    graph: TannerGraph,
    batch_size: int,
    device: torch.device,
    iterations: int = 20,
) -> DecoderMetrics:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for llr, _, bits in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        decoded = min_sum_decode(llr.to(device), graph.parity_check.to(device), iterations)
        predictions.append(decoded.cpu())
        targets.append(bits.to(torch.int64))
    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    return DecoderMetrics(
        ber=bit_error_rate(predicted, target, logits=False),
        bler=block_error_rate(predicted, target, logits=False),
        loss=float("nan"),
    )
