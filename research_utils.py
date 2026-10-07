"""Shared configuration, reproducibility, metrics, and result helpers."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def bit_error_rate(logits_or_bits: torch.Tensor, targets: torch.Tensor, logits: bool = True) -> float:
    predictions = (logits_or_bits >= 0).to(targets.dtype) if logits else logits_or_bits.to(targets.dtype)
    return float((predictions != targets).to(torch.float32).mean().item())


def block_error_rate(logits_or_bits: torch.Tensor, targets: torch.Tensor, logits: bool = True) -> float:
    predictions = (logits_or_bits >= 0).to(targets.dtype) if logits else logits_or_bits.to(targets.dtype)
    return float((predictions != targets).any(dim=1).to(torch.float32).mean().item())


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write experiment rows atomically enough for a single local process."""
    if not rows:
        raise ValueError("No result rows to write")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
