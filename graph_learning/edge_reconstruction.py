"""Attribute-only Tanner-edge reconstruction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .negative_sampling import (
    candidate_pairs,
    positional_node_features,
    sample_negative_pairs,
    validate_pair_labels,
)


@dataclass(frozen=True)
class EdgeMetrics:
    model: str
    negative_sampling: str
    roc_auc: float
    average_precision: float


def _pair_features(pairs: np.ndarray, variables: np.ndarray, checks: np.ndarray) -> np.ndarray:
    left = variables[pairs[:, 0]]
    right = checks[pairs[:, 1]]
    return np.concatenate((left, right, np.abs(left - right), left * right), axis=1)


def _stratified_split(
    pairs: np.ndarray, labels: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * test_size)))
        test_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train = np.asarray(train_indices)
    test = np.asarray(test_indices)
    return pairs[train], pairs[test], labels[train], labels[test]


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    positive_ranks = np.flatnonzero(sorted_labels == 1)
    return float(np.mean(cumulative[positive_ranks] / (positive_ranks + 1)))


def _fit_torch_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    model_name: str,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_x = torch.tensor((x_train - mean) / scale, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor((x_test - mean) / scale, dtype=torch.float32)
    if model_name == "logistic_regression":
        model: nn.Module = nn.Linear(train_x.shape[1], 1)
        epochs = 300
    elif model_name == "small_mlp":
        model = nn.Sequential(nn.Linear(train_x.shape[1], 32), nn.GELU(), nn.Linear(32, 1))
        epochs = 500
    else:
        raise ValueError(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x).squeeze(1), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(model(test_x).squeeze(1)).numpy()


def evaluate_edge_reconstruction(
    h: np.ndarray,
    negative_sampling: str = "uniform",
    seed: int = 0,
    test_size: float = 0.3,
) -> list[EdgeMetrics]:
    """Evaluate classifiers and similarity scores without structural features."""
    h = np.asarray(h, dtype=np.uint8)
    checks, variables = h.shape
    variable_features = positional_node_features(variables, node_type=0)
    check_features = positional_node_features(checks, node_type=1)
    positives, available_negatives = candidate_pairs(h)
    sample_count = min(len(positives), len(available_negatives))
    if sample_count < len(positives):
        rng = np.random.default_rng(seed)
        positives = positives[rng.choice(len(positives), size=sample_count, replace=False)]
    negatives = sample_negative_pairs(
        h, sample_count, negative_sampling, variable_features, check_features, seed
    )
    validate_pair_labels(h, positives, negatives)
    pairs = np.concatenate((positives, negatives))
    labels = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))))
    train_pairs, test_pairs, y_train, y_test = _stratified_split(
        pairs, labels, test_size, seed
    )
    x_train = _pair_features(train_pairs, variable_features, check_features)
    x_test = _pair_features(test_pairs, variable_features, check_features)
    output: list[EdgeMetrics] = []
    for name in ("logistic_regression", "small_mlp"):
        scores = _fit_torch_classifier(x_train, y_train, x_test, name, seed)
        output.append(
            EdgeMetrics(
                name,
                negative_sampling,
                _roc_auc(y_test, scores),
                _average_precision(y_test, scores),
            )
        )
    # Similarities use the shared graph-independent positional coordinates.
    left = variable_features[test_pairs[:, 0], :5]
    right = check_features[test_pairs[:, 1], :5]
    cosine = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )
    euclidean = -np.linalg.norm(left - right, axis=1)
    for name, scores in (("cosine", cosine), ("euclidean", euclidean)):
        output.append(
            EdgeMetrics(
                name,
                negative_sampling,
                _roc_auc(y_test, scores),
                _average_precision(y_test, scores),
            )
        )
    return output
