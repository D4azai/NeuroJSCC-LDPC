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


def _standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def _fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """Fit L2-regularized logistic regression with deterministic Newton steps."""
    train, test = _standardize(x_train, x_test)
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    coefficients = np.zeros(train.shape[1], dtype=np.float64)
    regularization = np.eye(train.shape[1], dtype=np.float64) * 1e-4
    regularization[0, 0] = 0.0  # Do not penalize the intercept.
    for _ in range(50):
        logits = np.clip(train @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = train.T @ (probabilities - y_train) + regularization @ coefficients
        weights = probabilities * (1.0 - probabilities)
        hessian = train.T @ (train * weights[:, None]) + regularization
        step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-8, gradient)
        coefficients -= step
        if np.linalg.norm(step) < 1e-7:
            break
    logits = np.clip(test @ coefficients, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_small_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    max_epochs: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    train, test = _standardize(x_train, x_test)
    train_x = torch.tensor(train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(test, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 32), nn.GELU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    epochs_without_improvement = 0
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x).squeeze(1), train_y)
        loss.backward()
        optimizer.step()
        current_loss = float(loss.item())
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 20:
            break
    with torch.no_grad():
        return torch.sigmoid(model(test_x).squeeze(1)).numpy()


def evaluate_edge_reconstruction(
    h: np.ndarray,
    negative_sampling: str = "uniform",
    seed: int = 0,
    test_size: float = 0.3,
    mlp_max_epochs: int = 200,
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
    classifier_scores = (
        ("logistic_regression", _fit_logistic_regression(x_train, y_train, x_test)),
        ("small_mlp", _fit_small_mlp(x_train, y_train, x_test, seed, mlp_max_epochs)),
    )
    for name, scores in classifier_scores:
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
