"""Associate edge predictability with the observed GNN decoding gain."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_utils import write_rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructibility", default="results/reconstructibility.csv")
    parser.add_argument("--decoding", default="results/perturbations.csv")
    parser.add_argument("--output", default="results/correlation.csv")
    parser.add_argument("--figure-directory", default="results/figures")
    args = parser.parse_args()
    edge = pd.read_csv(args.reconstructibility)
    decoding = pd.read_csv(args.decoding)
    edge = edge[(edge["model"] == "logistic_regression") & (edge["negative_sampling"] == "uniform")]
    edge = edge[["code", "seed", "perturbation_requested", "roc_auc", "average_precision"]]
    neural = decoding[decoding["model"].isin(["mlp", "gnn"])]
    pivot = neural.pivot_table(
        index=[
            "code", "seed", "snr_db", "perturbation_requested", "perturbation_actual",
            "num_variables", "num_checks", "num_edges", "density",
        ],
        columns="model", values="ber"
    ).reset_index()
    pivot["delta_graph"] = pivot["mlp"] - pivot["gnn"]
    combined = pivot.merge(edge, on=["code", "seed", "perturbation_requested"], how="inner")
    combined["reconstruction_model"] = "logistic_regression"
    combined["negative_sampling"] = "uniform"
    if len(combined) < 3:
        raise RuntimeError("At least three matched configurations are required for correlation")
    rows: list[dict] = []
    for metric in ("roc_auc", "average_precision"):
        reconstructibility = combined[metric].to_numpy(dtype=float)
        gain = combined["delta_graph"].to_numpy(dtype=float)
        pearson = _correlation(reconstructibility, gain)
        spearman = _correlation(_average_ranks(reconstructibility), _average_ranks(gain))
        rows.extend(
            [
                {"reconstructibility_metric": metric, "correlation": "pearson", "coefficient": pearson, "n": len(combined)},
                {"reconstructibility_metric": metric, "correlation": "spearman", "coefficient": spearman, "n": len(combined)},
            ]
        )
    write_rows(args.output, rows)
    figure = Path(args.figure_directory)
    figure.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(combined["roc_auc"], combined["delta_graph"], c=combined["perturbation_requested"], cmap="viridis")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Edge reconstruction ROC-AUC")
    ax.set_ylabel(r"Graph gain: $BER_{MLP} - BER_{GNN}$")
    ax.set_title("Reconstructibility and GNN gain (descriptive)")
    fig.colorbar(scatter, ax=ax, label="Rewiring fraction")
    fig.tight_layout()
    fig.savefig(figure / "reconstructibility_vs_graph_gain.png", dpi=180)
    plt.close(fig)
    combined.to_csv(Path(args.output).with_name("reconstructibility_vs_gain.csv"), index=False)
    print(f"Saved {len(rows)} correlation rows to {args.output}")


if __name__ == "__main__":
    main()
