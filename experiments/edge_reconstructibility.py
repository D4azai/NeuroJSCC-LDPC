"""Run attribute-only Tanner-edge reconstruction diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from graph_learning.edge_reconstruction import evaluate_edge_reconstruction
from graph_learning.perturbation import edge_change_fraction, perturb_parity_check
from ldpc.parity_check import make_parity_check
from research_utils import load_config, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edge_reconstruction.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["seeds"] = [0]
        config["negative_sampling"] = ["uniform", "hard"]
        config["perturbation"] = [0.0, 0.25, 0.5]
    original = make_parity_check(config["code"])
    rows: list[dict] = []
    for seed in config["seeds"]:
        for requested in config.get("perturbation", [0.0]):
            h = perturb_parity_check(original, float(requested), int(seed))
            actual = edge_change_fraction(original, h)
            for strategy in config["negative_sampling"]:
                for result in evaluate_edge_reconstruction(h, strategy, int(seed)):
                    rows.append(
                        {
                            "code": config["code"], "seed": seed,
                            "perturbation_requested": requested,
                            "perturbation_actual": actual,
                            "negative_sampling": strategy, "model": result.model,
                            "roc_auc": result.roc_auc,
                            "average_precision": result.average_precision,
                            "num_variables": h.shape[1], "num_checks": h.shape[0],
                            "num_edges": int(h.sum()), "density": float(h.mean()),
                        }
                    )
    write_rows(config["output"], rows)
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["negative_sampling", "model"])[["roc_auc", "average_precision"]]
    summary = grouped.agg(["mean", "std"])
    summary.to_csv(Path(config["output"]).with_name("reconstructibility_summary.csv"))
    ax = grouped.mean().plot(kind="bar", ylim=(0.0, 1.0), figsize=(9, 4))
    ax.set_ylabel("Score")
    ax.set_title("Attribute-only Tanner edge reconstruction")
    plt.tight_layout()
    figure = Path(config.get("figure_directory", "results/figures"))
    figure.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure / "edge_reconstructibility.png", dpi=180)
    plt.close()
    print(f"Saved {len(rows)} rows to {config['output']}")


if __name__ == "__main__":
    main()
