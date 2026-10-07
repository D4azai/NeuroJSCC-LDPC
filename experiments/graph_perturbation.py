"""Measure graph-model behavior under controlled Tanner rewiring."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import config_argument, load_experiment_config, run_decoder_comparison
from research_utils import write_rows


def main() -> None:
    args = config_argument("configs/graph_perturbation.yaml")
    config = load_experiment_config(args)
    if args.smoke:
        config["perturbation"] = [0.0, 0.25, 0.5]
    rows: list[dict] = []
    for fraction in config["perturbation"]:
        rows.extend(run_decoder_comparison(config, float(fraction)))
    write_rows(config["output"], rows)
    frame = pd.DataFrame(rows)
    neural = frame[frame["model"].isin(["mlp", "gnn"])]
    pivot = neural.pivot_table(
        index=["seed", "snr_db", "perturbation_requested"], columns="model", values="ber"
    ).reset_index()
    pivot["delta_graph"] = pivot["mlp"] - pivot["gnn"]
    summary = pivot.groupby("perturbation_requested")["delta_graph"].agg(["mean", "std"]).reset_index()
    summary.to_csv(Path(config["output"]).with_name("perturbations_summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(summary["perturbation_requested"], summary["mean"], yerr=summary["std"].fillna(0), marker="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Requested edge rewiring fraction")
    ax.set_ylabel(r"Graph gain: $BER_{MLP} - BER_{GNN}$")
    ax.set_title("GNN gain under Tanner-graph perturbation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure = Path(config.get("figure_directory", "results/figures"))
    figure.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure / "graph_gain_vs_perturbation.png", dpi=180)
    plt.close(fig)
    print(f"Saved {len(rows)} rows to {config['output']}")


if __name__ == "__main__":
    main()
