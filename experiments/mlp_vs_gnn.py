"""Compare BP, a feature-only MLP, and a Tanner-graph GNN."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import config_argument, load_experiment_config, run_decoder_comparison
from research_utils import write_rows


def main() -> None:
    args = config_argument("configs/mlp_vs_gnn.yaml")
    config = load_experiment_config(args)
    rows = run_decoder_comparison(config)
    write_rows(config["output"], rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["snr_db", "model"])["ber"].agg(["mean", "std"]).reset_index()
    summary.to_csv(Path(config["output"]).with_name("decoding_summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, group in summary.groupby("model"):
        ax.errorbar(group["snr_db"], group["mean"], yerr=group["std"].fillna(0), marker="o", label=model.upper())
    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("LDPC decoding under AWGN")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figure = Path(config.get("figure_directory", "results/figures"))
    figure.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure / "ber_vs_snr.png", dpi=180)
    plt.close(fig)
    print(f"Saved {len(rows)} rows to {config['output']}")


if __name__ == "__main__":
    main()
