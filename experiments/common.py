"""Experiment assembly helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from graph_learning.dataset import make_dataset_splits
from graph_learning.gnn_decoder import TannerGNNDecoder
from graph_learning.mlp_decoder import FeatureOnlyMLPDecoder
from graph_learning.training import evaluate_bp, evaluate_decoder, train_decoder
from ldpc.coding import BinaryLinearCode
from ldpc.parity_check import make_parity_check
from ldpc.tanner_graph import build_tanner_graph
from research_utils import count_parameters, load_config, set_seed


def config_argument(default: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default)
    parser.add_argument("--smoke", action="store_true", help="Use tiny sizes/epochs for validation")
    return parser.parse_args()


def load_experiment_config(arguments: argparse.Namespace) -> dict:
    config = load_config(arguments.config)
    if arguments.smoke:
        config["seeds"] = [0]
        config["snr_db"] = [2.0]
        config["epochs"] = 1
        config["train_size"] = 64
        config["validation_size"] = 32
        config["test_size"] = 64
    return config


def run_decoder_comparison(config: dict, perturbation: float = 0.0) -> list[dict]:
    rows: list[dict] = []
    configured_device = config.get("device", "auto")
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    h = make_parity_check(config["code"])
    code = BinaryLinearCode.from_parity_check(h)
    for seed in config["seeds"]:
        if perturbation:
            from graph_learning.perturbation import edge_change_fraction, perturb_parity_check

            altered_h = perturb_parity_check(h, perturbation, seed=int(seed))
            actual_perturbation = edge_change_fraction(h, altered_h)
        else:
            altered_h = h
            actual_perturbation = 0.0
        graph = build_tanner_graph(altered_h)
        for snr_db in config["snr_db"]:
            set_seed(int(seed))
            result_start = len(rows)
            splits = make_dataset_splits(
                code,
                float(snr_db),
                int(config["train_size"]),
                int(config["validation_size"]),
                int(config["test_size"]),
                int(seed),
            )
            mlp = FeatureOnlyMLPDecoder(
                code.n, int(config["mlp_hidden_dim"]), int(config["layers"])
            )
            gnn = TannerGNNDecoder(int(config["gnn_hidden_dim"]), int(config["layers"]))
            models = (("mlp", mlp), ("gnn", gnn))
            metrics_by_model = {}
            for model_name, model in models:
                label = (
                    f"seed={seed} Eb/N0={float(snr_db):g}dB "
                    f"rewire={perturbation:g} {model_name.upper()}"
                )
                print(f"Training {label} on {device}", flush=True)
                trained = train_decoder(
                    model,
                    splits.train,
                    splits.validation,
                    graph,
                    int(config["epochs"]),
                    int(config["batch_size"]),
                    float(config["learning_rate"]),
                    int(seed),
                    device,
                    int(config.get("early_stopping_patience", 5)),
                    label,
                )
                metrics = evaluate_decoder(
                    trained, splits.test, graph, int(config["batch_size"]), device
                )
                metrics_by_model[model_name] = metrics
                rows.append(
                    {
                        "code": config["code"],
                        "seed": seed,
                        "snr_db": snr_db,
                        "perturbation_requested": perturbation,
                        "perturbation_actual": actual_perturbation,
                        "model": model_name,
                        "ber": metrics.ber,
                        "bler": metrics.bler,
                        "test_bce": metrics.loss,
                        "parameters": count_parameters(trained),
                        "num_variables": h.shape[1],
                        "num_checks": h.shape[0],
                        "num_edges": int(h.sum()),
                        "density": float(h.mean()),
                    }
                )
            if perturbation == 0.0:
                bp = evaluate_bp(
                    splits.test, build_tanner_graph(h), int(config["batch_size"]), device,
                    int(config.get("bp_iterations", 20)),
                )
                rows.append(
                    {
                        "code": config["code"], "seed": seed, "snr_db": snr_db,
                        "perturbation_requested": 0.0, "perturbation_actual": 0.0,
                        "model": "bp", "ber": bp.ber, "bler": bp.bler,
                        "test_bce": "", "parameters": 0, "num_variables": h.shape[1],
                        "num_checks": h.shape[0], "num_edges": int(h.sum()),
                        "density": float(h.mean()),
                    }
                )
            graph_gain = metrics_by_model["mlp"].ber - metrics_by_model["gnn"].ber
            for row in rows[result_start:]:
                row["delta_graph"] = graph_gain
    return rows


def figure_directory(config: dict) -> Path:
    path = Path(config.get("figure_directory", "results/figures"))
    path.mkdir(parents=True, exist_ok=True)
    return path
