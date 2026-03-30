import gc
import os

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import get_cfg


base_dir = os.path.dirname(__file__)
results_dir = os.path.join(base_dir, "outputs", "mislabeled_experiments")
os.makedirs(results_dir, exist_ok=True)


def load_runtime_config():
    common_cfg = get_cfg("conf/config_common.yaml")
    server_cfg = get_cfg("conf/config_server.yaml")
    return {**common_cfg, **server_cfg}


def validate_runtime_config(cfg):
    required = [
        "task",
        "dataset",
        "model",
        "algorithm",
        "num_rounds",
        "min_fit_clients",
        "min_evaluate_clients",
        "min_available_clients",
    ]
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config values: {missing}")


def get_server_functions(cfg):
    task = cfg.get("task")
    model = cfg.get("model")

    if task == "classification":
        from servers.serverCNN import (
            get_weighted_average_fit,
            get_weighted_average_eval,
            get_evaluate_config_fn,
        )
        return get_weighted_average_fit, get_weighted_average_eval, get_evaluate_config_fn

    if task == "anomaly detection":
        if model == "Autoencoder":
            from servers.serverAE import (
                get_weighted_average_fit,
                get_weighted_average_eval,
                get_evaluate_config_fn,
            )
            return get_weighted_average_fit, get_weighted_average_eval, get_evaluate_config_fn

        if model == "Transformer":
            from servers.serverTR import (
                get_weighted_average_fit,
                get_weighted_average_eval,
                get_evaluate_config_fn,
            )
            return get_weighted_average_fit, get_weighted_average_eval, get_evaluate_config_fn

        raise ValueError(f"Unsupported anomaly detection model: {model}")

    raise ValueError(f"Unsupported task: {task}")


def save_and_plot(metrics_dict, percent):
    np.save(os.path.join(results_dir, f"metrics_m{percent}.npy"), metrics_dict)

    for key in ["train_loss", "test_loss", "accuracy"]:
        y = metrics_dict.get(key, [])
        if not y:
            continue

        rounds = np.arange(len(y)) if key == "accuracy" else np.arange(1, len(y) + 1)

        plt.figure()
        plt.plot(rounds, y, marker="o")
        plt.title(f"{key.replace('_', ' ').capitalize()} (Mislabel {percent}%)")
        plt.xlabel("Federated Round")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{key}_vs_round_m{percent}.png"))
        plt.close()


if __name__ == "__main__":
    cfg = load_runtime_config()
    validate_runtime_config(cfg)

    algorithm = str(cfg.get("algorithm")).lower()
    print(f"Launching server with learning algorithm: {algorithm}")

    global_metrics = {
        "train_loss": [],
        "test_loss": [],
        "accuracy": [],
        "aggregated_threshold": 0.0,
    }

    (
        get_weighted_average_fit,
        get_weighted_average_eval,
        get_evaluate_config_fn,
    ) = get_server_functions(cfg)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.get("fraction_fit"),
        min_fit_clients=cfg.get("min_fit_clients"),
        fraction_evaluate=cfg.get("fraction_evaluate"),
        min_evaluate_clients=cfg.get("min_evaluate_clients"),
        min_available_clients=cfg.get("min_available_clients"),
        fit_metrics_aggregation_fn=get_weighted_average_fit(global_metrics),
        evaluate_metrics_aggregation_fn=get_weighted_average_eval(global_metrics),
        on_evaluate_config_fn=get_evaluate_config_fn(global_metrics),
    )

    fl.server.start_server(
        server_address=os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        config=fl.server.ServerConfig(num_rounds=cfg.get("num_rounds", 10)),
        strategy=strategy,
    )

    save_and_plot(global_metrics, 0)

    del global_metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
