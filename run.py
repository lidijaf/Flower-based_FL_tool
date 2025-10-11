import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import flwr as fl
from utils import get_cfg


#results_dir = "C://Users//neman//PycharmProjects//tool_demo_server//outputs//mislabeled_experiments"
#os.makedirs(results_dir, exist_ok=True)


# --------------------------
# Paths made platform-independent
base_dir = os.path.dirname(__file__)  # <-- ADDED
results_dir = os.path.join(base_dir, "outputs", "mislabeled_experiments")  # <-- CHANGED
os.makedirs(results_dir, exist_ok=True)
threshold_dir = os.path.join(base_dir, "data", "treshold")  # <-- CHANGED
# --------------------------


def save_and_plot(metrics_dict, percent):
    np.save(os.path.join(results_dir, f"metrics_m{percent}.npy"), metrics_dict)

    for key in ["train_loss", "test_loss", "accuracy"]:
        y = metrics_dict[key]
        rounds = np.arange(len(y)) if key == "accuracy" else np.arange(1, len(y) + 1)

        plt.figure()
        plt.plot(rounds, y, marker='o')
        plt.title(f"{key.replace('_', ' ').capitalize()} (Mislabel {percent}%)")
        plt.xlabel("Federated Round")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{key}_vs_round_m{percent}.png"))
        plt.close()


if __name__ == "__main__":
    
    shared_cfg = get_cfg("conf/config_common.yaml")
    server_cfg = get_cfg("conf/config_server.yaml")
    # merge
    cfg = {**shared_cfg, **server_cfg}
    #cfg = get_cfg()
    algorithm = cfg.get("algorithm").lower()
    print(f"Launching server with learning algorithm: {algorithm}")

    global_metrics = {"train_loss": [], "test_loss": [], "accuracy": [], "aggregated_threshold": 0.0}

    if cfg.get("task") == "classification":
        from servers.serverCNN import (
            get_weighted_average_fit,
            get_weighted_average_eval,
            get_evaluate_config_fn
        )
    elif cfg.get("task") == "anomaly detection":
        if cfg.get("model") == "Autoencoder":
            from servers.serverAE import (
                get_weighted_average_fit,
                get_weighted_average_eval,
                get_evaluate_config_fn
            )
        else:
            from servers.serverTR import (
                get_weighted_average_fit,
                get_weighted_average_eval,
                get_evaluate_config_fn
            )

    else:
        raise ValueError(f"Unsupported task")




    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.get("fraction_fit"),
        min_fit_clients=cfg.get("min_fit_clients"),
        fraction_evaluate=cfg.get("fraction_evaluate"),
        min_evaluate_clients=cfg.get("min_evaluate_clients"),
        min_available_clients=cfg.get("min_available_clients"),
        #fit_metrics_aggregation_fn=lambda r: get_weighted_average_fit(global_metrics)(r),
        fit_metrics_aggregation_fn=get_weighted_average_fit(global_metrics),
        evaluate_metrics_aggregation_fn=get_weighted_average_eval(global_metrics),
        on_evaluate_config_fn=get_evaluate_config_fn(global_metrics),
    )

    fl.server.start_server(
        #server_address="127.0.0.1:8080",
        server_address = os.getenv("SERVER_ADDR", "127.0.0.1:8080"),

        config=fl.server.ServerConfig(num_rounds=cfg.get("num_rounds", 10)),
        strategy=strategy
    )

    save_and_plot(global_metrics, int(0 * 100))
    del global_metrics
    torch.cuda.empty_cache()
    gc.collect()
