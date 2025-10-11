import sys
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import numpy as np
from flwr.common import Metrics
from models.modelAE import Autoencoder, test
from utils import get_cfg

# --------------------------
# Threshold storage
threshold_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "threshold")
os.makedirs(threshold_dir, exist_ok=True)
os.chdir(threshold_dir)
# --------------------------


def get_evaluate_config_fn(metrics_dict: Dict):
    """Return evaluation configuration for each round."""
    def evaluate_config(server_round: int):
        return {
            "aggregated_threshold": metrics_dict.get("aggregated_threshold", 0)
        }
    return evaluate_config


def get_evaluate_fn(testloader, metrics_dict: Dict):
    """Define function for global evaluation on the server."""
    def evaluate_fn(server_round: int, parameters, config):
        from omegaconf import OmegaConf

        # Load configs
        shared_cfg = OmegaConf.to_container(get_cfg("conf/config_common.yaml"), resolve=True)
        server_cfg = OmegaConf.to_container(get_cfg("conf/config_server.yaml"), resolve=True)
        cfg = {**shared_cfg, **server_cfg}

        # Sample input dimension
        sample_x, _ = next(iter(testloader))
        input_dim = sample_x.view(sample_x.size(0), -1).size(1)

        # Initialize model
        model = Autoencoder(input_dim=input_dim)
        device = torch.device(cfg.get("device"))

        # Load threshold from file if exists
        filename = "threshold_server.txt"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0

        # Convert Flower parameters to state_dict
        model_keys = list(model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in zip(model_keys, parameters)})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        # Evaluate
        loss, accuracy, precision, recall, f1_score = test(model, testloader, threshold)
        metrics_dict["evaluate_accuracy"].append(accuracy)
        metrics_dict["evaluate_precision"].append(precision)
        metrics_dict["evaluate_recall"].append(recall)
        metrics_dict["evaluate_f1_score"].append(f1_score)

        return float(loss), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    return evaluate_fn


def get_weighted_average_fit(metrics_dict: Dict):
    def aggregate(metrics: list) -> dict:
        cfg = get_cfg()
        algorithm = cfg.get("algorithm")

        examples = []
        train_losses = []
        loss_personal = []
        loss_local = []

        for item in metrics:
            if isinstance(item, tuple) and len(item) == 2:
                num_examples, m = item
            elif isinstance(item, dict):
                m = item
                num_examples = m.get("num_examples", 1)
            else:
                continue

            examples.append(num_examples)
            if algorithm == "pfedme":
                loss_personal.append(num_examples * m.get("reconstruction_loss_personalized", 0))
                loss_local.append(num_examples * m.get("reconstruction_loss_local_global", 0))
                train_losses.append(num_examples * m.get("train_loss", 0))
            elif algorithm == "fedavg":
                train_losses.append(num_examples * m.get("train_loss", 0))

        if not examples:
            return {}

        result = {}
        if algorithm == "pfedme":
            result["reconstruction_loss_personalized"] = sum(loss_personal) / sum(examples)
            result["reconstruction_loss_local_global"] = sum(loss_local) / sum(examples)
            result["train_loss"] = sum(train_losses) / sum(examples)
        else:
            result["train_loss"] = sum(train_losses) / sum(examples)

        # **Append to lists instead of overwriting**
        for k, v in result.items():
            if k not in metrics_dict:
                metrics_dict[k] = []
            metrics_dict[k].append(v)

        return result

    return aggregate

def get_weighted_average_eval(metrics_dict: Dict):
    def aggregate(metrics: list) -> dict:
        examples = []
        losses = []
        thresholds = []

        for item in metrics:
            if isinstance(item, tuple) and len(item) == 2:
                num_examples, m = item
            elif isinstance(item, dict):
                m = item
                num_examples = m.get("num_examples", 1)
            else:
                continue

            examples.append(num_examples)
            losses.append(num_examples * m.get("client_eval_loss", 0))
            thresholds.append(m.get("threshold", 0))

        avg_threshold = float(np.mean(thresholds)) if thresholds else 0.0
        avg_loss = sum(losses) / sum(examples) if examples else 0.0

        # Save threshold
        with open("threshold_server.txt", "w") as f:
            f.write(str(avg_threshold))

        # Append to metrics_dict lists
        if "client_eval_loss" not in metrics_dict:
            metrics_dict["client_eval_loss"] = []
        if "threshold" not in metrics_dict:
            metrics_dict["threshold"] = []

        metrics_dict["client_eval_loss"].append(avg_loss)
        metrics_dict["threshold"].append(avg_threshold)

        return {"client_eval_loss": avg_loss, "threshold": avg_threshold}

    return aggregate

