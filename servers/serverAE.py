import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from models.modelAE import Autoencoder, test
from utils import get_cfg


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
THRESHOLD_DIR = os.path.join(BASE_DIR, "data", "threshold")
THRESHOLD_FILE = os.path.join(THRESHOLD_DIR, "threshold_server.txt")
os.makedirs(THRESHOLD_DIR, exist_ok=True)


def load_server_config():
    shared_cfg = get_cfg("conf/config_common.yaml")
    server_cfg = get_cfg("conf/config_server.yaml")
    return {**shared_cfg, **server_cfg}


def get_evaluate_config_fn(metrics_dict: Dict):
    """Return evaluation configuration for each round."""
    def evaluate_config(server_round: int):
        return {
            "aggregated_threshold": metrics_dict.get("aggregated_threshold", 0.0)
        }
    return evaluate_config


def get_evaluate_fn(testloader, metrics_dict: Dict):
    """Define function for global evaluation on the server."""
    def evaluate_fn(server_round: int, parameters, config):
        cfg = load_server_config()

        sample_x, _ = next(iter(testloader))
        input_dim = sample_x.view(sample_x.size(0), -1).size(1)

        model = Autoencoder(input_dim=input_dim)
        device_name = cfg.get("device") or "cpu"
        device = torch.device(device_name)

        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE, "r") as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.0

        model_keys = list(model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(device) for k, v in zip(model_keys, parameters)}
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        loss, accuracy, precision, recall, f1_score = test(model, testloader, threshold)

        for key, value in {
            "evaluate_accuracy": accuracy,
            "evaluate_precision": precision,
            "evaluate_recall": recall,
            "evaluate_f1_score": f1_score,
        }.items():
            metrics_dict.setdefault(key, []).append(value)

        return float(loss), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    return evaluate_fn


def get_weighted_average_fit(metrics_dict: Dict):
    def aggregate(metrics: list) -> dict:
        cfg = get_cfg("conf/config_common.yaml")
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

        for key, value in result.items():
            metrics_dict.setdefault(key, []).append(value)

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

        with open(THRESHOLD_FILE, "w") as f:
            f.write(str(avg_threshold))

        metrics_dict.setdefault("client_eval_loss", []).append(avg_loss)
        metrics_dict.setdefault("threshold", []).append(avg_threshold)
        metrics_dict["aggregated_threshold"] = avg_threshold

        return {"client_eval_loss": avg_loss, "threshold": avg_threshold}

    return aggregate
