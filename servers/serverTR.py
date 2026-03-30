import os
from typing import Dict, List, Tuple

from flwr.common import Metrics

from utils import get_cfg


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
THRESHOLD_DIR = os.path.join(BASE_DIR, "data", "threshold")
THRESHOLD_FILE = os.path.join(THRESHOLD_DIR, "threshold_server.txt")
os.makedirs(THRESHOLD_DIR, exist_ok=True)


def load_server_config():
    common_cfg = get_cfg("conf/config_common.yaml")
    server_cfg = get_cfg("conf/config_server.yaml")
    return {**common_cfg, **server_cfg}


def get_weighted_average_fit(metrics_dict: Dict):
    def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        cfg = load_server_config()
        algorithm = cfg.get("algorithm")
        examples = [num_examples for num_examples, _ in metrics]
        total_examples = sum(examples)

        if total_examples == 0:
            return {}

        if algorithm == "pfedme":
            loss_personal = [
                num_examples * m.get("reconstruction_loss_personalized", 0.0)
                for num_examples, m in metrics
            ]
            loss_local = [
                num_examples * m.get("reconstruction_loss_local_global", 0.0)
                for num_examples, m in metrics
            ]
            train_losses = [
                num_examples * m.get("train_loss", 0.0)
                for num_examples, m in metrics
            ]

            result = {
                "reconstruction_loss_personalized": sum(loss_personal) / total_examples,
                "reconstruction_loss_local_global": sum(loss_local) / total_examples,
                "train_loss": sum(train_losses) / total_examples,
            }

            metrics_dict.setdefault("train_loss", []).append(result["train_loss"])
            return result

        if algorithm == "fedavg":
            train_losses = [
                num_examples * m.get("train_loss", 0.0)
                for num_examples, m in metrics
            ]
            thresholds = [
                num_examples * m.get("threshold", 0.0)
                for num_examples, m in metrics
            ]

            result = {
                "train_loss": sum(train_losses) / total_examples,
            }

            metrics_dict.setdefault("train_loss", []).append(result["train_loss"])
            metrics_dict["aggregated_threshold"] = sum(thresholds) / total_examples

            with open(THRESHOLD_FILE, "w") as f:
                f.write(str(metrics_dict["aggregated_threshold"]))

            return result

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return weighted_average_fit


def get_weighted_average_eval(metrics_dict: Dict):
    def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        examples = [num_examples for num_examples, _ in metrics]
        total_examples = sum(examples)

        if total_examples == 0:
            return {}

        test_losses = [
            num_examples * m.get("test_loss", 0.0)
            for num_examples, m in metrics
        ]
        accuracies = [
            num_examples * m.get("accuracy", 0.0)
            for num_examples, m in metrics
        ]
        precisions = [
            num_examples * m.get("precision", 0.0)
            for num_examples, m in metrics
        ]
        recalls = [
            num_examples * m.get("recall", 0.0)
            for num_examples, m in metrics
        ]
        f1_scores = [
            num_examples * m.get("f1_score", 0.0)
            for num_examples, m in metrics
        ]

        result = {
            "test_loss": sum(test_losses) / total_examples,
            "accuracy": sum(accuracies) / total_examples,
            "precision": sum(precisions) / total_examples,
            "recall": sum(recalls) / total_examples,
            "f1_score": sum(f1_scores) / total_examples,
        }

        metrics_dict.setdefault("test_loss", []).append(result["test_loss"])
        metrics_dict.setdefault("accuracy", []).append(result["accuracy"])

        return result

    return weighted_average_eval


def get_evaluate_config_fn(metrics_dict: Dict):
    def evaluate_config(server_round: int):
        return {
            "aggregated_threshold": metrics_dict.get("aggregated_threshold", 0.0),
        }

    return evaluate_config
