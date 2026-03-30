from typing import Dict, List, Tuple

from flwr.common import Metrics


def get_weighted_average_fit(metrics_dict: Dict):
    def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        train_losses = [num_examples * m.get("train_loss", 0.0) for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        total_examples = sum(examples)
        if total_examples == 0:
            return {}

        result = {
            "train_loss": sum(train_losses) / total_examples,
        }

        metrics_dict.setdefault("train_loss", []).append(result["train_loss"])
        return result

    return weighted_average_fit


def get_weighted_average_eval(metrics_dict: Dict):
    def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        examples = [num_examples for num_examples, _ in metrics]
        total_examples = sum(examples)
        if total_examples == 0:
            return {}

        test_losses = [num_examples * m.get("test_loss", 0.0) for num_examples, m in metrics]
        accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
        precisions = [num_examples * m.get("precision", 0.0) for num_examples, m in metrics]
        recalls = [num_examples * m.get("recall", 0.0) for num_examples, m in metrics]
        f1_scores = [num_examples * m.get("f1_score", 0.0) for num_examples, m in metrics]

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


def get_evaluate_config_fn(metrics_dict=None):
    def evaluate_config(server_round: int):
        return {}

    return evaluate_config
