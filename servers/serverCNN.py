from typing import Dict, List, Tuple

from flwr.common import Metrics
from utils.drfl_payload import deserialize_gradient_vector
import numpy as np

def get_weighted_average_fit(metrics_dict: Dict):
    def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        # -------------------------------------------------
        # 1. Collect basic info from all clients
        # -------------------------------------------------
        client_infos = []
        examples = []
        train_losses_weighted = []

        for num_examples, m in metrics:
            info = {
                "num_examples": num_examples,
                "train_loss": m.get("train_loss", 0.0),
                "gradient": None,
            }

            examples.append(num_examples)
            train_losses_weighted.append(num_examples * info["train_loss"])

            if "gradients_blob" in m:
                grad_vec = deserialize_gradient_vector(m["gradients_blob"])
                info["gradient"] = grad_vec
                
            client_infos.append(info)

        total_examples = sum(examples)

        if total_examples == 0:
            return {}

        # -------------------------------------------------
        # 2. Build gradient matrix
        # -------------------------------------------------
        gradient_clients = [info for info in client_infos if info["gradient"] is not None]

        if gradient_clients:
            grad_matrix = np.stack(
                [info["gradient"] for info in gradient_clients],
                axis=0,
            )
            print("Gradient matrix shape:", grad_matrix.shape)

            avg_gradient = sum(
                info["num_examples"] * info["gradient"]
                for info in gradient_clients
            ) / sum(info["num_examples"] for info in gradient_clients)


            gradient_norms = [
                float(np.linalg.norm(info["gradient"]))
                for info in gradient_clients
            ]

            avg_gradient_norm = float(np.linalg.norm(avg_gradient))

        if gradient_clients:
            worst_idx = max(
            range(len(gradient_clients)),
            key=lambda i: gradient_clients[i]["train_loss"],
            )
            worst_grad = gradient_clients[worst_idx]["gradient"]


            cosine_denom = (
                np.linalg.norm(avg_gradient) * np.linalg.norm(worst_grad) + 1e-12
            )
            cosine_sim = float(np.dot(avg_gradient, worst_grad) / cosine_denom)

            robust_direction = worst_grad - avg_gradient
            robust_direction_norm = float(np.linalg.norm(robust_direction))

            metrics_dict.setdefault("worst_client_loss", []).append(
                float(gradient_clients[worst_idx]["train_loss"])
            )
            metrics_dict.setdefault("worst_client_grad_norm", []).append(
                float(np.linalg.norm(worst_grad))
            )
            metrics_dict.setdefault("avg_gradient_norm", []).append(
                float(np.linalg.norm(avg_gradient))
            )
            metrics_dict.setdefault("robust_direction_norm", []).append(
                float(robust_direction_norm)
            )
            metrics_dict.setdefault("avg_worst_cosine_similarity", []).append(
                float(cosine_sim)
            )

        # -------------------------------------------------
        # 3. Inspect client losses
        # -------------------------------------------------
        client_losses = [
            (info["num_examples"], info["train_loss"])
            for info in client_infos
        ]

        # -------------------------------------------------
        # 3b. Compute simple robust client weights from losses
        # -------------------------------------------------
        loss_values = np.array(
            [info["train_loss"] for info in client_infos],
            dtype=np.float32,
        )

        robust_weights = loss_values / (loss_values.sum() + 1e-12)

        metrics_dict.setdefault("worst_client_loss", []).append(
            float(max(loss_values))
        )
        metrics_dict.setdefault("avg_client_loss", []).append(
            float(np.mean(loss_values))
        )

      
        # -------------------------------------------------
        # 4. Keep normal FedAvg loss aggregation for now
        # -------------------------------------------------
        result = {
            "train_loss": sum(train_losses_weighted) / total_examples,
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
