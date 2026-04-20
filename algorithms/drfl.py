from typing import Dict, Any, List

from algorithms.base import BaseAlgorithm
from models.modelCNN import train_CNN, test_CNN
from utils.drfl_payload import extract_flat_gradients, gradients_to_metrics


class DRFLAlgorithm(BaseAlgorithm):
    @property
    def name(self) -> str:
        return "drfl"

    def validate_config(self) -> None:
        # Keep this minimal for now.
        return

    def initialize_state(self) -> Dict[str, Any]:
        return {}

    def fit(self, client, parameters: List, config: Dict[str, Any]):
        client.set_parameters(parameters)

        model_name = client.cfg.get("model", "").lower()

        if "cnn" in model_name:
            cnn_cfg = self._build_cnn_cfg(client)

            train_loss = train_CNN(
                client.model,
                client.trainloader,
                cnn_cfg,
                None,
                None,
                True,
            )

            grad_vector = extract_flat_gradients(client.model)

            metrics = {
                "train_loss": float(train_loss),
            }
            metrics.update(gradients_to_metrics(grad_vector))

            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                metrics,
            )

        raise ValueError(f"DRFL not implemented for model: {client.cfg.get('model')}")

    def evaluate(self, client, parameters: List, config: Dict[str, Any]):
        client.set_parameters(parameters)

        model_name = client.cfg.get("model", "").lower()

        if "cnn" in model_name:
            test_loss, accuracy, precision, recall, f1_score = test_CNN(
                client.model,
                client.testloader,
            )

            return (
                float(test_loss),
                len(client.testloader.dataset),
                {
                    "test_loss": float(test_loss),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                },
            )

        raise ValueError(f"DRFL not implemented for model: {client.cfg.get('model')}")

    def _build_cnn_cfg(self, client):
        """
        DRFL uses standard local training on the client side for now,
        so reuse the FedAvg local training path.
        """
        cfg_copy = dict(client.cfg)
        cfg_copy["algorithm"] = "fedavg"
        return cfg_copy
