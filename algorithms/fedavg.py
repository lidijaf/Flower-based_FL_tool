from typing import Dict, Any, List
from algorithms.base import BaseAlgorithm

from models.modelCNN import train_CNN, test_CNN
from models.modelAE import train as train_ae, vali as vali_ae
from models.modelTR import train as train_tr, vali as vali_tr, test as test_tr


class FedAvgAlgorithm(BaseAlgorithm):
    @property
    def name(self) -> str:
        return "fedavg"

    def fit(self, client, parameters: List, config: Dict[str, Any]):
        client.set_parameters(parameters)

        model_name = client.cfg.get("model", "").lower()

        if "cnn" in model_name:
            train_loss = train_CNN(
                client.model,
                client.trainloader,
                client.cfg,
                None,
                None,
                True,
            )
            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                {"train_loss": float(train_loss)},
            )

        elif "autoencoder" in model_name:
            train_loss = train_ae(
                client.model,
                client.trainloader,
                client.cfg,
            )
            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                {"train_loss": float(train_loss)},
            )

        elif "transformer" in model_name:
            k = client.cfg.get("k")
            win_size = client.cfg.get("win_size")

            train_rec_loss, threshold = train_tr(
                client.model,
                client.trainloader,
                client.testloader,
                k,
                win_size,
                client.cfg,
            )
            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                {
                    "train_loss": float(train_rec_loss),
                    "threshold": float(threshold),
                },
            )
        raise ValueError(f"FedAvg not implemented for model: {client.cfg.get('model')}")

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

        elif "autoencoder" in model_name:
            avg_loss, threshold = vali_ae(
                client.model,
                client.valloader,
                client.trainloader,
                client.cfg,
            )
            return (
                float(avg_loss),
                len(client.valloader.dataset),
                {
                    "client_eval_loss": float(avg_loss),
                    "threshold": float(threshold),
                },
            )

        elif "transformer" in model_name:
            global_threshold = config.get("aggregated_threshold", 0.0)
            print(f"Received global threshold from server: {global_threshold}")

            test_rec_loss, accuracy, precision, recall, f1_score = test_tr(
                client.model,
                client.testloader,
                global_threshold,
                client.win_size,
                client.cfg,
            )
            return (
                float(test_rec_loss),
                len(client.testloader.dataset),
                {
                    "test_loss": float(test_rec_loss),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                },
            )

        raise ValueError(f"FedAvg not implemented for model: {client.cfg.get('model')}")
