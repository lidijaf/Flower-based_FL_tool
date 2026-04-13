import os
from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
from omegaconf import OmegaConf

from models.modelCNN import CNN_CIFAR10, CNN_FMNIST, CNN_MNIST, test_CNN, train_CNN
from utils import get_cfg

from algorithms.registry import get_algorithm

class CNN_Client(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, cfg) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg

        model_name = cfg.get("model")
        if model_name == "CNN_FMNIST":
            self.model = CNN_FMNIST()
        elif model_name == "CNN_MNIST":
            self.model = CNN_MNIST()
        elif model_name == "CNN_CIFAR10":
            self.model = CNN_CIFAR10()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        device_name = cfg.get("device") or "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)

        self.algorithm = cfg.get("algorithm")

        self.algorithm_impl = get_algorithm(self.algorithm, cfg)
        self.algorithm_state = self.algorithm_impl.initialize_state()

        # Persistent across rounds
        self.theta_params = None
        self.first_round = True

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if self.algorithm in ["fedavg", "fedavg+KD"]:
            return self.algorithm_impl.fit(self, parameters, config)

        self.set_parameters(parameters)
        global_model_params = [
            p.detach().clone().to(self.device) for p in self.model.parameters()
        ]

        if self.algorithm == "pfedme":
            train_result = train_CNN(
                self.model,
                self.trainloader,
                cfg=self.cfg,
                theta_params=self.theta_params,
                global_model_params=global_model_params,
                first_round=self.first_round,
            )

            if isinstance(train_result, tuple) and len(train_result) == 2:
                updated_global, train_loss = train_result
                self.first_round = False
                return (
                    [p.detach().cpu().numpy() for p in updated_global],
                    len(self.trainloader.dataset),
                    {"train_loss": float(train_loss)},
                )

            raise ValueError(
                "pFedMe path in clientCNN expects train_CNN to return "
                "(updated_global, train_loss)."
            )

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def evaluate(self, parameters, config):
        if self.algorithm in ["fedavg", "fedavg+KD"]:
            return self.algorithm_impl.evaluate(self, parameters, config)
        if self.theta_params is not None:
            for param, theta in zip(self.model.parameters(), self.theta_params):
                param.data = theta.data.clone()
        else:
            self.set_parameters(parameters)

        test_loss, accuracy, precision, recall, f1_score = test_CNN(
            self.model,
            self.testloader,
        )

        return (
            float(test_loss),
            len(self.testloader.dataset),
            {
                "test_loss": float(test_loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
            },
        )

if __name__ == "__main__":
    import argparse

    from data_preparations.datasetCNN import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    shared_cfg = OmegaConf.to_container(
        get_cfg("conf/config_common.yaml"), resolve=True
    )
    client_cfg = OmegaConf.to_container(
        get_cfg("conf/config_client.yaml"), resolve=True
    )

    cfg = {**shared_cfg, **client_cfg}
    cfg["client_id"] = args.client_id
    cfg["data_path"] = args.data_path

    trainloader, testloader = load_data(cfg)
    client = CNN_Client(trainloader, testloader, cfg)

    fl.client.start_numpy_client(
        server_address=os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        client=client,
    )
