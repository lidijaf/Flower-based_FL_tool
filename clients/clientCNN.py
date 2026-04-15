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
        self.algorithm_impl.validate_config()
        self.algorithm_state = self.algorithm_impl.initialize_state()


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if self.algorithm in ["fedavg", "fedavg+KD", "pfedme"]:
            return self.algorithm_impl.fit(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def evaluate(self, parameters, config):
        if self.algorithm in ["fedavg", "fedavg+KD", "pfedme"]:
            return self.algorithm_impl.evaluate(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

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
