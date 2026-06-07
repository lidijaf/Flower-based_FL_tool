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
from monitoring.timing import Timer
from monitoring.logger import log_client_metric
from monitoring.memory import MemoryTracker
from monitoring.communication import communication_summary


def make_json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)

def make_scalar_loss(value):
    flat_values = []

    def collect(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()

        if isinstance(v, np.ndarray):
            flat_values.extend(np.asarray(v, dtype=float).ravel().tolist())
            return

        if isinstance(v, (list, tuple)):
            for item in v:
                collect(item)
            return

        try:
            flat_values.append(float(v))
        except (TypeError, ValueError):
            pass

    collect(value)

    if not flat_values:
        return 0.0

    return float(np.mean(flat_values))

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
        if self.algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new", "drfl"]:
            memory_tracker = MemoryTracker("client_fit_memory")
            memory_tracker.start()

            with Timer("client_fit") as timer:
                result = self.algorithm_impl.fit(self, parameters, config)

            memory_metrics = memory_tracker.stop()

            updated_parameters, num_examples, metrics = result
            communication_metrics = communication_summary(parameters=updated_parameters)

            if metrics is None:
                metrics = {}

            metrics["fit_time_sec"] = timer.elapsed
            metrics.update(memory_metrics)

            metrics.update(
                {
                    "communication_num_parameters": communication_metrics["num_parameters"],
                    "communication_total_bytes": communication_metrics["total_bytes"],
                    "communication_total_mb": communication_metrics["total_mb"],
                }
            )

            log_client_metric(
                self.cfg,
                {
                    "event": "fit",
                    "fit_time_sec": timer.elapsed,
                    "num_examples": num_examples,
                    "algorithm": self.algorithm,
                    **memory_metrics,
                    "communication_num_parameters": communication_metrics["num_parameters"],
                    "communication_total_bytes": communication_metrics["total_bytes"],
                    "communication_total_mb": communication_metrics["total_mb"],
                },
            )

            return updated_parameters, num_examples, metrics

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def evaluate(self, parameters, config):
        if self.algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new", "drfl"]:
            memory_tracker = MemoryTracker("client_evaluate_memory")
            memory_tracker.start()

            with Timer("client_evaluate") as timer:
                result = self.algorithm_impl.evaluate(self, parameters, config)

            memory_metrics = memory_tracker.stop()

            loss, num_examples, metrics = result
            scalar_loss = make_scalar_loss(loss)

            if metrics is None:
                metrics = {}

            metrics["evaluate_time_sec"] = timer.elapsed
            metrics.update(memory_metrics)

            log_client_metric(
                self.cfg,
                {
                    "event": "evaluate",
                    "evaluate_time_sec": timer.elapsed,
                    "num_examples": num_examples,
                    "loss": make_json_safe(loss),
                    "algorithm": self.algorithm,
                    **memory_metrics,
                },
            )

            return scalar_loss, num_examples, metrics

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
if __name__ == "__main__":
    import argparse

    from data_loading.tensor_loader import load_tensor_splits as load_data

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
