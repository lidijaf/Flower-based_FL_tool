import os
from collections import OrderedDict

import flwr as fl
import torch
from models.modelAE import Autoencoder, train, vali
from utils import get_cfg

from algorithms.registry import get_algorithm

from monitoring.timing import Timer
from monitoring.logger import log_client_metric
from monitoring.communication import communication_summary
from monitoring.memory import MemoryTracker
import numpy as np

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

class Autoencoder_Client(fl.client.NumPyClient):
    """Flower client for Autoencoder."""

    def __init__(self, trainloader, valloader, cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.cfg = cfg

        # Determine input dimension dynamically
        sample_batch = next(iter(trainloader))

        if isinstance(sample_batch, (tuple, list)):
            sample_x = sample_batch[0]
        else:
            sample_x = sample_batch

        input_dim = sample_x.view(sample_x.size(0), -1).size(1)

        self.model = Autoencoder(input_dim)

        device_name = cfg.get("device") or "cpu"
        self.device = torch.device(device_name)

        self.model.to(self.device)

        # Algorithm handling
        self.algorithm = cfg.get("algorithm", "fedavg")
        self.algorithm_impl = get_algorithm(self.algorithm, cfg)
        self.algorithm_impl.validate_config()
        self.algorithm_state = self.algorithm_impl.initialize_state()
        
    def set_parameters(self, parameters):
        """Load parameters into the model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
    def fit(self, parameters, config):
        """Train locally."""
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
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
                    "algorithm": algorithm,
                    **memory_metrics,
                    "communication_num_parameters": communication_metrics["num_parameters"],
                    "communication_total_bytes": communication_metrics["total_bytes"],
                    "communication_total_mb": communication_metrics["total_mb"],
                },
            )

            return updated_parameters, num_examples, metrics

        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def evaluate(self, parameters, config):
        """Evaluate model and compute anomaly threshold."""
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
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
                    "algorithm": algorithm,
                    **memory_metrics,
                },
            )

            return scalar_loss, num_examples, metrics

        raise ValueError(f"Unsupported algorithm: {algorithm}")    
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from data_loading.tensor_loader import load_tensor_splits as load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Load configs
    shared_cfg = OmegaConf.to_container(
        get_cfg("conf/config_common.yaml"), resolve=True
    )
    client_cfg = OmegaConf.to_container(
        get_cfg("conf/config_client.yaml"), resolve=True
    )

    cfg = {**shared_cfg, **client_cfg}

    # Override runtime args
    cfg["client_id"] = args.client_id
    cfg["data_path"] = args.data_path

    # Load data
    trainloader, testloader = load_data(cfg)

    # Initialize client
    client = Autoencoder_Client(trainloader, testloader, cfg)

    # Start client
    fl.client.start_numpy_client(
        server_address=os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        client=client,
    )
