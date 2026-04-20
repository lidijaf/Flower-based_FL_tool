import os
from collections import OrderedDict

import flwr as fl
import torch
from models.modelAE import Autoencoder, train, vali
from utils import get_cfg

from algorithms.registry import get_algorithm

class Autoencoder_Client(fl.client.NumPyClient):
    """Flower client for Autoencoder."""

    def __init__(self, trainloader, valloader, cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.cfg = cfg

        # Determine input dimension dynamically
        sample_x, _ = next(iter(trainloader))
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
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train locally."""
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
            return self.algorithm_impl.fit(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def evaluate(self, parameters, config):
        """Evaluate model and compute anomaly threshold."""
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
            return self.algorithm_impl.evaluate(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {algorithm}")

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from data_preparations.datasetTR import load_data

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
    trainloader, testloader = load_data(cfg, mislabel_percent=0.0)

    # Initialize client
    client = Autoencoder_Client(trainloader, testloader, cfg)

    # Start client
    fl.client.start_numpy_client(
        server_address=os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        client=client,
    )
