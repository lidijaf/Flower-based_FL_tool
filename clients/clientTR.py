import os
from collections import OrderedDict

import flwr as fl
import torch
from omegaconf import OmegaConf

from models.modelTR import AnomalyTransformer, test, train, vali
from utils import get_cfg

from algorithms.registry import get_algorithm

class TransformerClient(fl.client.NumPyClient):
    """Flower client for Transformer-based anomaly detection."""

    def __init__(self, trainloader, testloader, cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = testloader
        self.cfg = cfg

        self.win_size = cfg.get("win_size")
        self.k = cfg.get("k")
        self.input_c = cfg.get("input_c")
        self.output_c = cfg.get("output_c")

        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
        )

        device_name = cfg.get("device") or "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)

        self.algorithm = cfg.get("algorithm", "fedavg")
        self.algorithm_impl = get_algorithm(self.algorithm, cfg)
        self.algorithm_impl.validate_config()
        self.algorithm_state = self.algorithm_impl.initialize_state()

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Extract model parameters and return them as numpy arrays."""
        return [
            val.detach().cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def fit(self, parameters, config):
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
            return self.algorithm_impl.fit(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def evaluate(self, parameters, config):
        """Evaluate model and return loss/metrics."""
        algorithm = self.cfg.get("algorithm", "fedavg")

        if algorithm in ["fedavg", "fedavg+KD", "pfedme", "pfedme_new"]:
            return self.algorithm_impl.evaluate(self, parameters, config)

        raise ValueError(f"Unsupported algorithm: {algorithm}")

if __name__ == "__main__":
    import argparse

    from data_preparations.datasetTR import load_data

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

    trainloader, testloader = load_data(cfg, mislabel_percent=0.0)
    client = TransformerClient(trainloader, testloader, cfg)

    fl.client.start_numpy_client(
        server_address=os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        client=client,
    )
