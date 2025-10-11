import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
from models.modelAE import Autoencoder, train, vali
from utils import get_cfg
import torch
from collections import OrderedDict


class Autoencoder_Client(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, valloader, cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        # Determine input dimension by flattening a sample batch
        sample_x, _ = next(iter(trainloader))
        input_dim = sample_x.view(sample_x.size(0), -1).size(1)

        # Initialize Autoencoder with input_dim
        self.model = Autoencoder(input_dim)
        self.device = torch.device(cfg.get("device"))
        self.model.to(self.device)
        self.cfg = cfg

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        algorithm = self.cfg.get("algorithm")

        # Load the received parameters
        self.set_parameters(parameters)

        if algorithm == "pfedme":
            global_params, train_loss = train(self.model, self.trainloader, self.cfg)

            # Evaluate personalized model (w_i)
            loss_personalized, _ = vali(self.model, self.valloader, self.trainloader, self.cfg)

            # Evaluate local-global model (w)
            with torch.no_grad():
                for param, g_param in zip(self.model.parameters(), global_params):
                    param.data = g_param.data.clone()
            loss_local_global, _ = vali(self.model, self.valloader, self.trainloader, self.cfg)

            return self.get_parameters({}), len(self.trainloader.dataset), {
                "train_loss": float(train_loss),
                "reconstruction_loss_local_global": float(loss_local_global),
                "reconstruction_loss_personalized": float(loss_personalized)
            }

        elif algorithm == "fedavg":
            train_loss = train(self.model, self.trainloader, self.cfg)
            return self.get_parameters({}), len(self.trainloader.dataset), {
                "train_loss": float(train_loss)
            }

    def evaluate(self, parameters, config):
        """Evaluate model and return loss along with an anomaly threshold."""
        self.set_parameters(parameters)
        avg_loss, threshold = vali(self.model, self.valloader, self.trainloader, self.cfg)
        return float(avg_loss), len(self.valloader.dataset), {
            "client_eval_loss": float(avg_loss),
            "threshold": float(threshold)
        }


if __name__ == "__main__":
    from data_preparations.datasetTR import load_data
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    shared_cfg = OmegaConf.to_container(get_cfg("conf/config_common.yaml"), resolve=True)
    client_cfg = OmegaConf.to_container(get_cfg("conf/config_client.yaml"), resolve=True)

    # Merge configs
    cfg = {**shared_cfg, **client_cfg}

    # Override dynamic values
    cfg["client_id"] = args.client_id
    cfg["data_path"] = args.data_path

    # Load data
    trainloader, testloader = load_data(cfg, mislabel_percent=0.0)

    # Create client
    client = Autoencoder_Client(trainloader, testloader, cfg)

    # Start Flower client
    fl.client.start_numpy_client(
        #server_address="127.0.0.1:8080",
        server_address = os.getenv("SERVER_ADDR", "127.0.0.1:8080"),
        client=client,
    )

