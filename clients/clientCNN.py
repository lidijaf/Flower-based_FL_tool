import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import OrderedDict
from typing import List
import torch
import flwr as fl

#from models.modelCNN import CNN_MNIST, CNN_FMNIST, CNN_CIFAR10, CNN_CIFAR100, ResNet18, train_CNN, test_CNN
from models.modelCNN import CNN_MNIST, CNN_FMNIST, CNN_CIFAR10, train_CNN, test_CNN

from utils import get_cfg


class CNN_Client(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.testloader = testloader
        
        shared_cfg = get_cfg("conf/config_common.yaml")
        server_cfg = get_cfg("conf/config_client.yaml")
        # merge
        cfg = {**shared_cfg, **server_cfg}

        
        model_name = cfg.get("model")
        if model_name == "CNN_FMNIST":
            self.model = CNN_FMNIST()
        elif model_name == "CNN_MNIST":
            self.model = CNN_MNIST()
        elif model_name == "CNN_CIFAR10":
            self.model = CNN_CIFAR10()
        elif model_name == "CNN_CIFAR100":
            self.model = CNN_CIFAR100()
        elif model_name == "ResNet18":
            self.model = ResNet18()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.device = torch.device(cfg.get("device"))
        self.model.to(self.device)
        self.algorithm = cfg.get("algorithm")

        # Persistent across rounds
        self.theta_params = None      # personalized model θ
        self.first_round = True       # initialize θ ← w only once

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        # ✅ Overwrite the saved model each round
        torch.save(self.model.state_dict(), "latest_global_model.pth")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_model_params = [p.detach().clone().to(self.device) for p in self.model.parameters()]

        if self.algorithm == "pfedme":
            updated_global, updated_theta, train_loss = train_CNN(
                self.model, self.trainloader,
                cfg=cfg,    
                theta_params=self.theta_params,
                global_model_params=global_model_params,
                first_round=self.first_round
            )
            self.theta_params = updated_theta
            self.first_round = False

            return [p.cpu().numpy() for p in updated_global], len(self.trainloader), {
                "train_loss": float(train_loss)
            }

        elif self.algorithm in ["fedavg", "fedavg+KD"]:
            train_loss = train_CNN(self.model, self.trainloader, cfg, None, None, True)
            return self.get_parameters({}), len(self.trainloader), {"train_loss": float(train_loss)}

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def evaluate(self, parameters: List[np.ndarray], config):
        if self.theta_params is not None:
            for param, theta in zip(self.model.parameters(), self.theta_params):
                param.data = theta.data.clone()
        else:
            self.set_parameters(parameters)

        #test_loss, accuracy, precision, recall, f1_score = test_CNN(self.model, self.testloader)
        test_loss, accuracy, precision, recall, f1_score = test_CNN(self.model, self.testloader)
        return float(test_loss), len(self.testloader.dataset), {
            "test_loss": float(test_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
        
if __name__ == "__main__":
    from data_preparations.datasetCNN import load_data  # or wherever your load_data function is
    from omegaconf import OmegaConf
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    

    shared_cfg = OmegaConf.to_container(get_cfg("conf/config_common.yaml"), resolve=True)
    client_cfg = OmegaConf.to_container(get_cfg("conf/config_client.yaml"), resolve=True)

    # merge
    cfg = {**shared_cfg, **client_cfg}
    
    # override client_id and data_path dynamically
    cfg["client_id"] = args.client_id
    cfg["data_path"] = args.data_path


    trainloader, testloader = load_data(cfg)
    client = CNN_Client(trainloader, testloader)

    import flwr as fl
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


