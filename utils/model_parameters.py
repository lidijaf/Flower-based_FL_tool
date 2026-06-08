from collections import OrderedDict

import torch
from flwr.common import ndarrays_to_parameters

from models.modelCNN import CNN_CIFAR10, CNN_FMNIST, CNN_MNIST
from models.modelAE import Autoencoder
from models.modelTR import AnomalyTransformer

import os
from flwr.common import parameters_to_ndarrays


def build_model_from_cfg(cfg, input_dim=None):
    model_name = cfg.get("model")

    if model_name == "CNN_MNIST":
        return CNN_MNIST()

    if model_name == "CNN_FMNIST":
        return CNN_FMNIST()

    if model_name == "CNN_CIFAR10":
        return CNN_CIFAR10()

    if model_name == "Autoencoder":
        if input_dim is None:
            input_dim = cfg.get("input_c")
        return Autoencoder(input_dim)

    if model_name == "Transformer":
        return AnomalyTransformer(
            win_size=cfg.get("win_size"),
            enc_in=cfg.get("input_c"),
            c_out=cfg.get("output_c"),
        )

    raise ValueError(f"Unsupported model for parameter initialization: {model_name}")


def get_model_ndarrays(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def load_model_state(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=True)
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected state_dict or dict with key 'state_dict'."
        )

    return model


def get_initial_parameters_from_cfg(cfg):
    warm_start = cfg.get("warm_start", False)
    warm_start_path = cfg.get("warm_start_model_path")

    if not warm_start:
        return None

    if not warm_start_path:
        raise ValueError("warm_start is true, but warm_start_model_path is not set.")

    device = cfg.get("device") or "cpu"
    model = build_model_from_cfg(cfg)
    model = load_model_state(model, warm_start_path, device=device)

    return ndarrays_to_parameters(get_model_ndarrays(model))
    
def save_model_state_from_parameters(cfg, parameters, output_path):
    if parameters is None:
        raise ValueError("No parameters available to save.")

    device = cfg.get("device") or "cpu"
    model = build_model_from_cfg(cfg)
    ndarrays = parameters_to_ndarrays(parameters)

    state_dict = OrderedDict(
        {
            key: torch.tensor(value).to(device)
            for key, value in zip(model.state_dict().keys(), ndarrays)
        }
    )

    model.load_state_dict(state_dict, strict=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": cfg.get("model"),
            "dataset": cfg.get("dataset"),
            "algorithm": cfg.get("algorithm"),
        },
        output_path,
    )

    return output_path    
