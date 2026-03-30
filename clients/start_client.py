import os
import argparse

import flwr as fl

from utils import load_merged_config, resolve_client_data_path


def build_client(cfg):
    task = cfg.get("task")
    model = cfg.get("model")
    dataset = cfg.get("dataset")

    if task == "classification":
        from data_preparations.datasetCNN import load_data
        from clients.clientCNN import CNN_Client

        trainloader, testloader = load_data(cfg)
        return CNN_Client(trainloader, testloader, cfg)

    if task == "anomaly detection":
        from data_preparations.datasetTR import load_data

        if model == "Autoencoder":
            from clients.clientAE import Autoencoder_Client

            trainloader, testloader = load_data(cfg, mislabel_percent=0.0)
            return Autoencoder_Client(trainloader, testloader, cfg)

        if model == "Transformer":
            from clients.clientTR import TransformerClient

            trainloader, testloader = load_data(cfg, mislabel_percent=0.0)
            return TransformerClient(trainloader, testloader, cfg)

        raise ValueError(f"Unsupported anomaly detection model: {model}")

    raise ValueError(f"Unsupported task: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=False)
    parser.add_argument("--data_path", type=str, required=False, default=None)
    args = parser.parse_args()

    cfg = load_merged_config("conf/config_common.yaml", "conf/config_client.yaml")

    client_mode = cfg.get("client_mode", "simulation")
    if client_mode == "simulation" and args.client_id is None:
        raise ValueError("Simulation mode requires --client_id.")

    cfg["client_id"] = args.client_id
    cfg["data_path"] = resolve_client_data_path(
        cfg,
        client_id=args.client_id,
        cli_data_path=args.data_path,
    )

    client = build_client(cfg)

    fl.client.start_numpy_client(
        server_address=os.getenv("SERVER_ADDR", cfg.get("server_address", "127.0.0.1:8080")),
        client=client,
    )


if __name__ == "__main__":
    main()
