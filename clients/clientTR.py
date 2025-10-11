import sys
import os

# Add project root (the parent directory of 'clients') to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import flwr as fl
from models.modelTR import AnomalyTransformer, train, vali, test
from utils import get_cfg
import torch
from collections import OrderedDict



class TransformerClient(fl.client.NumPyClient):
    """Flower Client for Transformer-based anomaly detection."""

    def __init__(self, trainloader, testloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.testloader = testloader
        self.win_size = cfg.get("win_size")
        self.k = cfg.get("k")  # Weighting factor for series and prior losses
        self.input_c = cfg.get("input_c")
        self.output_c = cfg.get("output_c")

        # Initialize the Transformer model
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c)
        self.device = torch.device(cfg.get("device"))
        self.model.to(self.device)

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """Extract model parameters and return them as a list of numpy arrays."""
        model_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return model_parameters

    def fit(self, parameters, config):
        #cfg = get_cfg()
        algorithm = cfg.get("algorithm")

        self.set_parameters(parameters)

        if algorithm == "pfedme":   # MAYBE ADJUST VALI SO IT DOESNT COMPUTE TRESH ALWAYS
            global_params, train_loss = train(self.model, self.trainloader, self.testloader, self.k, self.win_size, cfg)

            # Evaluate personalized model
            loss_personalized, _ = vali(self.model, self.testloader, self.trainloader, self.k, self.win_size, cfg)

            # Evaluate local global model
            with torch.no_grad():
                for param, g_param in zip(self.model.parameters(), global_params):
                    param.data = g_param.data.clone()
            loss_local_global, _ = vali(self.model, self.testloader, self.trainloader, self.k, self.win_size)

            # params_to_return = [val.cpu().numpy() for val in global_params]
            return self.get_parameters({}), len(self.trainloader), {
                "train_loss": float(train_loss),
                "reconstruction_loss_local_global": float(loss_local_global),
                "reconstruction_loss_personalized": float(loss_personalized)
            }

        elif algorithm == "fedavg":
            train_rec_loss, threshold = train(self.model, self.trainloader, self.testloader, self.k, self.win_size, cfg)
            return self.get_parameters({}), len(self.trainloader), {
                "train_loss": float(train_rec_loss),
                "threshold": float(threshold),
            }

    def evaluate(self, parameters, config):
        """Evaluate model and return loss along with an anomaly threshold."""
        global_threshold = config.get("aggregated_threshold", 0.0)
        print(f"📥 Received global threshold from server: {global_threshold}")
        self.set_parameters(parameters)
        test_rec_loss, accuracy, precision, recall, f1_score = test(self.model, self.testloader, global_threshold, self.win_size, cfg)

        return float(test_rec_loss), len(self.testloader), {
            "test_loss": float(test_rec_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
    '''
    def evaluate(self, parameters, config):
        """Evaluate model and return loss along with an anomaly threshold."""
        global_threshold = config.get("aggregated_threshold", 0.0)
        print(f"📥 Received global threshold from server: {global_threshold}")
        self.set_parameters(parameters)
        test_rec_loss = test(self.model, self.testloader, global_threshold, self.win_size)

        return float(test_rec_loss), len(self.testloader), {
            "test_rec_loss": float(test_rec_loss),
        }
    '''
if __name__ == "__main__":
    from data_preparations.datasetTR import load_data  # or wherever your load_data function is
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


    trainloader, testloader = load_data(cfg, mislabel_percent=0.0)
    client = TransformerClient(trainloader, testloader)

    import flwr as fl
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

