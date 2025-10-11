import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_cfg
from flwr.common import Metrics
from typing import Dict, List, Tuple
import os

# Determine base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Path to store threshold files
threshold_dir = os.path.join(BASE_DIR, "data", "threshold")
os.makedirs(threshold_dir, exist_ok=True)
os.chdir(threshold_dir)


#path = "C://Users//neman//PycharmProjects//data//treshold"
#os.makedirs(path, exist_ok=True)
#os.chdir(path)

'''
def get_evaluate_fn(testloader, dict: Dict):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        cfg = get_cfg()
        win_size = cfg.get("win_size")
        input_c = cfg.get("input_c")
        output_c = cfg.get("output_c")

        model = AnomalyTransformer(win_size=win_size, enc_in=input_c, c_out=output_c)
        device = torch.device(cfg.get("device"))

        # Load threshold
        filename = "threshold_server.txt"
        if os.path.exists(filename):
            with open(filename, "r") as file:
                threshold = float(file.read().strip())
        else:
            threshold = 0

        model_keys = list(model.state_dict().keys())
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        # Perform evaluation
        loss, accuracy, precision, recall, f1_score = test(model, testloader, threshold, win_size)

        dict["evaluate_accuracy"].append(accuracy)


        return float(loss), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }

    return evaluate_fn
'''


def get_weighted_average_fit(dict: Dict):
    def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        #cfg = get_cfg()
        from omegaconf import OmegaConf

        # Load both common and server configs
        shared_cfg = OmegaConf.to_container(get_cfg("conf/config_common.yaml"), resolve=True)
        server_cfg = OmegaConf.to_container(get_cfg("conf/config_server.yaml"), resolve=True)

        # Merge into one dict
        cfg = {**shared_cfg, **server_cfg}
        
        algorithm = cfg.get("algorithm")
        examples = [n for n, _ in metrics]

        if algorithm == "pfedme":
            loss_personal = [n * m["reconstruction_loss_personalized"] for n, m in metrics]
            loss_local = [n * m["reconstruction_loss_local_global"] for n, m in metrics]
            train_losses = [n * m["train_loss"] for n, m in metrics]

            result = {
                "reconstruction_loss_personalized": sum(loss_personal) / sum(examples),
                "reconstruction_loss_local_global": sum(loss_local) / sum(examples),
                "train_loss": sum(train_losses) / sum(examples)
            }
            dict["train_loss"].append(result["train_loss"])
            return result

        elif algorithm == "fedavg":
            train_losses = [n * m["train_loss"] for n, m in metrics]
            thresholds = [n * m["threshold"] for n, m in metrics]

            result = {
                "train_loss": sum(train_losses) / sum(examples),
            }
            dict["train_loss"].append(result["train_loss"])
            dict["aggregated_threshold"] = sum(thresholds)/sum(examples)

            return result
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    return weighted_average_fit


def get_weighted_average_eval(dict: Dict):
    def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        examples = [n for n, _ in metrics]
        test_losses = [n * m["test_loss"] for n, m in metrics]
        accuracies = [n * m["accuracy"] for n, m in metrics]
        precisions = [n * m["precision"] for n, m in metrics]
        recalls = [n * m["recall"] for n, m in metrics]
        f1_scores = [n * m["f1_score"] for n, m in metrics]

        result = {
            "test_loss": sum(test_losses) / sum(examples),
            "accuracy": sum(accuracies) / sum(examples),
            "precision": sum(precisions) / sum(examples),
            "recall": sum(recalls) / sum(examples),
            "f1_score": sum(f1_scores) / sum(examples),
        }
        dict["test_loss"].append(result["test_loss"])
        dict["accuracy"].append(result["accuracy"])
        return result

    return weighted_average_eval
'''

def get_weighted_average_eval(dict: Dict):
    def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        examples = [n for n, _ in metrics]
        test_rec_losses = [n * m["test_rec_loss"] for n, m in metrics]

        result = {
            "test_rec_loss": sum(test_rec_losses) / sum(examples),
        }
        dict["test_rec_loss"].append(result["test_rec_loss"])
        return result
    return weighted_average_eval
'''

def get_evaluate_config_fn(metrics_dict):
    def evaluate_config(server_round: int):
        """Generate evaluation configuration for each round."""
        # Create the configuration dictionary
        config = {
            "aggregated_threshold": metrics_dict["aggregated_threshold"],
        }
        return config
    return evaluate_config
