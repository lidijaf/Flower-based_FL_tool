import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_cfg
import torch
from flwr.common import Metrics
from collections import OrderedDict
from typing import Dict, List, Tuple

'''
def get_evaluate_fn(testloader, dict: Dict):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        cfg = get_cfg()
        model_name = cfg.get("model")

        # Select model based on configuration
        if model_name == "CNN_FMNIST":
            model = CNN_FMNIST()
        elif model_name == "CNN_MNIST":
            model = CNN_MNIST()
        elif model_name == "CNN_CIFAR10":
            model = CNN_CIFAR10()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Determine the device to use
        device = torch.device(cfg.get("device"))

        # Convert the parameters to a state_dict
        model_keys = list(model.state_dict().keys())
        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.to(device)

        # Evaluate the model
        loss, accuracy = test_CNN(model, testloader)
        dict["evaluate_accuracy"].append(accuracy)
        dict["evaluate_loss"].append(loss)
        return float(loss), {"global_model_server_data_eval_accuracy": accuracy,
                             "global_model_server_data_eval_loss": loss
                             }

    return evaluate_fn
'''


def get_weighted_average_fit(dict: Dict):
    def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        train_losses = [n * m["train_loss"] for n, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        result = {
            "train_loss": sum(train_losses) / sum(examples),
        }
        dict["train_loss"].append(result['train_loss'])
        return result

    return weighted_average_fit


def get_weighted_average_eval(dict: Dict):
    def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:

        """The strategy will then call these functions whenever it receives fit or evaluate metrics from clients.
         As users, we need to tell the framework how to handle/aggregate these metrics,
         and we do so by passing metric aggregation functions to the strategy.
         The two possible functions are fit_metrics_aggregation_fn and evaluate_metrics_aggregation_fn"""

        # Multiply accuracy of each client by number of examples used
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

def get_evaluate_config_fn(metrics_dict=None):
    def evaluate_config(server_round: int):
        """Return an empty config dictionary for CNN, simulating default behavior."""
        return {}
    return evaluate_config

'''
def weighted_average_eval_pfedme(dict: Dict) -> Metrics:
    # evaluate_metrics_aggregation_fn
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        cfg = get_cfg()
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        if cfg["config_fit_pfedme"].get("new") == False:
            dict["accuracy_global_pfedme"].append(sum(accuracies) / sum(examples))
        else:
            dict["accuracy_global_pfedme"].append(sum(accuracies) / sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global": sum(accuracies) / sum(examples)}

    return evaluate



def weighted_average_fit_pfedme(dict: Dict) -> Metrics:
    # fit_metrics_aggregation_fn
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        cfg = get_cfg()
        # Multiply accuracy of each client by number of examples used
        accuracies_personalized = [num_examples * m["accuracy_personalized"] for num_examples, m in metrics]
        accuracies_local = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        if cfg["config_fit_pfedme"].get("new") == False:
            dict["accuracy_personalized_pfedme"].append(sum(accuracies_personalized) / sum(examples))
            dict["accuracy_local_pfedme"].append(sum(accuracies_local) / sum(examples))
        else:
            dict["accuracy_personalized_pfedme"].append(sum(accuracies_personalized) / sum(examples))
            dict["accuracy_local_pfedme"].append(sum(accuracies_local) / sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_personalized": sum(accuracies_personalized) / sum(examples),
                "accuracy_local": sum(accuracies_local) / sum(examples)}

    return evaluate
'''


