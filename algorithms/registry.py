from algorithms.fedavg import FedAvgAlgorithm
from algorithms.pfedme import PFedMeAlgorithm
from algorithms.pfedme_new import PFedMeNewAlgorithm
from algorithms.drfl import DRFLAlgorithm
from algorithms.pfedme_new import PFedMeNewAlgorithm
from algorithms.drfl import DRFLAlgorithm


def get_algorithm(algorithm_name: str, cfg):
    name = algorithm_name.lower()

    registry = {
        "fedavg": FedAvgAlgorithm,
        "fedavg+kd": FedAvgAlgorithm,
        "pfedme": PFedMeAlgorithm,
        "pfedmenew": PFedMeNewAlgorithm,
        "pfedme_new": PFedMeNewAlgorithm,
        "drfl": DRFLAlgorithm,
    }

    if name not in registry:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    return registry[name](cfg)
