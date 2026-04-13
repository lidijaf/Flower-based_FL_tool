from algorithms.base import BaseAlgorithm


class DRFLAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return "drfl"

    def fit(self, model, trainloader, valloader, device, global_params, client_state):
        raise NotImplementedError("Drfl not migrated yet")

    def evaluate(self, model, testloader, device, client_state):
        raise NotImplementedError("Drfl not migrated yet")
