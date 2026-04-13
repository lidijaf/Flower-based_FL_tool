from algorithms.base import BaseAlgorithm


class PFedMeNewAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return "pFedMeNew"

    def fit(self, model, trainloader, valloader, device, global_params, client_state):
        raise NotImplementedError("pFedMeNew not migrated yet")

    def evaluate(self, model, testloader, device, client_state):
        raise NotImplementedError("pFedMeNew not migrated yet")
