from algorithms.base import BaseAlgorithm


class PFedMeAlgorithm(BaseAlgorithm):
    def name(self) -> str:
        return "pfedme"

    def fit(self, model, trainloader, valloader, device, global_params, client_state):
        raise NotImplementedError("pFedMe not migrated yet")

    def evaluate(self, model, testloader, device, client_state):
        raise NotImplementedError("pFedMe not migrated yet")
