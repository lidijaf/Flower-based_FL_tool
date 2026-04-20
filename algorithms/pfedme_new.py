from typing import Dict, Any, List

import torch

from algorithms.base import BaseAlgorithm
from models.modelCNN import test_CNN, train_CNN
from models.modelAE import train as train_ae, vali as vali_ae
from models.modelTR import train as train_tr, vali as vali_tr, test as test_tr


class PFedMeNewAlgorithm(BaseAlgorithm):
    @property
    def name(self) -> str:
        return "pfedme_new"

    def validate_config(self) -> None:
        pfedme_cfg = self.cfg.get("config_fit_pfedme_new")
        if not pfedme_cfg:
            raise ValueError("Missing 'config_fit_pfedme_new' in configuration.")

        required = ["local_rounds", "local_iterations", "lambda_reg", "mu", "new"]
        missing = [key for key in required if pfedme_cfg.get(key) is None]

        if missing:
            raise ValueError(
                f"Missing pFedMeNew config fields in 'config_fit_pfedme_new': {missing}"
            )

    def _build_cnn_cfg(self, client):
        """
        Build a config for CNN training so train_CNN reads the pFedMeNew
        hyperparameters through the existing 'config_fit_pfedme' path.
        """
        cfg_copy = dict(client.cfg)

        pfedme_new_cfg = dict(cfg_copy.get("config_fit_pfedme_new", {}))
        pfedme_new_cfg["new"] = True

        # modelCNN.py currently expects algorithm == "pfedme"
        # and reads cfg["config_fit_pfedme"]
        cfg_copy["algorithm"] = "pfedme"
        cfg_copy["config_fit_pfedme"] = pfedme_new_cfg

        return cfg_copy
    
    def _build_ae_cfg(self, client):
        """
        Build a config for AE training so modelAE.train reads the pFedMeNew
        hyperparameters through the existing 'config_fit_pfedme' path.
        """
        cfg_copy = dict(client.cfg)

        pfedme_new_cfg = dict(cfg_copy.get("config_fit_pfedme_new", {}))
        pfedme_new_cfg["new"] = True

        # modelAE.py currently expects algorithm == "pfedme"
        # and reads cfg["config_fit_pfedme"]
        cfg_copy["algorithm"] = "pfedme"
        cfg_copy["config_fit_pfedme"] = pfedme_new_cfg

        return cfg_copy

    def _build_tr_cfg(self, client):
        """
        Build a config for Transformer training so modelTR.train reads the
        pFedMeNew hyperparameters through the existing 'config_fit_pfedme' path.
        """
        cfg_copy = dict(client.cfg)

        pfedme_new_cfg = dict(cfg_copy.get("config_fit_pfedme_new", {}))
        pfedme_new_cfg["new"] = True

        # modelTR.py currently expects algorithm == "pfedme"
        # and reads cfg["config_fit_pfedme"]
        cfg_copy["algorithm"] = "pfedme"
        cfg_copy["config_fit_pfedme"] = pfedme_new_cfg

        return cfg_copy    
        
    def initialize_state(self) -> Dict[str, Any]:
        return {
            "theta_params": None,
            "first_round": True,
        }

    def fit(self, client, parameters: List, config: Dict[str, Any]):
        client.set_parameters(parameters)

        model_name = client.cfg.get("model", "").lower()

        if "cnn" in model_name:
            global_model_params = [
                p.detach().clone().to(client.device) for p in client.model.parameters()
            ]

            cnn_cfg = self._build_cnn_cfg(client)

            train_result = train_CNN(
                client.model,
                client.trainloader,
                cfg=cnn_cfg,
                theta_params=client.algorithm_state["theta_params"],
                global_model_params=global_model_params,
                first_round=client.algorithm_state["first_round"],
            )

            if isinstance(train_result, tuple) and len(train_result) == 2:
                updated_global, train_loss = train_result
                client.algorithm_state["first_round"] = False

                return (
                    [p.detach().cpu().numpy() for p in updated_global],
                    len(client.trainloader.dataset),
                    {"train_loss": float(train_loss)},
                )

            raise ValueError(
                "pFedMeNew CNN path expects train_CNN to return "
                "(updated_global, train_loss)."
            )
        elif "autoencoder" in model_name:
            ae_cfg = self._build_ae_cfg(client)

            global_params, train_loss = train_ae(
                client.model,
                client.trainloader,
                ae_cfg,
            )

            loss_personalized, _ = vali_ae(
                client.model,
                client.valloader,
                client.trainloader,
                ae_cfg,
            )

            with torch.no_grad():
                for param, g_param in zip(client.model.parameters(), global_params):
                    param.data = g_param.data.clone()

            loss_local_global, _ = vali_ae(
                client.model,
                client.valloader,
                client.trainloader,
                ae_cfg,
            )

            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                {
                    "train_loss": float(train_loss),
                    "reconstruction_loss_local_global": float(loss_local_global),
                    "reconstruction_loss_personalized": float(loss_personalized),
                },
            )
        elif "transformer" in model_name:
            tr_cfg = self._build_tr_cfg(client)

            global_params, train_loss = train_tr(
                client.model,
                client.trainloader,
                client.testloader,
                client.k,
                client.win_size,
                tr_cfg,
            )

            loss_personalized = vali_tr(
                client.model,
                client.testloader,
                client.trainloader,
                client.k,
                client.win_size,
                tr_cfg,
            )

            with torch.no_grad():
                for param, g_param in zip(client.model.parameters(), global_params):
                    param.data = g_param.data.clone()

            loss_local_global = vali_tr(
                client.model,
                client.testloader,
                client.trainloader,
                client.k,
                client.win_size,
                tr_cfg,
            )

            return (
                client.get_parameters(None),
                len(client.trainloader.dataset),
                {
                    "train_loss": float(train_loss),
                    "reconstruction_loss_local_global": float(loss_local_global),
                    "reconstruction_loss_personalized": float(loss_personalized),
                },
            )
        
    def evaluate(self, client, parameters: List, config: Dict[str, Any]):
        model_name = client.cfg.get("model", "").lower()

        if "cnn" in model_name:
            theta_params = client.algorithm_state.get("theta_params")

            if theta_params is not None:
                for param, theta in zip(client.model.parameters(), theta_params):
                    param.data = theta.data.clone()
            else:
                client.set_parameters(parameters)

            test_loss, accuracy, precision, recall, f1_score = test_CNN(
                client.model,
                client.testloader,
            )

            return (
                float(test_loss),
                len(client.testloader.dataset),
                {
                    "test_loss": float(test_loss),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                },
            )

        elif "autoencoder" in model_name:
            client.set_parameters(parameters)

            avg_loss, threshold = vali_ae(
                client.model,
                client.valloader,
                client.trainloader,
                client.cfg,
            )

            return (
                float(avg_loss),
                len(client.valloader.dataset),
                {
                    "client_eval_loss": float(avg_loss),
                    "threshold": float(threshold),
                },
            )

        elif "transformer" in model_name:
            global_threshold = config.get("aggregated_threshold", 0.0)
            print(f"Received global threshold from server: {global_threshold}")

            client.set_parameters(parameters)

            test_rec_loss, accuracy, precision, recall, f1_score = test_tr(
                client.model,
                client.testloader,
                global_threshold,
                client.win_size,
                client.cfg,
            )

            return (
                float(test_rec_loss),
                len(client.testloader.dataset),
                {
                    "test_loss": float(test_rec_loss),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                },
            )

        raise ValueError(f"pFedMeNew not implemented for model: {client.cfg.get('model')}")    
