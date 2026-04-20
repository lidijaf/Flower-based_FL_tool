from typing import List, Tuple, Optional

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from sklearn.cluster import DBSCAN
from utils.drfl_payload import deserialize_gradient_vector

class DRFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = None
    """
    First simple DRFL strategy:
    - reads train_loss from client fit metrics
    - builds robust weights from losses
    - aggregates model parameters using robust weights instead of num_examples only
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        if self.initial_parameters is None:
            self.initial_parameters = results[0][1].parameters

        # ---------------------------------------------
        # 1. Read client parameters, train losses, and gradients
        # ---------------------------------------------
        client_ndarrays = []
        loss_values = []
        client_gradients = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays.append(ndarrays)

            train_loss = float(fit_res.metrics.get("train_loss", 0.0))
            loss_values.append(train_loss)

            gradients_blob = fit_res.metrics.get("gradients_blob", None)
            if gradients_blob is None:
                raise ValueError(
                    "DRFLStrategy expected 'gradients_blob' in client metrics."
                )

            grad_vec = deserialize_gradient_vector(gradients_blob)
            client_gradients.append(grad_vec)

        loss_values = np.array(loss_values, dtype=np.float32)

        # ---------------------------------------------
        # 2. Cluster clients by gradient similarity
        # ---------------------------------------------
        grad_matrix = np.stack(client_gradients, axis=0)
        print(f"[DRFL] Round {server_round} gradient matrix shape: {grad_matrix.shape}")
        print(f"[DRFL] Round {server_round} train losses: {loss_values.tolist()}")

        clustering = DBSCAN(eps=0.5, min_samples=1).fit(grad_matrix)
        cluster_labels = clustering.labels_

        # With very few clients, DBSCAN may isolate every client.
        # In that case, fall back to a single shared cluster for stability.
        # Fallback: if every client becomes its own cluster, use all clients
        if len(set(cluster_labels.tolist())) == len(client_gradients):
            print(f"[DRFL] Round {server_round} fallback: each client isolated, using all clients")
            cluster_labels = np.zeros(len(client_gradients), dtype=int)

        print(f"[DRFL] Round {server_round} cluster labels: {cluster_labels.tolist()}")

        # ---------------------------------------------
        # 3. Compute mean loss per cluster
        # ---------------------------------------------
        unique_labels = sorted(set(cluster_labels.tolist()))
        cluster_mean_losses = {}

        for label in unique_labels:
            member_idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == label]
            mean_loss = float(np.mean([loss_values[i] for i in member_idxs]))
            cluster_mean_losses[label] = mean_loss

        print(f"[DRFL] Round {server_round} cluster mean losses: {cluster_mean_losses}")

        # ---------------------------------------------
        # 4. Select worst cluster
        # ---------------------------------------------
        worst_cluster = max(cluster_mean_losses, key=cluster_mean_losses.get)
        selected_idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == worst_cluster]

        print(f"[DRFL] Round {server_round} worst cluster: {worst_cluster}")
        print(f"[DRFL] Round {server_round} selected client indices: {selected_idxs}")

        # ---------------------------------------------
        # 5. Build weights within selected cluster
        # ---------------------------------------------
        selected_losses = np.array(
            [loss_values[i] for i in selected_idxs],
            dtype=np.float32,
        )
        selected_weights = selected_losses / (selected_losses.sum() + 1e-12)

        print(f"[DRFL] Round {server_round} selected losses: {selected_losses.tolist()}")
        print(f"[DRFL] Round {server_round} selected weights: {selected_weights.tolist()}")

        # ---------------------------------------------
        # 6. Aggregate parameters using selected cluster only
        # ---------------------------------------------
        # Step 6a: Build robust gradient from selected cluster
        selected_gradients = [client_gradients[i] for i in selected_idxs]

        robust_gradient = sum(
            selected_weights[j] * selected_gradients[j]
            for j in range(len(selected_idxs))
        )

        robust_grad_norm = float(np.linalg.norm(robust_gradient))
        print(f"[DRFL] Round {server_round} robust gradient norm: {robust_grad_norm}")

        # Normalize robust gradient for stability
        if robust_grad_norm > 1e-12:
            robust_gradient = robust_gradient / robust_grad_norm

        # Step 6b: Apply gradient update to global parameters
        learning_rate = 0.001
        
        global_ndarrays = parameters_to_ndarrays(self.initial_parameters)

        flat_params = np.concatenate([p.flatten() for p in global_ndarrays])
        updated_flat = flat_params - learning_rate * robust_gradient

        # reconstruct parameters
        new_ndarrays = []
        offset = 0

        for param in global_ndarrays:
            size = param.size
            new_param = updated_flat[offset:offset + size].reshape(param.shape)
            new_ndarrays.append(new_param)
            offset += size

        aggregated_parameters = ndarrays_to_parameters(new_ndarrays)
        self.initial_parameters = aggregated_parameters

        # ---------------------------------------------
        # 7. Keep metrics aggregation from parent if available
        # ---------------------------------------------
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        metrics_aggregated["drfl_max_loss"] = float(np.max(loss_values))
        metrics_aggregated["drfl_mean_loss"] = float(np.mean(loss_values))

        return aggregated_parameters, metrics_aggregated
