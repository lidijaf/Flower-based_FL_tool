from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 14),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def _get_device(cfg) -> torch.device:
    device_name = cfg.get("device") or "cpu"
    return torch.device(device_name)


def train(model: nn.Module, trainloader, cfg):
    """Train the autoencoder.

    Returns:
        fedavg -> avg_training_loss
        pfedme -> (global_params, avg_training_loss)
    """
    algorithm = cfg.get("algorithm")
    device = _get_device(cfg)
    epochs = cfg.get("epochs", 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        momentum=cfg.get("momentum", 0.0),
    )
    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    model.to(device)
    model.train()

    if algorithm == "fedavg":
        for _ in range(epochs):
            for x, _ in trainloader:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)

                optimizer.zero_grad()
                outputs = model(x_flat)
                loss = criterion(outputs, x_flat)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_training_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_training_loss

    if algorithm == "pfedme":
        global_params = [param.detach().clone() for param in model.parameters()]
        pfedme_cfg = cfg["config_fit_pfedme"]

        local_rounds = pfedme_cfg.get("local_rounds")
        local_iterations = pfedme_cfg.get("local_iterations")
        lambda_reg = pfedme_cfg.get("lambda_reg")
        mu = pfedme_cfg.get("mu")
        new = pfedme_cfg.get("new")

        for _ in range(local_rounds):
            if not new:
                data_iterator = iter(trainloader)
                data, _ = next(data_iterator)
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)

                for _ in range(local_iterations):
                    optimizer.zero_grad()
                    penalty_term = sum(
                        (local_w - global_w).norm(2) ** 2
                        for local_w, global_w in zip(model.parameters(), global_params)
                    )
                    loss = criterion(model(data_flat), data_flat) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
            else:
                for _ in range(local_iterations):
                    data_iterator = iter(trainloader)
                    data, _ = next(data_iterator)
                    data = data.to(device)
                    data_flat = data.view(data.size(0), -1)

                    optimizer.zero_grad()
                    penalty_term = sum(
                        (local_w - global_w).norm(2) ** 2
                        for local_w, global_w in zip(model.parameters(), global_params)
                    )
                    loss = criterion(model(data_flat), data_flat) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param -= mu * lambda_reg * (global_param - param)

        avg_training_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return global_params, avg_training_loss

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def vali(
    model: nn.Module,
    valloader: DataLoader,
    trainloader: DataLoader,
    cfg,
):
    """Validate the autoencoder and return average loss and anomaly threshold."""
    device = _get_device(cfg)
    criterion = nn.MSELoss(reduction="none")

    model.eval()
    model.to(device)

    reconstruction_errors = []

    with torch.no_grad():
        for x, _ in trainloader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            outputs = model(x_flat)
            losses = criterion(outputs, x_flat).mean(dim=1)
            reconstruction_errors.extend(losses.cpu().numpy())

    mean_err = np.mean(reconstruction_errors)
    std_err = np.std(reconstruction_errors)
    threshold = mean_err + 3 * std_err

    print(
        f"[Threshold] mean: {mean_err:.4f}, std: {std_err:.4f}, "
        f"threshold = {threshold:.4f}"
    )

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, _ in valloader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            outputs = model(x_flat)

            batch_losses = criterion(outputs, x_flat).mean(dim=1)
            batch_loss_mean = batch_losses.mean()

            total_loss += batch_loss_mean.item()
            num_batches += 1

    avg_validation_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_validation_loss, threshold


def test(
    model: nn.Module,
    testloader,
    threshold: float,
    cfg,
) -> Tuple[float, float, float, float, float]:
    """Evaluate the autoencoder on the test set and detect anomalies."""
    device = _get_device(cfg)
    criterion = nn.MSELoss(reduction="none")

    total_loss = 0.0
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in testloader:
            x = x.to(device)
            labels = labels.to(device)

            x_flat = x.view(x.size(0), -1)
            outputs = model(x_flat)

            batch_losses = criterion(outputs, x_flat).mean(dim=1)
            predicted_anomalies = (batch_losses > threshold).int()

            true_labels = labels.int()

            all_preds.extend(predicted_anomalies.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

            total_loss += batch_losses.sum().item()

    avg_test_loss = total_loss / len(testloader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )

    print(
        f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, "
        f"Recall : {recall:.4f}, F-score : {f_score:.4f}"
    )

    return avg_test_loss, accuracy, precision, recall, f_score
