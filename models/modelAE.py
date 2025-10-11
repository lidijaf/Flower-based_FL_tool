import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from utils import get_cfg
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 14),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)  # reconstruct original input
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model: nn.Module, dataloader, cfg):
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    model.train()
    total_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)  # flatten target
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, x_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)

def train(model: nn.Module, trainloader, cfg):
    """Train the autoencoder and return average batch loss (and global_params for pFedMe)."""
    algorithm = cfg.get("algorithm")
    device = torch.device(cfg.get("device"))
    epochs = cfg.get("epochs")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get("learning_rate"),
        momentum=cfg.get("momentum")
    )

    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    model.to(device)
    model.train()

    if algorithm == "fedavg":
        for epoch in range(epochs):
            for x, _ in trainloader:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)  # Flatten input
                optimizer.zero_grad()
                outputs = model(x_flat)
                loss = criterion(outputs, x_flat)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_training_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_training_loss

    elif algorithm == "pfedme":
        global_params = [val.detach().clone() for val in model.parameters()]
        local_rounds = cfg["config_fit_pfedme"].get("local_rounds")
        local_iterations = cfg["config_fit_pfedme"].get("local_iterations")
        lambda_reg = cfg["config_fit_pfedme"].get("lambda_reg")
        mu = cfg["config_fit_pfedme"].get("mu")
        new = cfg["config_fit_pfedme"].get("new")

        for r in range(local_rounds):
            if not new:
                data_iterator = iter(trainloader)
                data, _ = next(data_iterator)
                data = data.to(device)
                data_flat = data.view(data.size(0), -1)
                for _ in range(local_iterations):
                    optimizer.zero_grad()
                    penalty_term = sum((lw - gw).norm(2) ** 2 for lw, gw in zip(model.parameters(), global_params))
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
                    penalty_term = sum((lw - gw).norm(2) ** 2 for lw, gw in zip(model.parameters(), global_params))
                    loss = criterion(model(data_flat), data_flat) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1

            # Update global parameters
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param -= mu * lambda_reg * (global_param - param)

        avg_training_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return global_params, avg_training_loss

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

def vali(model: nn.Module, valloader: DataLoader, trainloader: DataLoader, cfg):
    """Validate the autoencoder and return average loss and anomaly threshold."""
    device = torch.device(cfg.get("device"))
    criterion = nn.MSELoss(reduction='none')  # Per-sample loss

    model.eval()
    model.to(device)

    # --- Compute reconstruction errors on training data to set threshold ---
    reconstruction_errors = []

    with torch.no_grad():
        for x, _ in trainloader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)  # Flatten
            outputs = model(x_flat)
            losses = criterion(outputs, x_flat).mean(dim=1)  # Per-sample loss
            reconstruction_errors.extend(losses.cpu().numpy())

    mean_err = np.mean(reconstruction_errors)
    std_err = np.std(reconstruction_errors)
    threshold = mean_err + 3 * std_err  # Adjustable factor
    print(f"[Threshold] mean: {mean_err:.4f}, std: {std_err:.4f}, threshold = {threshold:.4f}")

    # --- Compute average validation loss ---
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, _ in valloader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)  # Flatten
            outputs = model(x_flat)
            batch_losses = criterion(outputs, x_flat).mean(dim=1)  # Per-sample mean
            batch_loss_mean = batch_losses.mean()  # Batch mean

            total_loss += batch_loss_mean.item()
            num_batches += 1

    avg_validation_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_validation_loss, threshold

def test(model: nn.Module, testloader, threshold: float, cfg) -> Tuple[float, float, float, float, float]:
    """Validate the network on the test set and detect anomalies."""
    device = torch.device(cfg.get("device"))
    criterion = nn.MSELoss(reduction='none')

    total_loss = 0.0
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in testloader:
            x, labels = x.to(device), labels.to(device)

            # Flatten input for linear layers
            x_flat = x.view(x.size(0), -1)
            outputs = model(x_flat)

            # Compute per-sample loss
            batch_losses = criterion(outputs, x_flat).mean(dim=1)

            # Predict anomalies
            predicted_anomalies = (batch_losses > threshold).int()
            true_labels = labels.int()

            # Collect predictions and labels
            all_preds.extend(predicted_anomalies.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

            total_loss += batch_losses.sum().item()

    avg_test_loss = total_loss / len(testloader.dataset)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")

    return avg_test_loss, accuracy, precision, recall, f_score

