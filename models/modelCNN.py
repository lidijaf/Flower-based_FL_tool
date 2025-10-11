import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple

from utils import get_cfg

# Note: The model and functions defined here do not have any FL-specific components.


# MNIST MODEL
class CNN_MNIST(nn.Module):
    """A customizable CNN for different datasets."""

    def __init__(self) -> None:
        super(CNN_MNIST, self).__init__()

        # Define the architecture for MNIST
        self.conv1 = nn.Conv2d(1, 6, 5)  # Grayscale images
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# CIFAR10 model

class CNN_CIFAR10(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# FashionMNIST model

class CNN_FMNIST(nn.Module):

    def __init__(self):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_CNN(model: nn.Module, trainloader: DataLoader, cfg, theta_params=None, global_model_params=None,first_round=None):
    # Common configuration
    #cfg = get_cfg()
    algorithm = cfg.get("algorithm")
    device = torch.device(cfg.get("device"))
    epochs = cfg.get("epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate"))

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    if algorithm == "fedavg":
        # Train the model
        model.train()
        model.to(device)
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
        avg_train_loss = total_loss / count if count > 0 else 0.0
        return avg_train_loss

    elif algorithm == "pfedme":
        # Copy the parameters obtained from the server (global model),
        # this is done because of the penalty term (mozda obrisati)
        global_params = [val.detach().clone() for val in model.parameters()]
        model.train()
        model.to(device)

        # pfedme-specific configurations
        local_rounds = cfg["config_fit_pfedme"].get("local_rounds")
        local_iterations = cfg["config_fit_pfedme"].get("local_iterations")
        lambda_reg = cfg["config_fit_pfedme"].get("lambda_reg")
        mu = cfg["config_fit_pfedme"].get("mu")
        new = cfg["config_fit_pfedme"].get("new")

        for r in range(local_rounds):
            if not new:  # The same batch is reused for all local iterations (i loop) within that local round.
                data_iterator = iter(trainloader)
                data, target = next(data_iterator)
                data, target = data.to(device), target.to(device)
                for i in range(local_iterations):
                    optimizer.zero_grad()
                    penalty_term = sum((lw - gw).norm(2) ** 2 for lw, gw in zip(model.parameters(), global_params))
                    loss = criterion(model(data), target) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
            else:  # A new batch is sampled for every local iteration (i loop).
                for i in range(local_iterations):
                    data_iterator = iter(trainloader)
                    data, target = next(data_iterator)
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    penalty_term = sum((lw - gw).norm(2) ** 2 for lw, gw in zip(model.parameters(), global_params))
                    loss = criterion(model(data), target) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1

            # at the end of each local round after local_iterations happen,
            # update the local(global_params) model according to the personalized model
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param -= mu * lambda_reg * (global_param - param)

        avg_train_loss = total_loss / count if count > 0 else 0.0
        return global_params, avg_train_loss
        # at the end we are left with a personalized model (model.parameters()),
        # and local (global_params),

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def test_CNN(model: nn.Module, testloader) -> Tuple[float, float]:
    """Validate the network on the entire test set and report loss and accuracy."""
    cfg = get_cfg()
    device = torch.device(cfg.get("device"))
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    zero = 0
    return total_loss, accuracy, zero, zero, zero
