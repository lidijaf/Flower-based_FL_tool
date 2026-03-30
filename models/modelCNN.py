from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from utils import get_cfg


class CNN_MNIST(nn.Module):
    """CNN model for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
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


class CNN_CIFAR10(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN_FMNIST(nn.Module):
    """CNN model for FashionMNIST."""

    def __init__(self) -> None:
        super().__init__()
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def _get_device(cfg) -> torch.device:
    device_name = cfg.get("device") or "cpu"
    return torch.device(device_name)


def train_CNN(
    model: nn.Module,
    trainloader: DataLoader,
    cfg,
    theta_params=None,
    global_model_params=None,
    first_round=None,
):
    """Train CNN model locally.

    Returns:
        fedavg -> avg_train_loss
        pfedme -> (global_params, avg_train_loss)
    """
    algorithm = cfg.get("algorithm")
    device = _get_device(cfg)
    epochs = cfg.get("epochs", 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    if algorithm == "fedavg":
        model.train()
        model.to(device)

        for _ in range(epochs):
            for images, labels in trainloader:
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
        global_params = [val.detach().clone() for val in model.parameters()]
        model.train()
        model.to(device)

        pfedme_cfg = cfg["config_fit_pfedme"]
        local_rounds = pfedme_cfg.get("local_rounds")
        local_iterations = pfedme_cfg.get("local_iterations")
        lambda_reg = pfedme_cfg.get("lambda_reg")
        mu = pfedme_cfg.get("mu")
        new = pfedme_cfg.get("new")

        for _ in range(local_rounds):
            if not new:
                data_iterator = iter(trainloader)
                data, target = next(data_iterator)
                data, target = data.to(device), target.to(device)

                for _ in range(local_iterations):
                    optimizer.zero_grad()
                    penalty_term = sum(
                        (local_w - global_w).norm(2) ** 2
                        for local_w, global_w in zip(model.parameters(), global_params)
                    )
                    loss = criterion(model(data), target) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    count += 1
            else:
                for _ in range(local_iterations):
                    data_iterator = iter(trainloader)
                    data, target = next(data_iterator)
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    penalty_term = sum(
                        (local_w - global_w).norm(2) ** 2
                        for local_w, global_w in zip(model.parameters(), global_params)
                    )
                    loss = criterion(model(data), target) + (lambda_reg / 2) * penalty_term
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    count += 1

            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param -= mu * lambda_reg * (global_param - param)

        avg_train_loss = total_loss / count if count > 0 else 0.0
        return global_params, avg_train_loss

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def test_CNN(model: nn.Module, testloader) -> Tuple[float, float, float, float, float]:
    """Evaluate CNN model on the full test set.

    Returns:
        total_loss, accuracy, precision, recall, f1_score
    """
    cfg = get_cfg("conf/config_common.yaml")
    device = _get_device(cfg)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)

    zero = 0.0
    return total_loss, accuracy, zero, zero, zero
