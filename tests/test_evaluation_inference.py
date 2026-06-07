import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.inference import (
    predict_classification,
    predict_probabilities,
    reconstruction_errors,
)


def main():
    # Classification test
    x_cls = torch.randn(12, 4)
    y_cls = torch.tensor([0, 1, 2] * 4)

    cls_loader = DataLoader(
        TensorDataset(x_cls, y_cls),
        batch_size=4,
        shuffle=False,
    )

    cls_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )

    preds, labels = predict_classification(cls_model, cls_loader)
    probs, _ = predict_probabilities(cls_model, cls_loader)

    print("Classification predictions shape:", preds.shape)
    print("Classification labels shape:", labels.shape)
    print("Probability shape:", probs.shape)

    # Autoencoder test
    x_ae = torch.randn(10, 5)
    y_ae = torch.tensor([0, 1] * 5)

    ae_loader = DataLoader(
        TensorDataset(x_ae, y_ae),
        batch_size=2,
        shuffle=False,
    )

    ae_model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 5),
    )

    errors, ae_labels = reconstruction_errors(ae_model, ae_loader)

    print("Reconstruction errors shape:", errors.shape)
    print("AE labels shape:", ae_labels.shape)
    print("First errors:", errors[:3])


if __name__ == "__main__":
    main()
