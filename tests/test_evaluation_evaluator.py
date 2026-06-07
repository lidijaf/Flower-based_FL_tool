import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluator import (
    evaluate_classification_model,
    evaluate_autoencoder_model,
)


def main():
    # Classification evaluator test
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

    cls_result = evaluate_classification_model(
        model=cls_model,
        dataloader=cls_loader,
        output_dir="outputs/test_metrics/evaluator_classification",
        metadata={"test": "classification"},
    )

    print("Classification evaluator metrics:")
    print(cls_result["metrics"])
    print("Classification predictions shape:", cls_result["predictions"].shape)

    # Autoencoder evaluator test
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

    ae_result = evaluate_autoencoder_model(
        model=ae_model,
        dataloader=ae_loader,
        output_dir="outputs/test_metrics/evaluator_autoencoder",
        threshold_method="percentile",
        percentile=80,
        metadata={"test": "autoencoder"},
    )

    print("Autoencoder evaluator metrics:")
    print(ae_result["metrics"])
    print("Autoencoder threshold:", ae_result["threshold"])
    print("Autoencoder scores shape:", ae_result["scores"].shape)


if __name__ == "__main__":
    main()
