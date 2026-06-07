from typing import Tuple

import numpy as np
import torch


def get_device(device=None):
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model_outputs(model, dataloader, device=None):
    device = get_device(device)
    model = model.to(device)
    model.eval()

    outputs = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x = batch
                y = None

            x = x.to(device)
            out = model(x)

            outputs.append(out.detach().cpu())

            if y is not None:
                labels.append(y.detach().cpu())

    outputs = torch.cat(outputs, dim=0)

    if labels:
        labels = torch.cat(labels, dim=0)
    else:
        labels = None

    return outputs, labels


def predict_classification(model, dataloader, device=None) -> Tuple[np.ndarray, np.ndarray]:
    outputs, labels = run_model_outputs(model, dataloader, device=device)

    preds = torch.argmax(outputs, dim=1).numpy()
    y_true = labels.numpy() if labels is not None else None

    return preds, y_true


def predict_probabilities(model, dataloader, device=None) -> Tuple[np.ndarray, np.ndarray]:
    outputs, labels = run_model_outputs(model, dataloader, device=device)

    probs = torch.softmax(outputs, dim=1).numpy()
    y_true = labels.numpy() if labels is not None else None

    return probs, y_true


def reconstruction_errors(model, dataloader, device=None, reduction="mean") -> Tuple[np.ndarray, np.ndarray]:
    device = get_device(device)
    model = model.to(device)
    model.eval()

    errors = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x = batch
                y = None

            x = x.to(device)
            reconstructed = model(x)

            per_sample_error = (x - reconstructed) ** 2

            if reduction == "mean":
                per_sample_error = per_sample_error.view(per_sample_error.size(0), -1).mean(dim=1)
            elif reduction == "sum":
                per_sample_error = per_sample_error.view(per_sample_error.size(0), -1).sum(dim=1)
            else:
                raise ValueError("reduction must be 'mean' or 'sum'")

            errors.append(per_sample_error.detach().cpu())

            if y is not None:
                labels.append(y.detach().cpu())

    errors = torch.cat(errors, dim=0).numpy()

    if labels:
        labels = torch.cat(labels, dim=0).numpy()
    else:
        labels = None

    return errors, labels
