import json
import os

import numpy as np


def percentile_threshold(errors, percentile=95):
    errors = np.asarray(errors, dtype=float)
    return float(np.percentile(errors, percentile))


def mean_std_threshold(errors, std_factor=3.0):
    errors = np.asarray(errors, dtype=float)

    mean = np.mean(errors)
    std = np.std(errors)

    return float(mean + std_factor * std)


def predict_anomalies(errors, threshold):
    errors = np.asarray(errors, dtype=float)
    return (errors > threshold).astype(int)


def save_threshold(threshold, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)


def load_threshold(path):
    with open(path, "r") as f:
        return json.load(f)["threshold"]
