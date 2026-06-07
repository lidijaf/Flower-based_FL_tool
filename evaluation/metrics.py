import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def classification_metrics(
    y_true,
    y_pred,
    average: str = "weighted",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def anomaly_metrics(
    y_true,
    y_pred,
    scores: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if scores is not None:
        scores = np.asarray(scores)

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        except ValueError:
            metrics["pr_auc"] = None

    return metrics


def summarize_values(values) -> Dict[str, Optional[float]]:
    values = np.asarray(list(values), dtype=float)

    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "final": None,
        }

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "final": float(values[-1]),
    }


def summarize_round_metrics(round_metrics, metric_name: str) -> Dict[str, Optional[float]]:
    values = []

    for item in round_metrics:
        if metric_name in item:
            values.append(item[metric_name])

    return summarize_values(values)


def save_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
