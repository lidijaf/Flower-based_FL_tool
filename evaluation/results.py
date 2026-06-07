import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def create_results_dir(base_dir: str = "outputs/evaluation", run_name: Optional[str] = None) -> str:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"evaluation_{timestamp}"

    path = os.path.join(base_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions_csv(
    y_pred,
    path: str,
    y_true=None,
    scores=None,
    probabilities=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "y_pred": np.asarray(y_pred),
    }

    if y_true is not None:
        data["y_true"] = np.asarray(y_true)

    if scores is not None:
        data["score"] = np.asarray(scores)

    df = pd.DataFrame(data)

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        for i in range(probabilities.shape[1]):
            df[f"prob_class_{i}"] = probabilities[:, i]

    df.to_csv(path, index=False)


def save_numpy_array(array, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(array))


def save_evaluation_report(
    metrics: Dict[str, Any],
    output_dir: str,
    predictions=None,
    y_true=None,
    scores=None,
    probabilities=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "metrics": metrics,
        "metadata": metadata or {},
    }

    save_json(report, os.path.join(output_dir, "evaluation_report.json"))

    if predictions is not None:
        save_predictions_csv(
            y_pred=predictions,
            y_true=y_true,
            scores=scores,
            probabilities=probabilities,
            path=os.path.join(output_dir, "predictions.csv"),
        )
