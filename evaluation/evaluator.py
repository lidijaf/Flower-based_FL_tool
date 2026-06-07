from typing import Any, Dict, Optional

from evaluation.inference import (
    predict_classification,
    predict_probabilities,
    reconstruction_errors,
)
from evaluation.metrics import classification_metrics, anomaly_metrics
from evaluation.thresholds import (
    percentile_threshold,
    mean_std_threshold,
    predict_anomalies,
)
from evaluation.results import save_evaluation_report


def evaluate_classification_model(
    model,
    dataloader,
    output_dir: Optional[str] = None,
    device=None,
    average: str = "weighted",
    save_report: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
):
    predictions, y_true = predict_classification(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    probabilities, _ = predict_probabilities(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    if y_true is None:
        metrics = {}
    else:
        metrics = classification_metrics(
            y_true=y_true,
            y_pred=predictions,
            average=average,
        )

    if save_report and output_dir is not None:
        save_evaluation_report(
            metrics=metrics,
            output_dir=output_dir,
            predictions=predictions,
            y_true=y_true,
            probabilities=probabilities,
            metadata=metadata,
        )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "y_true": y_true,
        "probabilities": probabilities,
    }


def evaluate_autoencoder_model(
    model,
    dataloader,
    output_dir: Optional[str] = None,
    device=None,
    threshold: Optional[float] = None,
    threshold_method: str = "percentile",
    percentile: float = 95,
    std_factor: float = 3.0,
    reduction: str = "mean",
    save_report: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
):
    errors, y_true = reconstruction_errors(
        model=model,
        dataloader=dataloader,
        device=device,
        reduction=reduction,
    )

    if threshold is None:
        if threshold_method == "percentile":
            threshold = percentile_threshold(errors, percentile=percentile)
        elif threshold_method == "mean_std":
            threshold = mean_std_threshold(errors, std_factor=std_factor)
        else:
            raise ValueError("threshold_method must be 'percentile' or 'mean_std'")

    predictions = predict_anomalies(errors, threshold)

    if y_true is None:
        metrics = {}
    else:
        metrics = anomaly_metrics(
            y_true=y_true,
            y_pred=predictions,
            scores=errors,
        )

    report_metadata = metadata or {}
    report_metadata = {
        **report_metadata,
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "reduction": reduction,
    }

    if save_report and output_dir is not None:
        save_evaluation_report(
            metrics=metrics,
            output_dir=output_dir,
            predictions=predictions,
            y_true=y_true,
            scores=errors,
            metadata=report_metadata,
        )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "y_true": y_true,
        "scores": errors,
        "threshold": threshold,
    }
