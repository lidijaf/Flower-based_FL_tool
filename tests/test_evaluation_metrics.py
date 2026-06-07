import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    classification_metrics,
    anomaly_metrics,
    summarize_values,
    save_metrics_json,
    load_metrics_json,
)


def main():
    y_true_cls = [0, 1, 1, 0, 2, 2]
    y_pred_cls = [0, 1, 0, 0, 2, 1]

    cls_metrics = classification_metrics(y_true_cls, y_pred_cls)
    print("Classification metrics:")
    print(cls_metrics)

    y_true_anom = [0, 0, 1, 1, 0, 1]
    y_pred_anom = [0, 0, 1, 0, 0, 1]
    scores = [0.1, 0.2, 0.9, 0.4, 0.3, 0.8]

    anom_metrics = anomaly_metrics(y_true_anom, y_pred_anom, scores=scores)
    print("Anomaly metrics:")
    print(anom_metrics)

    summary = summarize_values([0.5, 0.6, 0.7])
    print("Summary:")
    print(summary)

    path = "outputs/test_metrics/evaluation_metrics.json"
    save_metrics_json(
        {
            "classification": cls_metrics,
            "anomaly": anom_metrics,
            "summary": summary,
        },
        path,
    )

    loaded = load_metrics_json(path)
    print("Loaded metrics keys:", loaded.keys())


if __name__ == "__main__":
    main()
