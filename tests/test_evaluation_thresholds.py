import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from evaluation.thresholds import (
    percentile_threshold,
    mean_std_threshold,
    predict_anomalies,
    save_threshold,
    load_threshold,
)


def main():
    errors = [
        0.1,
        0.2,
        0.15,
        0.3,
        0.25,
        0.18,
        0.22,
        1.5,
        2.0,
    ]

    p_thresh = percentile_threshold(errors, percentile=90)
    print("Percentile threshold:", p_thresh)

    ms_thresh = mean_std_threshold(errors, std_factor=2)
    print("Mean/std threshold:", ms_thresh)

    preds = predict_anomalies(errors, p_thresh)
    print("Predictions:", preds)

    path = "outputs/test_metrics/threshold.json"

    save_threshold(p_thresh, path)

    loaded = load_threshold(path)

    print("Loaded threshold:", loaded)


if __name__ == "__main__":
    main()
