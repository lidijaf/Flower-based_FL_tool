import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.results import (
    create_results_dir,
    save_json,
    load_json,
    save_predictions_csv,
    save_numpy_array,
    save_evaluation_report,
)


def main():
    output_dir = create_results_dir(
        base_dir="outputs/test_metrics",
        run_name="results_test",
    )

    metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.85,
        "f1": 0.82,
    }

    json_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, json_path)

    loaded = load_json(json_path)
    print("Loaded JSON:", loaded)

    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    scores = [0.1, 0.9, 0.4, 0.2]

    save_predictions_csv(
        y_pred=y_pred,
        y_true=y_true,
        scores=scores,
        path=os.path.join(output_dir, "predictions.csv"),
    )

    save_numpy_array(scores, os.path.join(output_dir, "scores.npy"))

    save_evaluation_report(
        metrics=metrics,
        output_dir=output_dir,
        predictions=y_pred,
        y_true=y_true,
        scores=scores,
        metadata={
            "model": "test_model",
            "dataset": "test_dataset",
        },
    )

    print("Saved files in:", output_dir)
    print("Files:", os.listdir(output_dir))


if __name__ == "__main__":
    main()
