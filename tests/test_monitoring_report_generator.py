import os
import sys
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.report_generator import generate_monitoring_report


def main():
    monitoring_dir = "outputs/test_monitoring"
    output_dir = "outputs/test_reports"

    os.makedirs(monitoring_dir, exist_ok=True)

    client_path = os.path.join(monitoring_dir, "client1_client_metrics.jsonl")
    server_path = os.path.join(monitoring_dir, "server_metrics.jsonl")

    with open(client_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "fit",
            "fit_time_sec": 1.2,
            "delta_memory_mb": 10.0,
            "peak_memory_mb": 100.0,
            "communication_total_mb": 2.5,
        }) + "\n")
        f.write(json.dumps({
            "event": "evaluate",
            "evaluate_time_sec": 0.4,
            "delta_memory_mb": 2.0,
            "peak_memory_mb": 105.0,
        }) + "\n")

    with open(server_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "server_aggregate_fit",
            "round": 1,
            "aggregation_time_sec": 0.01,
            "peak_memory_mb": 200.0,
        }) + "\n")
        f.write(json.dumps({
            "event": "server_training",
            "training_time_sec": 5.0,
            "peak_memory_mb": 210.0,
        }) + "\n")

    report = generate_monitoring_report(
        monitoring_dir=monitoring_dir,
        output_dir=output_dir,
    )

    print(json.dumps(report, indent=2))
    print("Generated files:", os.listdir(output_dir))


if __name__ == "__main__":
    main()
