import csv
import json
import os
from statistics import mean

import matplotlib.pyplot as plt

def read_jsonl(path):
    if not os.path.exists(path):
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def summarize_values(records, key):
    values = [
        record[key]
        for record in records
        if key in record and isinstance(record[key], (int, float))
    ]

    if not values:
        return {}

    return {
        f"{key}_min": min(values),
        f"{key}_max": max(values),
        f"{key}_mean": mean(values),
        f"{key}_final": values[-1],
    }


def summarize_client_metrics(records):
    fit_records = [r for r in records if r.get("event") == "fit"]
    evaluate_records = [r for r in records if r.get("event") == "evaluate"]

    summary = {
        "num_client_records": len(records),
        "num_fit_records": len(fit_records),
        "num_evaluate_records": len(evaluate_records),
    }

    for key in [
        "fit_time_sec",
        "evaluate_time_sec",
        "delta_memory_mb",
        "peak_memory_mb",
        "communication_total_mb",
        "communication_total_bytes",
        "communication_num_parameters",
    ]:
        summary.update(summarize_values(records, key))

    return summary


def summarize_server_metrics(records):
    training_records = [r for r in records if r.get("event") == "server_training"]
    aggregate_fit_records = [r for r in records if r.get("event") == "server_aggregate_fit"]
    aggregate_eval_records = [r for r in records if r.get("event") == "server_aggregate_evaluate"]

    summary = {
        "num_server_records": len(records),
        "num_server_training_records": len(training_records),
        "num_server_aggregate_fit_records": len(aggregate_fit_records),
        "num_server_aggregate_evaluate_records": len(aggregate_eval_records),
    }

    for key in [
        "training_time_sec",
        "aggregation_time_sec",
        "delta_memory_mb",
        "peak_memory_mb",
        "loss",
    ]:
        summary.update(summarize_values(records, key))

    return summary


def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in sorted(data.items()):
            writer.writerow([key, value])

def plot_metric(records, key, output_path, title, ylabel):
    values = [
        record[key]
        for record in records
        if key in record and isinstance(record[key], (int, float))
    ]

    if not values:
        return None

    x = list(range(1, len(values) + 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()
    plt.plot(x, values, marker="o")
    plt.title(title)
    plt.xlabel("Record")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_event_metric(records, event, key, output_path, title, ylabel):
    filtered = [
        record
        for record in records
        if record.get("event") == event
        and key in record
        and isinstance(record[key], (int, float))
    ]

    if not filtered:
        return None

    x = [
        record.get("round", index + 1)
        for index, record in enumerate(filtered)
    ]
    y = [record[key] for record in filtered]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def generate_monitoring_report(
    monitoring_dir="outputs/monitoring",
    output_dir="outputs/reports",
):
    client_records = []

    if os.path.exists(monitoring_dir):
        for filename in os.listdir(monitoring_dir):
            if filename.startswith("client") and filename.endswith(".jsonl"):
                path = os.path.join(monitoring_dir, filename)
                client_records.extend(read_jsonl(path))

    server_path = os.path.join(monitoring_dir, "server_metrics.jsonl")
    server_records = read_jsonl(server_path)

    client_summary = summarize_client_metrics(client_records)
    server_summary = summarize_server_metrics(server_records)

    report = {
        "client_summary": client_summary,
        "server_summary": server_summary,
    }

    os.makedirs(output_dir, exist_ok=True)

    write_json(report, os.path.join(output_dir, "monitoring_summary.json"))

    plots_dir = os.path.join(output_dir, "plots")

    generated_plots = {
        "client_fit_time": plot_event_metric(
            client_records,
            event="fit",
            key="fit_time_sec",
            output_path=os.path.join(plots_dir,  "client_fit_time.png"),
            title="Client Fit Time",
            ylabel="Seconds",
        ),
        "client_evaluate_time": plot_event_metric(
            client_records,
            event="evaluate",
            key="evaluate_time_sec",
            output_path=os.path.join(plots_dir, "client_evaluate_time.png"),
            title="Client Evaluation Time",
            ylabel="Seconds",
        ),
        "client_peak_memory": plot_metric(
            client_records,
            key="peak_memory_mb",
            output_path=os.path.join(plots_dir, "client_peak_memory.png"),
            title="Client Peak Memory",
            ylabel="MB",
        ),
        "client_communication": plot_event_metric(
            client_records,
            event="fit",
            key="communication_total_mb",
            output_path=os.path.join(plots_dir, "client_communication_mb.png"),
            title="Client Communication Volume",
            ylabel="MB",
        ),
        "server_training_time": plot_event_metric(
            server_records,
            event="server_training",
            key="training_time_sec",
            output_path=os.path.join(plots_dir, "server_training_time.png"),
            title="Server Training Time",
            ylabel="Seconds",
        ),
        "server_fit_aggregation_time": plot_event_metric(
            server_records,
            event="server_aggregate_fit",
            key="aggregation_time_sec",
            output_path=os.path.join(plots_dir, "server_fit_aggregation_time.png"),
            title="Server Fit Aggregation Time",
            ylabel="Seconds",
        ),
        "server_evaluate_aggregation_time": plot_event_metric(
            server_records,
            event="server_aggregate_evaluate",
            key="aggregation_time_sec",
            output_path=os.path.join(plots_dir, "server_evaluate_aggregation_time.png"),
            title="Server Evaluation Aggregation Time",
            ylabel="Seconds",
        ),
        "server_peak_memory": plot_metric(
            server_records,
            key="peak_memory_mb",
            output_path=os.path.join(plots_dir, "server_peak_memory.png"),
            title="Server Peak Memory",
            ylabel="MB",
        ),
    }

    report["plots"] = {
        key: value
        for key, value in generated_plots.items()
        if value is not None
    }

    write_json(report, os.path.join(output_dir, "monitoring_summary.json"))

    flat_report = {}

    for key, value in client_summary.items():
        flat_report[f"client_{key}"] = value

    for key, value in server_summary.items():
        flat_report[f"server_{key}"] = value

    write_csv(flat_report, os.path.join(output_dir, "monitoring_summary.csv"))

    return report


if __name__ == "__main__":
    report = generate_monitoring_report()
    print(json.dumps(report, indent=2))
