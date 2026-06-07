import json
import os
from datetime import datetime


def append_jsonl(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_client_metric(cfg, record, filename="client_metrics.jsonl"):
    output_dir = cfg.get("output_dir", "outputs")
    client_id = cfg.get("client_id", "unknown")

    path = os.path.join(output_dir, "monitoring", f"client{client_id}_{filename}")

    enriched = {
        "timestamp": datetime.now().isoformat(),
        "client_id": client_id,
        **record,
    }

    append_jsonl(enriched, path)
