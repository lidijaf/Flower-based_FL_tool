import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.base import PreprocessingContext
from data_preprocessing.recipes.act import build_act_anomaly_pipeline


context = PreprocessingContext()

pipeline = build_act_anomaly_pipeline(
    input_path="data/raw/run1.json",
    output_dir="data/processed/act_anomaly",
)

bundle = pipeline.fit_transform(None, context)

print("Anomaly injection metadata:")
print(context.metadata.get("anomaly_injection"))

print("Number of split client datasets:", len(bundle.splits))

first_client = list(bundle.splits.keys())[0]
print("First client:", first_client)
print("Train anomalies:", bundle.splits[first_client]["train"]["is_anomaly"].sum())
print("Val anomalies:", bundle.splits[first_client]["val"]["is_anomaly"].sum())
print("Test anomalies:", bundle.splits[first_client]["test"]["is_anomaly"].sum())

print("Split summary sample:")
print(context.metadata["anomaly_split_summary"][first_client])

total_train_anom = 0
total_val_anom = 0
total_test_anom = 0

for client_id, stats in context.metadata["anomaly_split_summary"].items():
    total_train_anom += stats["train_anomalies"]
    total_val_anom += stats["val_anomalies"]
    total_test_anom += stats["test_anomalies"]

print("Global anomaly totals:")
print("  Train anomalies:", total_train_anom)
print("  Val anomalies:", total_val_anom)
print("  Test anomalies:", total_test_anom)
print("  Total anomalies in splits:", total_train_anom + total_val_anom + total_test_anom)
