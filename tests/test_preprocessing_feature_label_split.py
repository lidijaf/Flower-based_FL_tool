import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.pipeline import PreprocessingPipeline
from data_preprocessing.loaders.json_loader import JSONLoaderStep
from data_preprocessing.transforms.flatten import FlattenJSONStep
from data_preprocessing.transforms.sort import SortByColumnsStep
from data_preprocessing.transforms.time import DeltaTimeFeatureStep
from data_preprocessing.transforms.features import StringLengthFeatureStep
from data_preprocessing.transforms.encode import OneHotEncodeStep
from data_preprocessing.transforms.scale import StandardScaleStep
from data_preprocessing.transforms.anomalies import InjectSyntheticAnomaliesStep
from data_preprocessing.transforms.select import SeparateFeaturesAndLabelsStep
from data_preprocessing.partitioning.by_group import PartitionByGroupStep
from data_preprocessing.splitting.anomaly_aware import AnomalyAwareSplitStep
from data_preprocessing.export.separated_export import ExportSeparatedSplitsStep
from data_preprocessing.base import PreprocessingContext


def flatten_act_event(e):
    meta = e.get("meta")
    payload = e.get("payload")

    if not isinstance(meta, dict):
        meta = {}

    if not isinstance(payload, dict):
        payload = {}

    tags = meta.get("tags", [])
    if isinstance(tags, list):
        tags = ",".join(str(tag) for tag in tags)
    elif tags is None:
        tags = ""
    else:
        tags = str(tags)

    return {
        "timestampMicros": meta.get("timestampMicros"),
        "appId": meta.get("appId"),
        "stream": meta.get("stream"),
        "tags": tags,
        "offset": meta.get("offset"),
        "eventId": meta.get("eventId"),
        "streamName": payload.get("streamName"),
        "streamNr": payload.get("streamNr"),
        "payload_keys": ",".join(payload.keys()),
        "payload_raw": str(payload),
        "type": e.get("type"),
    }


pipeline = PreprocessingPipeline([
    JSONLoaderStep("data/raw/run1.json"),
    FlattenJSONStep(flatten_act_event),
    SortByColumnsStep(by=["stream", "timestampMicros"]),
    DeltaTimeFeatureStep(
        timestamp_col="timestampMicros",
        output_col="delta_t",
        group_by="stream",
        scale_factor=1e6,
        fillna_value=0,
    ),
    StringLengthFeatureStep(
        source_col="payload_raw",
        output_col="payload_len",
    ),
    InjectSyntheticAnomaliesStep(
        anomaly_fraction=0.05,
        numeric_cols=["offset", "delta_t", "payload_len"],
        categorical_cols=["streamName", "tags"],
        temporal_cols=["delta_t"],
        label_col="is_anomaly",
        noise_std=0.5,
        shift_multiplier=3.0,
        random_state=42,
    ),
    OneHotEncodeStep(
        columns=["streamName", "tags"],
        drop_first=True,
    ),
    StandardScaleStep(
        columns=["offset", "delta_t", "payload_len"],
        artifact_name="act_scaler",
    ),
    PartitionByGroupStep(group_col="stream"),
    AnomalyAwareSplitStep(
        label_col="is_anomaly",
        normal_value=0,
        anomaly_value=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sort_by=["timestampMicros"],
        anomaly_distribution="half_half",
    ),
    SeparateFeaturesAndLabelsStep(
    label_cols=["is_anomaly"],
    drop_cols=["payload_raw", "appId", "payload_keys", "type"],
    metadata_cols=["timestampMicros", "stream", "eventId"],
    ),
    ExportSeparatedSplitsStep(output_dir="data/processed/act_separated_test"),
])

context = PreprocessingContext()
bundle = pipeline.fit_transform(None, context)

first_client_id = list(bundle.splits.keys())[0]
first_train = bundle.splits[first_client_id]["train"]

print("First client:", first_client_id)
print("Train X shape:", first_train["X"].shape)
print("Train y shape:", first_train["y"].shape if first_train["y"] is not None else None)
print("Train meta shape:", first_train["meta"].shape if first_train["meta"] is not None else None)

print("X columns sample:", first_train["X"].columns[:10].tolist())
print("y columns:", first_train["y"].columns.tolist())
print("meta columns:", first_train["meta"].columns.tolist())
