from data_preprocessing.pipeline import PreprocessingPipeline
from data_preprocessing.loaders.json_loader import JSONLoaderStep
from data_preprocessing.transforms.flatten import FlattenJSONStep
from data_preprocessing.transforms.sort import SortByColumnsStep
from data_preprocessing.transforms.time import DeltaTimeFeatureStep
from data_preprocessing.transforms.features import StringLengthFeatureStep
from data_preprocessing.transforms.encode import OneHotEncodeStep
from data_preprocessing.transforms.scale import StandardScaleStep
from data_preprocessing.transforms.anomalies import InjectSyntheticAnomaliesStep
from data_preprocessing.partitioning.by_group import PartitionByGroupStep
from data_preprocessing.partitioning.balanced_group import PartitionBalancedGroupsStep
from data_preprocessing.splitting.train_val_test import TrainValTestSplitStep
from data_preprocessing.splitting.anomaly_aware import AnomalyAwareSplitStep
from data_preprocessing.export.csv_export import ExportCSVSplitsStep
from data_preprocessing.config_loader import load_json_config
from data_preprocessing.flatten_builders import make_flatten_fn


DEFAULT_ACT_SCHEMA_PATH = "data_preprocessing/configs/act_schema.json"


def load_act_schema(config_path=DEFAULT_ACT_SCHEMA_PATH):
    return load_json_config(config_path)


def build_act_base_steps(input_path, config_path=DEFAULT_ACT_SCHEMA_PATH):
    schema = load_act_schema(config_path)
    flatten_fn = make_flatten_fn(schema)

    steps = [
        JSONLoaderStep(input_path),
        FlattenJSONStep(flatten_fn),
        SortByColumnsStep(
            by=[
                schema["columns"]["group_col"],
                schema["columns"]["timestamp_col"],
            ]
        ),
    ]

    delta_cfg = schema["features"]["delta_time"]
    if delta_cfg.get("enabled", False):
        steps.append(
            DeltaTimeFeatureStep(
                timestamp_col=delta_cfg["timestamp_col"],
                output_col=delta_cfg["output_col"],
                group_by=delta_cfg["group_by"],
                scale_factor=delta_cfg["scale_factor"],
                fillna_value=delta_cfg["fillna_value"],
            )
        )

    strlen_cfg = schema["features"]["string_length"]
    if strlen_cfg.get("enabled", False):
        steps.append(
            StringLengthFeatureStep(
                source_col=strlen_cfg["source_col"],
                output_col=strlen_cfg["output_col"],
            )
        )

    return steps


def build_act_common_feature_steps(config_path=DEFAULT_ACT_SCHEMA_PATH):
    schema = load_act_schema(config_path)

    return [
        OneHotEncodeStep(
            columns=schema["encoding"]["columns"],
            drop_first=schema["encoding"]["drop_first"],
        ),
        StandardScaleStep(
            columns=schema["scaling"]["columns"],
            artifact_name=schema["scaling"]["artifact_name"],
        ),
    ]


def build_act_standard_pipeline(input_path, output_dir, config_path=DEFAULT_ACT_SCHEMA_PATH):
    schema = load_act_schema(config_path)

    steps = build_act_base_steps(input_path, config_path) + build_act_common_feature_steps(config_path) + [
        PartitionByGroupStep(group_col=schema["columns"]["group_col"]),
        TrainValTestSplitStep(
            train_ratio=schema["split"]["train_ratio"],
            val_ratio=schema["split"]["val_ratio"],
            test_ratio=schema["split"]["test_ratio"],
            sort_by=schema["split"]["sort_by"],
        ),
        ExportCSVSplitsStep(output_dir=output_dir),
    ]

    return PreprocessingPipeline(steps)


def build_act_balanced_pipeline(input_path, output_dir, num_clients, config_path=DEFAULT_ACT_SCHEMA_PATH):
    schema = load_act_schema(config_path)

    steps = build_act_base_steps(input_path, config_path) + build_act_common_feature_steps(config_path) + [
        PartitionBalancedGroupsStep(
            group_col=schema["columns"]["group_col"],
            num_clients=num_clients,
        ),
        TrainValTestSplitStep(
            train_ratio=schema["split"]["train_ratio"],
            val_ratio=schema["split"]["val_ratio"],
            test_ratio=schema["split"]["test_ratio"],
            sort_by=schema["split"]["sort_by"],
        ),
        ExportCSVSplitsStep(output_dir=output_dir),
    ]

    return PreprocessingPipeline(steps)


def build_act_anomaly_pipeline(input_path, output_dir, config_path=DEFAULT_ACT_SCHEMA_PATH):
    schema = load_act_schema(config_path)
    anomaly_cfg = schema["anomaly"]

    steps = build_act_base_steps(input_path, config_path) + [
        InjectSyntheticAnomaliesStep(
            anomaly_fraction=anomaly_cfg["anomaly_fraction"],
            numeric_cols=anomaly_cfg["numeric_cols"],
            categorical_cols=anomaly_cfg["categorical_cols"],
            temporal_cols=anomaly_cfg["temporal_cols"],
            label_col=anomaly_cfg["label_col"],
            noise_std=anomaly_cfg["noise_std"],
            shift_multiplier=anomaly_cfg["shift_multiplier"],
            random_state=anomaly_cfg["random_state"],
        ),
    ] + build_act_common_feature_steps(config_path) + [
        PartitionByGroupStep(group_col=schema["columns"]["group_col"]),
        AnomalyAwareSplitStep(
            label_col=anomaly_cfg["label_col"],
            normal_value=anomaly_cfg["normal_value"],
            anomaly_value=anomaly_cfg["anomaly_value"],
            train_ratio=schema["split"]["train_ratio"],
            val_ratio=schema["split"]["val_ratio"],
            test_ratio=schema["split"]["test_ratio"],
            sort_by=schema["split"]["sort_by"],
            anomaly_distribution=anomaly_cfg["anomaly_distribution"],
        ),
        ExportCSVSplitsStep(output_dir=output_dir),
    ]

    return PreprocessingPipeline(steps)
