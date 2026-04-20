import os
from data_preprocessing.datasets.registry import DATASET_REGISTRY
from data_preprocessing.datasets.unified_registry import UNIFIED_DATASET_REGISTRY
from data_preprocessing.base import PreprocessingContext


def prepare_builtin_dataset(dataset_name: str, data_dir: str):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    os.makedirs(data_dir, exist_ok=True)

    download_fn = DATASET_REGISTRY[dataset_name]
    trainset, testset = download_fn(data_dir)

    print(f"{dataset_name} downloaded to {data_dir}")
    print(f"Train size: {len(trainset)}")
    print(f"Test size: {len(testset)}")

    return trainset, testset


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    input_path: str = None,
    num_clients: int = 1,
    mode: str = "standard",
):
    if dataset_name not in UNIFIED_DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    entry = UNIFIED_DATASET_REGISTRY[dataset_name]
    dataset_type = entry["type"]

    print(f"Preparing dataset: {dataset_name} (type={dataset_type})")

    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == "vision":
        return prepare_builtin_dataset(dataset_name, output_dir)

    if dataset_name == "metropt":
        if input_path is None:
            raise ValueError("metropt requires 'input_path'")
        if num_clients < 1:
            raise ValueError("metropt requires num_clients >= 1")

        entry["prepare"](
            input_csv=input_path,
            output_dir=output_dir,
            num_clients=num_clients,
        )
        return

    if dataset_name == "psm":
        if input_path is None:
            raise ValueError("psm requires 'input_path' (directory containing train.csv, test.csv, test_label.csv)")
        if num_clients < 1:
            raise ValueError("psm requires num_clients >= 1")

        entry["prepare"](
            input_dir=input_path,
            output_dir=output_dir,
            num_clients=num_clients,
        )
        return

    if dataset_name == "tabular":
        if input_path is None:
            raise ValueError("tabular requires 'input_path'")
        if num_clients < 1:
            raise ValueError("tabular requires num_clients >= 1")

        entry["prepare"](
            input_csv=input_path,
            output_dir=output_dir,
            num_clients=num_clients,
        )
        return

    if dataset_name == "act":
        if input_path is None:
            raise ValueError("act requires 'input_path'")

        context = PreprocessingContext()

        if mode == "standard":
            pipeline = entry["prepare_pipeline"](input_path, output_dir)
        elif mode == "balanced":
            if num_clients < 1:
                raise ValueError("act balanced mode requires num_clients >= 1")
            pipeline = entry["prepare_balanced_pipeline"](
                input_path, output_dir, num_clients
            )
        elif mode == "anomaly":
            pipeline = entry["prepare_anomaly_pipeline"](input_path, output_dir)
        else:
            raise ValueError(f"Unsupported ACT mode: {mode}")

        pipeline.fit_transform(None, context)
        return

    raise ValueError(f"Unhandled dataset: {dataset_name}")
