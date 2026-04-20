import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_preprocessing.config_loader import load_json_config


DEFAULT_METRO_AE_SCHEMA_PATH = "data_preprocessing/configs/metro_ae_schema.json"


def load_metro_ae_schema(config_path=DEFAULT_METRO_AE_SCHEMA_PATH):
    return load_json_config(config_path)


def prepare_metro_ae_data(
    input_csv: str,
    output_dir: str,
    num_clients: int = None,
    config_path: str = DEFAULT_METRO_AE_SCHEMA_PATH,
):
    schema = load_metro_ae_schema(config_path)

    val_ratio = schema["split"]["val_ratio"]
    random_seed = schema["split"]["random_seed"]
    normal_value = schema["columns"]["normal_value"]
    anomaly_value = schema["columns"]["anomaly_value"]

    if num_clients is None:
        num_clients = schema["partition"]["num_clients"]

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_csv)

    # Assume first column = label, remaining columns = features
    labels = df.iloc[:, 0].astype(int).to_numpy()
    features = df.iloc[:, 1:].astype(float).to_numpy()

    # Split normal vs anomaly
    normal_data = features[labels == normal_value]
    anomaly_data = features[labels == anomaly_value]

    # Train/validation split only on normal data
    train_data, val_data = train_test_split(
        normal_data,
        test_size=val_ratio,
        random_state=random_seed,
    )

    # Test uses full dataset
    test_data = features
    test_labels = labels

    # Shuffle test data
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(test_data))
    test_data = test_data[perm]
    test_labels = test_labels[perm]

    # Split arrays across clients
    train_splits = np.array_split(train_data, num_clients)
    val_splits = np.array_split(val_data, num_clients)
    test_splits = np.array_split(test_data, num_clients)
    test_label_splits = np.array_split(test_labels, num_clients)

    # Save per-client tensors
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}")
        os.makedirs(client_dir, exist_ok=True)

        torch.save(
            torch.tensor(train_splits[i], dtype=torch.float32),
            os.path.join(client_dir, "train.pt"),
        )
        torch.save(
            torch.tensor(val_splits[i], dtype=torch.float32),
            os.path.join(client_dir, "val.pt"),
        )
        torch.save(
            (
                torch.tensor(test_splits[i], dtype=torch.float32),
                torch.tensor(test_label_splits[i], dtype=torch.int64),
            ),
            os.path.join(client_dir, "test.pt"),
        )

        print(
            f"[Client {i+1}] "
            f"train: {train_splits[i].shape}, "
            f"val: {val_splits[i].shape}, "
            f"test: {test_splits[i].shape}"
        )
