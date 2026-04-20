import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing.config_loader import load_json_config


DEFAULT_PSM_SCHEMA_PATH = "data_preprocessing/configs/psm_schema.json"


def load_psm_schema(config_path=DEFAULT_PSM_SCHEMA_PATH):
    return load_json_config(config_path)


def prepare_psm_data(
    input_dir: str,
    output_dir: str,
    num_clients: int = None,
    config_path: str = DEFAULT_PSM_SCHEMA_PATH,
):
    schema = load_psm_schema(config_path)

    train_file = schema["input"]["train_file"]
    test_file = schema["input"]["test_file"]
    test_label_file = schema["input"]["test_label_file"]
    drop_first_column = schema["input"]["drop_first_column"]

    val_ratio = schema["split"]["val_ratio"]
    random_seed = schema["split"]["random_seed"]

    if num_clients is None:
        num_clients = schema["partition"]["num_clients"]

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(input_dir, train_file))
    test_df = pd.read_csv(os.path.join(input_dir, test_file))
    test_label_df = pd.read_csv(os.path.join(input_dir, test_label_file))

    if drop_first_column:
        train_data = train_df.values[:, 1:]
        test_data = test_df.values[:, 1:]
        test_labels = test_label_df.values[:, 1:]
    else:
        train_data = train_df.values
        test_data = test_df.values
        test_labels = test_label_df.values

    train_data = np.nan_to_num(train_data).astype(float)
    test_data = np.nan_to_num(test_data).astype(float)
    test_labels = np.nan_to_num(test_labels).astype(float)

    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_split, val_split = train_test_split(
        train_data,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    train_splits = np.array_split(train_split, num_clients)
    val_splits = np.array_split(val_split, num_clients)

    # keep full test set for every client
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}")
        os.makedirs(client_dir, exist_ok=True)

        pd.DataFrame(train_splits[i]).to_csv(
            os.path.join(client_dir, "train.csv"), index=False
        )
        pd.DataFrame(val_splits[i]).to_csv(
            os.path.join(client_dir, "val.csv"), index=False
        )
        pd.DataFrame(test_data).to_csv(
            os.path.join(client_dir, "test.csv"), index=False
        )
        pd.DataFrame(test_labels).to_csv(
            os.path.join(client_dir, "test_label.csv"), index=False
        )

        print(
            f"[Client {i+1}] "
            f"train: {train_splits[i].shape}, "
            f"val: {val_splits[i].shape}, "
            f"test: {test_data.shape}, "
            f"test_labels: {test_labels.shape}"
        )
