import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def prepare_tabular_fl_data(
    input_csv: str,
    output_dir: str,
    num_clients: int = 2,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    label_col: int | None = 0,
    random_seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Load dataset
    df = pd.read_csv(input_csv)
    data = df.values.astype(float)

    # Separate features and labels
    if label_col is not None and 0 <= label_col < data.shape[1]:
        labels = data[:, label_col]
        features = np.delete(data, label_col, axis=1)
        has_labels = True
    else:
        features = data
        labels = None
        has_labels = False

    # Train/test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
        labels if has_labels else np.zeros(len(features)),
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels if has_labels else None,
    )

    # Train/val split
    if val_ratio > 0:
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data,
            train_labels,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=train_labels if has_labels else None,
        )
    else:
        val_data, val_labels = np.empty((0, features.shape[1])), np.empty((0,))

    # Split across clients
    def split(arr):
        return np.array_split(arr, num_clients)

    train_splits = split(train_data)
    val_splits = split(val_data)
    test_splits = split(test_data)

    train_label_splits = split(train_labels)
    val_label_splits = split(val_labels)
    test_label_splits = split(test_labels)

    # Save
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}")
        os.makedirs(client_dir, exist_ok=True)

        def save(split_name, data_split, label_split):
            if has_labels:
                torch.save(
                    (
                        torch.tensor(data_split, dtype=torch.float32),
                        torch.tensor(label_split, dtype=torch.long),
                    ),
                    os.path.join(client_dir, f"{split_name}.pt"),
                )
            else:
                torch.save(
                    torch.tensor(data_split, dtype=torch.float32),
                    os.path.join(client_dir, f"{split_name}.pt"),
                )

        save("train", train_splits[i], train_label_splits[i])
        save("val", val_splits[i], val_label_splits[i])
        save("test", test_splits[i], test_label_splits[i])

        print(
            f"[Client {i+1}] "
            f"train: {train_splits[i].shape}, "
            f"val: {val_splits[i].shape}, "
            f"test: {test_splits[i].shape}"
        )
