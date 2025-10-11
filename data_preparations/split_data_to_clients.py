import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


def prepare_data_for_clients(
    input_csv: str,
    output_dir: str,
    num_clients: int = 2,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    label_col: int | None = 0,
    random_seed: int = 42,
):
    """
    Splits any tabular dataset into per-client subsets for Federated Learning.

    Args:
        input_csv: Path to the CSV dataset.
        output_dir: Directory to save client data.
        num_clients: Number of FL clients.
        val_ratio: Fraction of training data for validation.
        test_ratio: Fraction of total data for testing.
        label_col: Index of label column (0-based). Set to None if there are no labels.
        random_seed: Random seed for reproducibility.
    """

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

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
        print("No label column specified — treating all columns as features.")

    # Split into train/val/test
    train_data, test_data, train_labels, test_labels = train_test_split(
        features,
        labels if has_labels else np.zeros(len(features)),
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels if has_labels else None,
    )

    # Further split train → train/val
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

    # Split each subset evenly across clients
    def split_for_clients(data_array):
        return np.array_split(data_array, num_clients)

    train_splits = split_for_clients(train_data)
    val_splits = split_for_clients(val_data)
    test_splits = split_for_clients(test_data)

    label_splits_train = split_for_clients(train_labels)
    label_splits_val = split_for_clients(val_labels)
    label_splits_test = split_for_clients(test_labels)

    # Save data for each client
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}")
        os.makedirs(client_dir, exist_ok=True)

        # Save train
        if has_labels:
            torch.save(
                (torch.tensor(train_splits[i], dtype=torch.float32),
                 torch.tensor(label_splits_train[i], dtype=torch.long)),
                os.path.join(client_dir, "train.pt"),
            )
        else:
            torch.save(torch.tensor(train_splits[i], dtype=torch.float32),
                       os.path.join(client_dir, "train.pt"))

        # Save val
        if len(val_splits[i]) > 0:
            if has_labels:
                torch.save(
                    (torch.tensor(val_splits[i], dtype=torch.float32),
                     torch.tensor(label_splits_val[i], dtype=torch.long)),
                    os.path.join(client_dir, "val.pt"),
                )
            else:
                torch.save(torch.tensor(val_splits[i], dtype=torch.float32),
                           os.path.join(client_dir, "val.pt"))

        # Save test
        if has_labels:
            torch.save(
                (torch.tensor(test_splits[i], dtype=torch.float32),
                 torch.tensor(label_splits_test[i], dtype=torch.long)),
                os.path.join(client_dir, "test.pt"),
            )
        else:
            torch.save(torch.tensor(test_splits[i], dtype=torch.float32),
                       os.path.join(client_dir, "test.pt"))

        print(f"[Client {i+1}] train: {train_splits[i].shape}, "
              f"val: {val_splits[i].shape}, test: {test_splits[i].shape}")

    print(f"\nData successfully split into {num_clients} clients and saved under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split any tabular dataset for FL clients")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./data/clients", help="Output directory for client data")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--label_col", type=int, default=0, help="Index of label column (0-based). Set -1 if none.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    label_column = None if args.label_col < 0 else args.label_col

    prepare_data_for_clients(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        label_col=label_column,
        random_seed=args.seed,
    )

