import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import numpy as np

def prepare_metro_data(
    input_csv: str,
    output_dir: str,
    num_clients: int = 2,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
):
    """
    Prepares METRO dataset for FL anomaly detection with Autoencoder.
    
    Args:
        input_csv: path to METRO dataset CSV file
        output_dir: directory to save per-client data
        num_clients: number of FL clients
        val_ratio: fraction of training data for validation
        test_ratio: fraction of data for testing
        random_seed: random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    data = pd.read_csv(input_csv)
    data = data.values.astype(float)
    
    # Assume first column = label (0=normal,1=anomaly), rest = features
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    
    # Split normal vs anomaly
    normal_data = features[labels == 0]
    anomaly_data = features[labels == 1]
    
    # Train/validation split (only normal data for AE training)
    train_data, val_data = train_test_split(
        normal_data, test_size=val_ratio, random_state=random_seed
    )
    
    # Test set includes all data (normal + anomalies)
    test_data, test_labels = features, labels
    
    # Shuffle test data
    np.random.seed(random_seed)
    perm = np.random.permutation(len(test_data))
    test_data, test_labels = test_data[perm], test_labels[perm]
    
    # Split train/val/test across clients
    def split_for_clients(data_array):
        return np.array_split(data_array, num_clients)
    
    train_splits = split_for_clients(train_data)
    val_splits = split_for_clients(val_data)
    test_splits = split_for_clients(test_data)
    test_label_splits = split_for_clients(test_labels)
    
    # Save per-client data
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}")
        os.makedirs(client_dir, exist_ok=True)
        
        torch.save(torch.tensor(train_splits[i], dtype=torch.float32), os.path.join(client_dir, "train.pt"))
        torch.save(torch.tensor(val_splits[i], dtype=torch.float32), os.path.join(client_dir, "val.pt"))
        torch.save(
            (torch.tensor(test_splits[i], dtype=torch.float32),
             torch.tensor(test_label_splits[i], dtype=torch.int64)),
            os.path.join(client_dir, "test.pt")
        )
        
        print(f"[Client {i+1}] train: {train_splits[i].shape}, val: {val_splits[i].shape}, test: {test_splits[i].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare METRO dataset for AE FL clients")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to METRO CSV file")
    parser.add_argument("--output_dir", type=str, default="./data/metro_clients", help="Output directory for client data")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of FL clients")
    args = parser.parse_args()
    
    prepare_metro_data(args.input_csv, args.output_dir, args.num_clients)

